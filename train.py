from monai.utils import first, set_determinism
import time
import csv
from pytorchtools import EarlyStopping
from monai.transforms import (
    AddChanneld,
    AsDiscrete,
    AsDiscreted,
    EnsureChannelFirstd,
    Compose,
    Resized,
    CropForegroundd,
    RandScaleIntensityd,
    DataStatsd,
    LoadImaged,
    Orientationd,
    Activationsd,
    RandCropByPosNegLabeld,
    RandShiftIntensityd,
    NormalizeIntensityd,
    RandSpatialCropd,
    ScaleIntensityRanged,
    Spacingd,
    ScaleIntensityd,
    SaveImaged,
    EnsureTyped,
    EnsureType,
    Invertd,
)
from monai.handlers.utils import from_engine
from monai.losses import DiceCELoss
from monai.networks.nets import UNet
from monai.networks.layers import Norm
from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference
from monai.data import CacheDataset, DataLoader, decollate_batch
from monai.config import print_config
from monai.apps import download_and_extract
import torch
from torch.utils.data import Dataset
import pandas as pd
import matplotlib.pyplot as plt
import monai
import logging
import os
import glob
import sys
import numpy as np


print_config()


root_dir = "E:/Data"
csv_path = "E:/Data/train_data.csv" 
image_folder = "E:/Data/CTPelvic1K_dataset6_data"
mask_folder = "E:/Data/ipcai2021_dataset6_Anonymized"

def main(root_dir):
    monai.config.print_config()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    set_determinism(seed=0)

    #Data Loading using csv file
    class CustomDataset(Dataset):
        def __init__(self, csv_path, image_folder, mask_folder):
            self.data = pd.read_csv(csv_path)
            self.image_folder=image_folder
            self.mask_folder=mask_folder
            return
        
        def __len__(self):
            return len(self.data)
            
        def __getitem__(self, idx):
            if torch.is_tensor(idx):
                idx = idx.tolist()
            image = self.image_folder + "/" + self.data['images'][idx]
            label = self.mask_folder + "/" + self.data['masks'][idx]
            sample = [{"image": image, "mask": label} for image, label in zip(image, label)]
            return sample
    
    

    data_dicts = CustomDataset(csv_path, image_folder, mask_folder)
    train_files, val_files = data_dicts[:-10], data_dicts[-10:]
    
    #Formula to compute minimum/maximum scale internsity based on known Wide window and Window level. For bones L=500 and W=2000

    L=500
    W=2000
    i_min = L - (W/2)
    i_max =  L + (W/2)

    # define transforms to augment the dataset
    train_transforms = Compose(
        [
            LoadImaged(keys=["image", "mask"]),   #Dictionary-based wrapper which can load both images and labels
            EnsureChannelFirstd(keys=["image", "mask"]), #Ensures the channel first input for both images and labels
            Orientationd(keys=["image", "mask"], axcodes="RAS"), #Reorienting the input array
            Spacingd(keys=["image", "mask"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")), #Resampling the input array, interpolation mode to calculate output values.
            ScaleIntensityRanged(keys=["image"], a_min=i_min, a_max=i_max,b_min=0.0, b_max=4.0, clip=True,), #Scale min/max intensity ranges for image and mask, clip after scaling.
            CropForegroundd(keys=["image", "mask"], source_key="image"), #Crops only the foreground object of the expected images. 
            
            #Crop random fixed sized regions with the center being a foreground or background voxel based on the Pos Neg Ratio. 
            RandCropByPosNegLabeld(   
                keys=["image", "mask"],
                label_key="mask",
                spatial_size=(128, 128, 128),
                pos=1,
                neg=1,
                num_samples=4,
                image_key="image",
                image_threshold=0,
        ),
            EnsureTyped(keys=["image", "mask"]), #Ensure the input data to be a PyTorch Tensor or numpy array
        ]
    )



    val_transforms = Compose(
    [
        LoadImaged(keys=["image", "mask"]),  #Dictionary-based wrapper which can load both images and labels
        EnsureChannelFirstd(keys=["image", "mask"]), #Ensures the channel first input for both images and labels
        Orientationd(keys=["image", "mask"], axcodes="RAS"), #Reorienting the input array
        Spacingd(keys=["image", "mask"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")), #Resampling the input array, interpolation mode to calculate output values.
        ScaleIntensityRanged(keys=["image"], a_min=i_min, a_max=i_max, b_min=0.0, b_max=4.0, clip=True,), #Scale min/max intensity ranges for image and mask, clip after scaling
        CropForegroundd(keys=["image", "mask"], source_key="image"),  #Crops only the foreground object of the expected images. 
        
        #Crop random fixed sized regions with the center being a foreground or background voxel based on the Pos Neg Ratio. 
        RandCropByPosNegLabeld(
            keys=["image", "mask"],
            label_key="mask",
            spatial_size=(128, 128, 128),
            pos=1,
            neg=1,
            num_samples=4,
            image_key="image",
            image_threshold=0,),
            
        
        EnsureTyped(keys=["image", "mask"]), #Ensure the input data to be a PyTorch Tensor or numpy array
    ]
)


    # create a training data loader
    train_ds = CacheDataset(data=train_files, transform=train_transforms, cache_rate=1.0, num_workers=0) #CacheDataset will iterate over the input_dataset, and store tensors. 
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=0)    

    # create a validation data loader
    val_ds = CacheDataset(data=val_files, transform=val_transforms, cache_rate=1.0, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=0)
    
    # create UNet, DiceCELoss and Adam optimizer  
    device=torch.device("cuda:0")  
    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=5,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    ).to(device)
    loss_function = DiceCELoss(to_onehot_y=True, softmax=True)  #DiceLoss was also tested, the results were better in case of DiceCELoss
    optimizer = torch.optim.Adam(model.parameters(), 2e-4)    #Novograd optimizer was alos tested, the results were better in case of Adam optimizer

    dice_metric = DiceMetric(include_background=False, reduction="mean")
    

    # use amp to accelerate training
    scaler = torch.cuda.amp.GradScaler()
    # enable cuDNN benchmark
    torch.backends.cudnn.benchmark = True    


    max_epochs = 400
    val_interval = 1

    train_values = []
    epoch_loss_values = []
    metric_values = []

    post_pred = Compose([EnsureType(), AsDiscrete(argmax=True, to_onehot=5)])
    post_label = Compose([EnsureType(), AsDiscrete(to_onehot=5)])


    #Early stopping with patience (number of epochs to wait if there is no increase of validation mean dice)
    early_stopping = EarlyStopping(patience=80, verbose=True, delta=0.01)  #EarlyStopping class is imported from pytorchtools.py file


    for epoch in range(max_epochs):
        model.train()
        epoch_loss = 0
        step = 0
        for batch_data in train_loader:
            step += 1
            inputs, labels = batch_data["image"].to(device), batch_data["mask"].to(device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = loss_function(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        
        epoch_loss += loss.item()
        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)


        if (epoch + 1) % val_interval == 0:
            model.eval()
            with torch.no_grad():
                for val_data in val_loader:
                    val_inputs, val_labels = (val_data["image"].to(device),val_data["mask"].to(device),)
                    val_outputs = model(val_inputs)
                    val_outputs = [post_pred(i) for i in decollate_batch(val_outputs)] #Excute the decollate batch logic for engine.state.batch and engine.state.output.
                    val_labels = [post_label(i) for i in decollate_batch(val_labels)]  #Excute the decollate batch logic for engine.state.batch and engine.state.output.
                
                    # compute metric for current iteration
                    dice_metric(y_pred=val_outputs, y=val_labels)

                metric = dice_metric.aggregate().item()
                dice_metric.reset()
                metric_values.append(metric)
            

            model.eval()
            with torch.no_grad():
                for train_data in train_loader:
                    train_inputs, train_labels = (train_data["image"].to(device),train_data["mask"].to(device),)
                    train_outputs = model(train_inputs) 
                    train_outputs = [post_pred(i) for i in decollate_batch(train_outputs)] #Excute the decollate batch logic for engine.state.batch and engine.state.output.
                    train_labels = [post_label(i) for i in decollate_batch(train_labels)] #Excute the decollate batch logic for engine.state.batch and engine.state.output.
                    dice_metric(y_pred=train_outputs, y=train_labels)

                train_metric = dice_metric.aggregate().item()
                dice_metric.reset()
                train_values.append(train_metric)
        
            
        epoch_len = len(str(max_epochs))
    
        print_msg = (f'[{epoch:>{epoch_len}}/{max_epochs:>{epoch_len}}] ' +
                     f'Validation Mean Dice: {metric:.5f} ')

        print(print_msg)

        #You can open these two lines if you want to save the validation mean dice on csv file.
        array = np.array(metric_values)
        np.savetxt('E:/Data/train_dicescore.csv', array, delimiter=',')  
    
        early_stopping(metric, model)   #the training should stop on epoch 152 with a mean dice score of 0.9454
        
        if early_stopping.early_stop:
             print("Early stopping")
             break
        
    torch.save(model.state_dict(), os.path.join(root_dir, "best_metric_model.pth"))    #saving the trained model
    

    # visualize the mean cdice when the network trained 
    fig = plt.figure(figsize=(10,8))
    plt.plot(range(1,len(metric_values)+1), metric_values, label='Validation Mean Dice')
    plt.plot(range(1,len(train_values)+1), train_values, label='Training Mean Dice')


    # find position of the highest validation mean dice
    maxposs = metric_values.index(max(metric_values))+1 
    plt.axvline(maxposs, linestyle='--', color='r',label='Early Stopping Checkpoint')

    plt.xlabel('Epochs')
    plt.ylabel('Mean Dice')
    plt.ylim(0, 1.0) 
    plt.xlim(0, len(train_values)+1) 
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    fig.savefig('meandice_plot.png', bbox_inches='tight')   


if __name__ == "__main__":
    main(root_dir)     