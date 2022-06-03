import logging
import os
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader
import glob
import time
import monai
from monai.handlers.utils import from_engine
from monai.data import CacheDataset, DataLoader, decollate_batch
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from torch.utils.data import Dataset
import pandas as pd
from monai.networks.nets import UNet
from monai.transforms import (
    AsDiscreted,
    EnsureChannelFirstd,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    ScaleIntensityRanged,
    Spacingd,
    EnsureTyped,
    Invertd,
)

root_dir = "E:/Data"
csv_path = "E:/Data/test_data.csv"
image_folder = "E:/Data/CTPelvic1K_dataset6_data"
mask_folder = "E:/Data/ipcai2021_dataset6_Anonymized"



def main(root_dir):
    monai.config.print_config()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

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
    val_files = data_dicts[:]

    #Formula to compute minimum/maximum scale internsity based on known Wide window and Window level. For bones L=500 and W=2000

    L=500
    W=2000
    i_min = L - (W/2)   #output  -500
    i_max =  L + (W/2)  #output  1500


    # define pre transforms
    
    val_transforms = Compose([
        LoadImaged(keys=["image", "mask"]), #Dictionary-based wrapper which can load both images and labels
        EnsureChannelFirstd(keys=["image", "mask"]), #Ensures the channel first input for both images and labels
        Orientationd(keys=["image"], axcodes="RAS"), #Reorienting the input array
        Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode="bilinear"), #Resampling the input array, interpolation mode to calculate output values.
        ScaleIntensityRanged(keys=["image"], a_min=i_min, a_max=i_max,b_min=0.0, b_max=4.0, clip=True,), #Scale min/max intensity ranges for image and mask, clip after scaling
        CropForegroundd(keys=["image"], source_key="image"),  #Crops only the foreground object of the expected images. 
        EnsureTyped(keys=["image", "mask"]), #Ensure the input data to be a PyTorch Tensor or numpy array
    ])
    
   
    #Data Loading and augmentation
    val_ds = CacheDataset(data=val_files, transform=val_transforms, cache_rate=1.0, num_workers=0) #CacheDataset will iterate over the input_dataset, and store tensors. 
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=0)
    
    #Define a DiceMetric
    dice_metric = DiceMetric(include_background=False, reduction="mean")

    
    # define post transforms
    post_transforms = Compose([
        EnsureTyped(keys="pred"),  #Ensure the input data to be a PyTorch Tensor or numpy array
        Invertd(keys="pred", transform=val_transforms, orig_keys="image", meta_keys="pred_meta_dict", 
            orig_meta_keys="image_meta_dict", meta_key_postfix="meta_dict", nearest_interp=False,to_tensor=True,),  #Utility transform to automatically invert the previously applied transforms.
        AsDiscreted(keys="pred", argmax=True, to_onehot=5),
        AsDiscreted(keys="mask", to_onehot=5),
    ])
    
    
    #create 3D UNet architecture
    device = torch.device("cuda:0")   #define the device which should be either GPU or CPU
    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=5,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    ).to(device)


    #loading the saved model of previously trained 3D U-Net model
    model.load_state_dict(torch.load(os.path.join(root_dir, "best_metric_model.pth")))

    
    #evaluation of the model by checking the dice score for testing cases
    model.eval()
    start_time = time.time()
    with torch.no_grad():
        for val_data in val_loader:
            val_inputs = val_data["image"].to(device)
            roi_size = (160, 160, 160)
            sw_batch_size = 4

            val_data["pred"] = sliding_window_inference(val_inputs, roi_size, sw_batch_size, model,sw_device="cuda:0", device="cpu") #Sliding window inference on inputs with predictor.
       
            val_data = [post_transforms(i) for i in decollate_batch(val_data)] #Excute the decollate batch logic for engine.state.batch and engine.state.output.
            val_outputs, val_labels = from_engine(["pred", "mask"])(val_data)    
        
            
            # compute metric for current iteration
            dice_metric(y_pred=val_outputs, y=val_labels)

        # aggregate the final mean dice result
        metric_test = dice_metric.aggregate().item()
        # reset the status for next validation round
        # dice_metric.reset()
        
        #You can open these two lines if you want to save dice scores on csv file.
        array = np.array(dice_metric.get_buffer())
        np.savetxt('E:/Data/test_dicescore.csv', array, delimiter=',')      
            
        
    #print("Metric on original image spacing: ", metric_test)
    print (dice_metric.get_buffer())                              #"get_buffer" helps to see dice score of each labels for each case
    print("Avarage Metric: ", metric_test)   
    print("Total time seconds: {:.2f}".format((time.time()- start_time)))


if __name__ == "__main__":
    main(root_dir)





