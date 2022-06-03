import logging
import os
import sys
import glob
from monai.transforms import LoadImage
import matplotlib.pyplot as plt
import time
import monai
import numpy as np
import torch
from monai.handlers.utils import from_engine
from monai.visualize import blend_images
import argparse
import sys
import os
from monai.config import print_config
from monai.data import CacheDataset, DataLoader, Dataset, decollate_batch
from monai.inferers import sliding_window_inference
from monai.networks.nets import UNet
from monai.transforms import (
    AsDiscreted,
    Compose,
    EnsureChannelFirstd,
    Invertd,
    LoadImaged,
    ScaleIntensityRanged,
    Spacingd,
    CropForegroundd,
    Orientationd,
    SaveImaged,
    EnsureTyped,
)

#Using argparse module which makes it easy to write user-friendly command-line interfaces.
parser = argparse.ArgumentParser(description='Predict masks from input images')
parser.add_argument("-i", "--input", type=str, required=True, help="path to input image")    #input CT image we can call by "-i" command
parser.add_argument("-o", "--output", type=str, help="path to output mask")                  #output segmented mask we can call by "-o" command


def main():
    #print_config()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    args = parser.parse_args()
    
    #The path to the folder where the trained model is saved
    model_dir  = "E:/Data"

    #The path of the image that will be used for the segmentation, user of the code can chose an image using command-line.
    test_images = sorted(glob.glob(os.path.join(args.input)))
    test_dicts = [{"image": image_name} for image_name in test_images]
    files = test_dicts[:]

    #Formula to compute minimum/maximum scale internsity based on known Wide window and Window level. For bones L=500 and W=2000

    L=500
    W=2000
    i_min = L - (W/2)   #output  -500
    i_max =  L + (W/2)  #output  1500


    # define pre transforms
    pre_transforms = Compose([
        LoadImaged(keys="image"),  #Dictionary-based wrapper which can load both images and labels
        EnsureChannelFirstd(keys="image"), #Ensures the channel first input for both images and labels
        Orientationd(keys=["image"], axcodes="RAS"), #Reorienting the input array
        Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode="bilinear"), #Resampling the input array, interpolation mode to calculate output values.
        ScaleIntensityRanged(keys=["image"], a_min=i_min, a_max=i_max,b_min=0.0, b_max=4.0, clip=True,), #Scale min/max intensity ranges for image and mask, clip after scaling
        CropForegroundd(keys=["image"], source_key="image"),  #Crops only the foreground object of the expected images. 
        EnsureTyped(keys="image"), #Ensure the input data to be a PyTorch Tensor or numpy array
        ])
    

    #Data Loading and augmentation
    dataset = CacheDataset(data=files, transform=pre_transforms, cache_rate=1.0, num_workers=0) #CacheDataset will iterate over the input_dataset, and store tensors. 
    test_loader = DataLoader(dataset, batch_size=1, num_workers=0)
    
    
    # define post transforms
    post_transforms = Compose([
        EnsureTyped(keys="pred"),   #Ensures the input data to be a PyTorch Tensor or numpy array
        Invertd(                    #Utility transform to automatically invert the previously applied transforms.
            keys="pred",
            transform=pre_transforms,
            orig_keys="image",
            meta_keys="pred_meta_dict",
            orig_meta_keys="image_meta_dict",
            meta_key_postfix="meta_dict",
            nearest_interp=False,
            to_tensor=True,
            ),
        
        AsDiscreted(keys="pred", argmax=True, to_onehot=5),
        SaveImaged(keys="pred", meta_keys="pred_meta_dict", output_dir="./out_seg", output_postfix="seg", resample=False),

    ])

    #create 3D UNet architecture
    device = torch.device("cuda:0")  #define the device which should be either GPU or CPU
    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=5,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    ).to(device)
    

    #loading the saved model of previously trained 3D U-Net model
    model.load_state_dict(torch.load(os.path.join(model_dir, "best_metric_model.pth"))) 

    
    #evaluation of the model on inference mode, upload one CT image and as an outcome we get its segmented mask 
    model.eval()
    with torch.no_grad():
        for test_data in test_loader:
            start_time = time.time()
            test_inputs = test_data["image"].to(device)
            roi_size = (160, 160, 160)
            sw_batch_size = 4
            test_data["pred"] = sliding_window_inference(test_inputs, roi_size, sw_batch_size, model) #Sliding window inference on inputs with predictor.
            test_data = [post_transforms(i) for i in decollate_batch(test_data)] #Excute the decollate batch logic for engine.state.batch and engine.state.output. 
            test_outputs = from_engine(["pred"])(test_data)
            
            loader = LoadImage()
            original_image = loader(test_data[0]["image_meta_dict"]["filename_or_obj"])[0]
            test_output_argmax = torch.argmax(test_outputs[0], dim=0, keepdim=True)
            
            rety = blend_images(image=original_image[None], label=test_output_argmax, cmap="jet", alpha=0.5, rescale_arrays=True) #Blend an image and a label.
            
            print("Total time seconds: {:.2f}".format((time.time()- start_time)))
    
            
            fig, (ax1, ax2, ax3) = plt.subplots(figsize=(18, 6), ncols=3)
            ax1.title.set_text('Image')
            ax2.title.set_text('Predicted Mask')
            ax3.title.set_text('Segmented Image')
            bar1 = ax1.imshow(original_image[None][ 0, :, :, 250], cmap="gray")
            fig.colorbar(bar1, ax=ax1)
            bar2 = ax2.imshow(test_output_argmax [ 0, :, :, 250],  cmap="jet")
            fig.colorbar(bar2, ax=ax2)
            ax3.imshow(torch.moveaxis(rety[:, :, :, 250], 0, -1))
            fig.colorbar(bar1, ax=ax3)
            plt.show()
            fig.savefig('segmentation.png', bbox_inches='tight')  #the visualization will be saved in the same folder where is your predict.py file.

            
    
if __name__ == '__main__':
    main()









