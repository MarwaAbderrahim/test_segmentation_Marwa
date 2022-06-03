
import nibabel
import numpy as np
import numpy
import nibabel as nib
import os
import argparse

parser = argparse.ArgumentParser(description='Transform segmented mask')
parser.add_argument("-i", "--input", type=str, required=True, help="path to input image")
parser.add_argument("-o", "--output", type=str, help="path to output mask")


def main():
    
    args = parser.parse_args()

    image = os.path.join(args.input)
    data = nibabel.load(image).get_fdata().astype(int)
    data_multiplied = data * np.arange(data.shape[-1])
    data_labels = data_multiplied.sum(axis=-1)
    
    converted_array = numpy.array(data_labels, dtype=numpy.int32) 
    nifti_file = nibabel.Nifti1Image(converted_array, None)
    nibabel.save(nifti_file, os.path.join(args.output))  


if __name__ == "__main__":
    main()  