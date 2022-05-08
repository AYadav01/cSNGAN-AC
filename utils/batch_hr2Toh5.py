import argparse
import sys
import os
from os import listdir
import glob
import logging
import struct
from tqdm import tqdm
import numpy as np
import zlib
import json
import h5py
import nrrd
import scipy.ndimage
import pickle

parser = argparse.ArgumentParser(description="Read and convert hr2 to h5")
parser.add_argument("--input", type=str, default='./', help="input folder")
parser.add_argument("--output", type=str, default='./', help="output folder")
parser.add_argument("--d", type=int, default=10, help="dose level")
parser.add_argument("--k", type=int, default=1, help="kernel")
parser.add_argument("--st", type=float, default=1.0, help="slice thickness")
parser.add_argument("-n", action='store_true', default=False, 
                            help="store as nrrd as well for easy visualization")
parser.add_argument("-resample", action='store_true', default=False, 
                            help="resample to a new spacing")
parser.add_argument("--new_xy", type=float, default=0.7, help="resample xy spacing")


def usage():
    print(
        """
        Usage: python batch_read_hr2.py --input=/path/to/input_folder 
                                        --output=/path/to/output_folder 
                                        --d=100 --k=1 --st=0.6
        Output is a h5 file and its orginal parameter file .json
        By Leihao Wei
        """
    )
    sys.exit()


def read_hr2(filepath):

    hr2_dict = {}

    with open(filepath, 'rb') as f:

        # Ensure we have an HR2 file
        magic_number = f.read(3)
        if magic_number != b"HR2":
            #print("File is not an hr2 file. Exiting")
            sys.exit(1)

        # Read the file into memory
        while True:
            # Get tag
            chunk_tag_size = int.from_bytes(f.read(1), byteorder='little')
            chunk_tag = f.read(chunk_tag_size)
            # print(str(chunk_tag,'utf-8'))

            # Get value
            # Handle all tags *except* image data (size=uint16)
            if chunk_tag != b'ImageData':
                chunk_val_size = int.from_bytes(f.read(2), byteorder='little')
                chunk_val = f.read(chunk_val_size)
                hr2_dict[str(chunk_tag, 'utf-8')] = str(chunk_val, 'utf-8')
            # Handle image data tag (size=uint32)
            else:
                chunk_val_size = int.from_bytes(f.read(4), byteorder='little')
                header_end_byte = f.tell()
                chunk_val = f.read(chunk_val_size)
                hr2_dict[str(chunk_tag, 'utf-8')] = chunk_val
                break

        # Decompress the image data using zlib
        if hr2_dict['Compression'] == "ZLib":
            hr2_dict['ImageData'] = zlib.decompress(hr2_dict['ImageData'])

        # Parse image data byte string into numpy array
        hr2_dict['Size'] = [int(x) for x in hr2_dict['Size'].split(' ')]
        hr2_dict['ImageData'] = np.frombuffer(
            hr2_dict['ImageData'], dtype='int16')
        hr2_dict['ImageData'] = hr2_dict['ImageData'].reshape(
            hr2_dict['Size'][2], hr2_dict['Size'][1], hr2_dict['Size'][0])
        
        return hr2_dict
    

def resample(image, old_spacing, new_xy=0.7):
    # Determine current pixel spacing [W H D]
    spacing = map(float, old_spacing.split())
    spacing = np.array(list(spacing))

    if not opt.resample:
        return image, image.shape, spacing

    new_spacing = [new_xy, new_xy, spacing[-1]]
    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor[[2,0,1]]
    new_shape = np.round(new_real_shape) # [D H W]
    real_resize_factor = new_shape / image.shape 
    new_spacing = spacing / real_resize_factor[[2,1,0]] #[W H D]
    
    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor)
    
    return image, new_shape, new_spacing


def writeH5(label, data_dict, outpath, lastIndex=-1):    
    pixel_data, new_shape, new_spacing = resample(
        data_dict['ImageData'][:lastIndex], data_dict['Spacing'], opt.new_xy)

    pixel_data = np.clip(pixel_data, -1000, None)
    pixel_data += 1000
    
    # write h5
    with h5py.File(outpath, 'w') as outfile:
        dset = outfile.create_dataset(label, data=pixel_data, 
                shape=(new_shape[0],new_shape[1], new_shape[2]), 
                dtype='int16',  chunks=(16, 64, 64), 
                maxshape=(None, new_shape[1], new_shape[2]))
    # write nrrd
    if opt.n:
        nrrd_header = {'units': ['mm', 'mm', 'mm'], 'spacings':new_spacing}
        nrrd.write(outpath.replace('.h5','.nrrd'), pixel_data, nrrd_header, index_order='C')

def main(argc, argv):
    global opt
    opt = parser.parse_args()

    # Parse CLI inputs
    if argc < 5:
        usage()
        sys.exit(1)

    input_folder = str(opt.input) + "/" + str(opt.d)
    output_folder = str(opt.output) + "/k{}_d{}_st{:2.1f}".format(opt.k, opt.d, opt.st)

    if not os.path.exists(input_folder):
        print("check your input folder path")
        print(input_folder)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    casePaths = glob.glob(
        input_folder + '/*_k{}_st{:2.1f}'.format(opt.k, opt.st))

    # read last good index from .json
    with open("index_dict_st{:2.1f}.json".format(opt.st), "r") as f:
        last_index_dict = json.load(f)

    for casePath in tqdm(casePaths):

        # read the hr2 file name
        infilename = os.path.splitext(listdir(casePath + "/img/")[0])[0]
        input_filepath = casePath + "/img/" + infilename + '.hr2'

        outfilename=os.path.splitext(os.path.basename(infilename))[0].split("_", 1)[0]
        output_filepath = output_folder + '/' + outfilename + '.h5'

        # check if the file exists
        if not os.path.exists(output_filepath):
            # Read the HR2 file into a dictionary
            hr2_dict = read_hr2(input_filepath)
            # Write image data to disk
            writeH5('data', hr2_dict,  output_filepath, last_index_dict[outfilename])
            
        # Write header as json to disk for easy repackaging
        output_dirpath = os.path.dirname(output_filepath)
        hr2_header_output_filepath = os.path.splitext(
            os.path.basename(output_filepath))[0] + ".json"
        del hr2_dict['ImageData']
        with open(os.path.join(output_dirpath, hr2_header_output_filepath), 'w') as f:
            json.dump(hr2_dict, f)
        
    print('Done converting HR2 to h5 of int16')

if __name__ == "__main__":
    main(len(sys.argv), sys.argv)
