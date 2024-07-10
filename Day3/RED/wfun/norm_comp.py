from PIL import Image, ImageDraw, ImageFont
import shutil
import os

import numpy as np
import astropy.io.fits as fits

#make a folder called 'wfun'
def mkdir(path):
    path = path.strip()
    path = path.rstrip("\\")
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)
        return True
    else:
        return False
def extract_name(file_name):
    return file_name.split('.')[0]

def combine_images_grid(image_paths, output_path, grid_size):
    images = [Image.open(path) for path in image_paths]
    width, height = images[0].size
    
    # Calculate the size of the combined image
    combined_width = width * grid_size[0]
    combined_height = height * grid_size[1]
    
    combined_image = Image.new('RGB', (combined_width, combined_height))
    
    for index, image in enumerate(images):
        x_offset = (index % grid_size[0]) * width
        y_offset = (index // grid_size[0]) * height
        combined_image.paste(image, (x_offset, y_offset))
    
    combined_image.save(output_path)

def scan_picture(source_folder, extension=".png"):
    file_list = os.listdir(source_folder+'\\norm_check')
    print(file_list)
    #specific_files = [os.path.join(source_folder, file) for file in file_list if file.lower().endswith(extension.lower())]
    return file_list


source_folder = os.getcwd()
print(source_folder)
 # Paths to the images to combine
mkdir(source_folder+'\\norm_compare')
for i in scan_picture(source_folder, extension=".jpg"):
    print(extract_name(i).replace('wfun_','')+':')
    image_paths=[os.path.join(source_folder+'\\norm_check', i)]+[os.path.join(source_folder+'\\norm_check_M', i)]
    #print( image_paths)

    # Path to the output image
    output_path = source_folder+'\\norm_compare\\'+i


    # Combine images in a designed grid
    combine_images_grid(image_paths, output_path, [2,1])

# Define the directory path to be deleted
dir_path = 'path/to/directory'

# Ensure the directory exists before attempting to delete
if os.path.exists(source_folder+'\\norm_check_M'):
    shutil.rmtree(source_folder+'\\norm_check_M')
if os.path.exists(source_folder+'\\norm_check'):
    shutil.rmtree(source_folder+'\\norm_check')
