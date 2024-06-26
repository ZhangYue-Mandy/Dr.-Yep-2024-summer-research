import os
def scan_file(source_folder, extension=".fits"):
    file_list = os.listdir(source_folder)
    specific_files = [os.path.join(source_folder, file) for file in file_list if file.lower().endswith(extension.lower()) and 'target' in file ]
    return specific_files

def write_io(info_file,info): #Function to write picture to a file
    #     # Write in text mode
    with open(info_file+'\\target_information.txt', 'a') as f:
        f.write(info)

#get our currently working folder
source_folder = os.getcwd()
print(scan_file(source_folder))
for i in scan_file(source_folder):
   write_io(source_folder,i+'\n')