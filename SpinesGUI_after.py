import pickle
import os
import organise_paths
import glob
import shutil

userID = 'rubencorreia'
expID = '2025-04-03_05_ESRC004' #in case it's a combined experiment ID, put expID which contains the combined data

animalID, remote_repository_root, processed_root, exp_dir_processed, exp_dir_raw = organise_paths.find_paths(userID, expID)
suite2p_folder = os.path.join(exp_dir_processed,'suite2p')
new_folder =  os.path.join(suite2p_folder,"suite2p_original_files") 
print("suite2p_original_files folder has been created") #create folder for which step1 preprocessing files will be moved, with exception of the data.bin
os.makedirs(new_folder,exist_ok = True) 

print(f"Moving folders to {new_folder}")
items_in_root = os.listdir(suite2p_folder) #Check items in the root folder
print(items_in_root)
for item in items_in_root:
    if item != 'SpinesGUI' and item != 'suite2p_original_files':
        print(f"Moving {item}...")
        item_path = os.path.join(suite2p_folder,item)
        shutil.move(item_path, new_folder)
        print(f"{item} folder has been moved from {suite2p_folder} to {new_folder}")
print(f"All the plane folders have been moved to the {new_folder}")
spines_gui_subfolder = os.path.join(suite2p_folder,"SpinesGUI")
items_in_spines_gui = os.listdir(spines_gui_subfolder)
print(items_in_spines_gui)
for item in items_in_spines_gui:
    item_path = os.path.join(spines_gui_subfolder,item)
    if os.path.isdir(item_path):
        destination_folder = os.path.join(suite2p_folder,item)
        os.makedirs(destination_folder,exist_ok = True)
        print(f"Creating {item_path} in the suite2p folder")
        files_in_item = os.listdir(item_path)
        print(files_in_item)
        for file in files_in_item:
            print(f"Copying {file}...")
            file_path = os.path.join(item_path,file)
            shutil.copy(file_path,destination_folder)
            print(f"{file} has been copied")
        print(f"{item} folder has been copied from {spines_gui_subfolder} to {suite2p_folder}")
print("All files have been copied or moved to their correct directories")
print("You can continue with the pre-processing...")




