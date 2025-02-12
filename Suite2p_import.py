import pickle
import os
import organise_paths
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import pyplot as plt


userID = 'rubencorreia'
expID = '2025-01-31_07_ESRC004' 
plane = "plane0"
# the organise_paths.find_paths(userID, expID) gives you various useful
# paths based on an experiment ID
animalID, remote_repository_root, processed_root, exp_dir_processed, exp_dir_raw = organise_paths.find_paths(userID, expID)
# os.path.join combined strings to make a path that will work on whatever 
# operating system the function is run on
suite2p_folder = os.path.join(exp_dir_processed,'suite2p')
print(os.listdir(suite2p_folder))
exp_plane = os.path.join(suite2p_folder, plane)

activity_file = os.path.join(exp_plane,('F.npy'))

activity = np.load(activity_file)
print(activity)

stat_file = os.path.join(exp_plane,('stat.npy'))
stat = np.load(stat_file, allow_pickle=True)


cell_file = os.path.join(exp_plane,('iscell.npy'))
cell = np.load(cell_file, allow_pickle=True)
print(cell)

ops_file = os.path.join(exp_plane,('ops.npy'))
ops = np.load(ops_file, allow_pickle=True).item()
meanImg = ops.get("meanImg", None)
batch_size = ops.get("batch_size", None)
print(batch_size)
print(ops.get("diameter"))
plt.imshow(meanImg)
plt.show()
