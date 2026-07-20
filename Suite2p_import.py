import pickle
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import pyplot as plt

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from suite2p_numpy_compat import load_suite2p_dict, load_suite2p_npy

_LAB_PIPELINE_SRC = Path(__file__).resolve().parent.parent / "lab_pipeline" / "src"
if str(_LAB_PIPELINE_SRC) not in sys.path:
    sys.path.insert(0, str(_LAB_PIPELINE_SRC))

from preprocess_pipeline.shared import paths

userID = 'rubencorreia'
expID = '2025-12-11_07_ESRC023' 
plane = "plane0"
# the preprocess_pipeline.shared.paths.find_paths(userID, expID) helper gives you various useful
# paths based on an experiment ID
animalID, remote_repository_root, processed_root, exp_dir_processed, exp_dir_raw = paths.find_paths(
    userID,
    expID,
)
# os.path.join combined strings to make a path that will work on whatever 
# operating system the function is run on
suite2p_candidates = [
    os.path.join(exp_dir_processed, "suite2p_combined"),
    os.path.join(exp_dir_processed, "suite2p"),
]
suite2p_folder = next((candidate for candidate in suite2p_candidates if os.path.isdir(candidate)), None)
if suite2p_folder is None:
    raise FileNotFoundError(
        f"No Suite2p output folder found under {exp_dir_processed!r}. "
        "Expected suite2p_combined/ or suite2p/."
    )

print(os.listdir(suite2p_folder))
exp_plane = os.path.join(suite2p_folder, plane)

activity_file = os.path.join(exp_plane,('F.npy'))

activity = load_suite2p_npy(activity_file)
print(activity)

stat_file = os.path.join(exp_plane,('stat.npy'))
stat = load_suite2p_npy(stat_file)


cell_file = os.path.join(exp_plane,('iscell.npy'))
cell = load_suite2p_npy(cell_file)
print(cell)

ops_file = os.path.join(exp_plane,('ops.npy'))
ops = load_suite2p_dict(ops_file)
print("Loaded ops file")
print(ops)
print('Finished printing ops file')

meanImg = ops.get("meanImg", None)
batch_size = ops.get("batch_size", None)
print(batch_size)
print(ops.get("diameter"))

print()
print(f'Inserted path {exp_plane}')
print(f"Path for the bin file {ops['reg_file']}")
print(f"Number of frames {ops['nframes']}")

size_bytes = os.path.getsize(ops['reg_file'])
Ly, Lx = ops['Ly'], ops['Lx']
expected_frame_size = Ly * Lx * np.dtype('float32').itemsize
expected_frames = size_bytes // expected_frame_size
print(f"File has {expected_frames} frames")

plt.imshow(meanImg)
plt.show()
