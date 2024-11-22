import torch
import numpy as np
import os

FOLDER_PATH = '/root/diffusion_mujoco_panda/inference_SG_MG' 

# npy load
npy_path = '/root/diffusion_mujoco_panda/inference_SG_MG/SG_time_cat.npy'
npy_data = np.load(npy_path)


SG_j_list = []
SG_time_list = []
MG_j_list = []
MG_time_list = []

# SG j & time
for idx in range(0,20):
    file_name = 'j_SG_' + 'idx-' + str(idx) + '_test.npy'
    file_path = os.path.join(FOLDER_PATH , file_name)
    j_SG = np.load(file_path)
    SG_j_list.append(j_SG)
SG_j = np.array(SG_j_list).reshape(20, 200)
np.save(os.path.join(FOLDER_PATH , f'SG_j_' + 'cat.npy'), SG_j)

for idx in range(0,20):
    file_name = 'time_SG_' + 'idx-' + str(idx) + '_test.npy'
    file_path = os.path.join(FOLDER_PATH , file_name)
    time_SG = np.load(file_path)
    SG_time_list.append(time_SG)
SG_time = np.array(SG_time_list).reshape(20, 200)
np.save(os.path.join(FOLDER_PATH , f'SG_time_' + 'cat.npy'), SG_time)

# MG j & time
for idx in range(20,40):
    file_name = 'j_MG_' + 'idx-' + str(idx) + '_test.npy'
    file_path = os.path.join(FOLDER_PATH , file_name)
    j_MG = np.load(file_path)
    MG_j_list.append(j_MG)
MG_j = np.array(MG_j_list).reshape(20, 200)
np.save(os.path.join(FOLDER_PATH , f'MG_j_' + 'cat.npy'), MG_j)

for idx in range(20,40):
    file_name = 'time_MG_' + 'idx-' + str(idx) + '_test.npy'
    file_path = os.path.join(FOLDER_PATH , file_name)
    time_MG = np.load(file_path)
    MG_time_list.append(time_MG)
MG_time = np.array(MG_time_list).reshape(20, 200)
np.save(os.path.join(FOLDER_PATH , f'MG_time_' + 'cat.npy'), MG_time)