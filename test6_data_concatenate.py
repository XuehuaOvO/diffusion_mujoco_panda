import torch
import numpy as np
import os

FOLDER_PATH = '/root/diffusion_mujoco_panda/collecting_test/collecting_6' 

# npy load
npy_path = '/root/diffusion_mujoco_panda/collecting_test/collecting_6/new_cost_idx-103.npy'
npy_data = np.load(npy_path)
print(f'cost -- {npy_data}')

npy_path = '/root/diffusion_mujoco_panda/collecting_test/collecting_6/time_mpc_idx-102.npy'
data = np.load(npy_path)
print(f'delta time -- {data}')

# tensor load
# file_path = "/root/diffusion_mujoco_panda/collecting_test/collecting_6/j_data_idx-101_test6.pt"
# data = torch.load(file_path)
# print(f'delta time -- {data}')

# # tensor size
# print(f'data_size -- {data.size()}')
# print(data[4200,:])

data_idxs = [0, 4, 8, 9, 13, 14, 17, 21, 23, 24, 27, 28, 29, 30, 31, 32, 39, 40, 41, 42, 43, 44, 45, 46, 47, 50, 51, 52]

for idx in data_idxs:
    print(f'idx -- {idx}')

tensor_u_list = []
tensor_x_list = []
tensor_j_list = []

# u data cat
for idx in data_idxs:
    file_name = 'u_data_' + 'idx-' + str(idx) + '_test6.pt'
    file_path = os.path.join(FOLDER_PATH , file_name)
    tensor = torch.load(file_path)
    tensor_u_list.append(tensor)

# for idx in range(5,28):
#     file_name = 'u_data_' + 'idx-' + str(idx) + '_test5.pt'
#     file_path = os.path.join(FOLDER_PATH , file_name)
#     tensor = torch.load(file_path)
#     tensor_u_list.append(tensor)

concatenated_u_tensor = torch.cat(tensor_u_list, dim=0)
torch.save(concatenated_u_tensor, os.path.join(FOLDER_PATH , f'u_data_' + 'cat' + '_test6.pt'))

# x data cat
for idx in data_idxs:
    file_name = 'x_data_' + 'idx-' + str(idx) + '_test6.pt'
    file_path = os.path.join(FOLDER_PATH , file_name)
    tensor = torch.load(file_path)
    tensor_x_list.append(tensor)

# for idx in range(5,28):
#     file_name = 'x_data_' + 'idx-' + str(idx) + '_test5.pt'
#     file_path = os.path.join(FOLDER_PATH , file_name)
#     tensor = torch.load(file_path)
#     tensor_x_list.append(tensor)

concatenated_x_tensor = torch.cat(tensor_x_list, dim=0)
torch.save(concatenated_x_tensor, os.path.join(FOLDER_PATH , f'x_data_' + 'cat' + '_test6.pt'))

# j data cat
for idx in data_idxs:
    file_name = 'j_data_' + 'idx-' + str(idx) + '_test6.pt'
    file_path = os.path.join(FOLDER_PATH , file_name)
    tensor = torch.load(file_path)
    tensor_j_list.append(tensor)

# for idx in range(5,28):
#     file_name = 'j_data_' + 'idx-' + str(idx) + '_test5.pt'
#     file_path = os.path.join(FOLDER_PATH , file_name)
#     tensor = torch.load(file_path)
#     tensor_j_list.append(tensor)

concatenated_j_tensor = torch.cat(tensor_j_list, dim=0)
torch.save(concatenated_j_tensor, os.path.join(FOLDER_PATH , f'j_data_' + 'cat' + '_test6.pt'))