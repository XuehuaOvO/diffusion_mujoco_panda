import torch
import numpy as np
import os

FOLDER_PATH = '/root/diffusion_mujoco_panda/selected_data' 

# tensor load
file_path = "/root/diffusion_mujoco_panda/selected_data/j_data_idx-0_selected_test.pt"
data = torch.load(file_path)


tensor_u_list = []
tensor_x_list = []
tensor_j_list = []

# u data cat
for idx in range(0,5):
    file_name = 'u_data_' + 'idx-' + str(idx) + '_selected_test.pt'
    file_path = os.path.join(FOLDER_PATH , file_name)
    tensor = torch.load(file_path)
    tensor_u_list.append(tensor)

concatenated_u_tensor = torch.cat(tensor_u_list, dim=0)
torch.save(concatenated_u_tensor, os.path.join(FOLDER_PATH , f'u_selected_data_' + 'cat.pt'))

# x data cat
for idx in range(0,5):
    file_name = 'x_data_' + 'idx-' + str(idx) + '_selected_test.pt'
    file_path = os.path.join(FOLDER_PATH , file_name)
    tensor = torch.load(file_path)
    tensor_x_list.append(tensor)

concatenated_x_tensor = torch.cat(tensor_x_list, dim=0)
torch.save(concatenated_x_tensor, os.path.join(FOLDER_PATH , f'x_selected_data_' + 'cat.pt'))

# j data cat
for idx in range(0,5):
    file_name = 'j_data_' + 'idx-' + str(idx) + '_selected_test.pt'
    file_path = os.path.join(FOLDER_PATH , file_name)
    tensor = torch.load(file_path)
    tensor_j_list.append(tensor)

concatenated_j_tensor = torch.cat(tensor_j_list, dim=0)
torch.save(concatenated_j_tensor, os.path.join(FOLDER_PATH , f'j_selected_data_' + 'cat.pt'))