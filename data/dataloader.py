from torch.utils.data import Dataset
import torch
import numpy as np
import csv
# import pyvista as pv
import util.utils as util
# import util.mesh as u_mesh
import h5py
from torch_geometric.data import InMemoryDataset, Data
# from tqdm import tqdm

# import util.transform as transform

class UKbiobankMesh(Dataset):

    def __init__(self, config, data_usage='train'):

        data_list = []
        csvpath = f"{config.label_dir}/mesh_{data_usage}.csv"
        with open(csvpath, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                data_list.append((row['Unnamed: 0'],
                                  float(row['Age']),
                                  float(row['Sex']),
                                  float(row['Weight']),
                                  float(row['Height'])))

        self.data_list = data_list
        self.data_dir = config.target_seg_dir
        self.device = config.device
        self.seq_len = config.seq_len
        self.label_dir = config.label_dir
        self.normalize = config.normalize
        self.n_samples = config.n_samples
        self.train_type = config.train_type
        self.surf_type = config.surf_type

    def __getitem__(self, index):
        # subject_name = self.file_list[index]

        subid, age, sex, weight, height = self.data_list[index]
        # print(subid)
        ## transfer age and sex into one-hot
        # age_group = util.age_transform(age)
        # age_group, sex, weight, height = util.condition_normal(np.asarray(age_group), np.asarray(sex), np.asarray(weight), np.asarray(height))
        # # torch.Tensor(np.asarray((age_group, sex)))
        # gender_temp = torch.zeros(2)
        # gender_temp[int(np.asarray(sex))] += 1
        # sex = gender_temp
        age, weight, height = torch.Tensor(np.asarray((age, weight, height)))
        age_group = util.age_transform(age)
        age_group, sex, weight, height = util.condition_normal(np.asarray(age_group), np.asarray(sex),
                                                               weight, height)
        weight, height = torch.round(weight), torch.round(height)

        data_dir = self.data_dir
        mesh_path = f"{data_dir}/{subid}/image_space_pipemesh"
        if self.surf_type == 'all':
            h5filepath = f'{mesh_path}/preprossed_vtk.hdf5'
        elif self.surf_type == 'sample':
            h5filepath = f'{mesh_path}/preprossed_decimate.hdf5'
        f = h5py.File(h5filepath, "r")
        mesh_verts = torch.Tensor(np.array(f['heart_v']))
        return mesh_verts, torch.LongTensor(np.array(f['heart_f'])), torch.LongTensor(np.array(f['heart_e'])),\
                   torch.concat((age_group, sex, weight.unsqueeze(0), height.unsqueeze(0)), dim=0), \
                   subid



    def __len__(self):
        return len(self.data_list)
