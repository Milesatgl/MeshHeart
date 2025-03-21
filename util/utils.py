import os
import torch
import math
import pickle

def setup_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    return dir_path

def age_transform(age):
    ## divide the age into 6 groups
    # <=50
    # years in[50,75], 5 years intervals
    # >=75
    if age < 55:
        return 0
    else:
        return min(math.floor((age - 50) // 5), 6 - 1)

def condition_normal_batch(age, gender,
                           normaltype=0, samesize=True,
                           NUM_AGES=6, NUM_GENDERS=2):
    ##normalization 0 is one-hot
    ##normalization 1 is as AgeHeart modified from str_to_tensor
    ##normal means the normalization is True

    batchsize= 1
    # initialize
    if normaltype==0:
        age_tensor = torch.zeros([batchsize, NUM_AGES])
        gender_tensor = torch.zeros([batchsize, NUM_GENDERS])
    else:
        age_tensor = -torch.ones([batchsize, NUM_AGES])
        gender_tensor = -torch.ones([batchsize, NUM_GENDERS])
    if samesize:
        gender_tensor = age_tensor


    for i in range(batchsize):
        if normaltype == 0:
            age_temp = torch.zeros(NUM_AGES)
            age_temp[int(age[i])] += 1
            gender_temp = torch.zeros(NUM_GENDERS)
            gender_temp[int(gender[i])] += 1
        else:
            age_temp = -torch.ones(NUM_AGES)
            age_temp[int(age[i])] *= -1
            gender_temp = -torch.ones(NUM_GENDERS)
            gender_temp[int(gender[i])] *= -1
        if samesize:#gender have the same weight as age
            gender_temp = gender_temp.repeat(NUM_AGES // NUM_GENDERS)#TODO: modify it, only be int,(6) at non-singleton dimension 0.  Target sizes: [7].  Tensor sizes: [6]

        age_tensor[i] = age_temp
        gender_tensor[i] = gender_temp
    return age_tensor, gender_tensor

def condition_normal(age, gender, dise, hypert,
                     normaltype=0, samesize=False,
                     NUM_AGES=6, NUM_GENDERS=2):
    ##normalization 0 is one-hot
    ##normalization 1 is as AgeHeart modified from str_to_tensor
    ##normal means the normalization is True

    if normaltype == 0:
        age_temp = torch.zeros(NUM_AGES)
        age_temp[int(age)] += 1
        gender_temp = torch.zeros(NUM_GENDERS)
        gender_temp[int(gender)] += 1
        # dise_temp = torch.zeros(NUM_GENDERS)
        # dise_temp[int(dise)] += 1
        # hypert_temp = torch.zeros(NUM_GENDERS)
        # hypert_temp[int(hypert)] += 1
    else:
        age_temp = -torch.ones(NUM_AGES)
        age_temp[int(age)] *= -1
        gender_temp = -torch.ones(NUM_GENDERS)
        gender_temp[int(gender)] *= -1
        # dise_temp = torch.ones(NUM_GENDERS)
        # dise_temp[int(dise)] *= -1
        # hypert_temp = torch.ones(NUM_GENDERS)
        # hypert_temp[int(hypert)] *= -1
    if samesize:
        ##gender have the same weight as age
        gender_temp = gender_temp.repeat(NUM_AGES // NUM_GENDERS)#TODO: modify it, only be int,(6) at non-singleton dimension 0.  Target sizes: [7].  Tensor sizes: [6]
        # dise_temp = dise_temp.repeat(NUM_AGES // NUM_GENDERS)
        # hypert_temp = hypert_temp.repeat(NUM_AGES // NUM_GENDERS)

    return age_temp,\
           gender_temp,\
           dise, \
           hypert

def pet_save(pet, filename):
    with open(filename, 'wb') as f:
        pickle.dump(pet, f, pickle.HIGHEST_PROTOCOL)

def pet_load(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

import numpy as np
def scipy_to_torch_sparse(scp_matrix):
    values = scp_matrix.data
    indices = np.vstack((scp_matrix.row, scp_matrix.col))
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = scp_matrix.shape

    sparse_tensor = torch.sparse.FloatTensor(i, v, torch.Size(shape))
    return sparse_tensor