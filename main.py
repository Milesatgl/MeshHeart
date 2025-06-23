import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import os

from data.dataloader import UKbiobankMesh
# import util.mesh as u_mesh
import util.utils as util
import loss.loss as Loss
# import util.transform as transform

# import logging
from config import load_config

import timeit


from torch.utils.tensorboard import SummaryWriter
# import model.lstm_vae_batch as lstm
# import model.coma_transformer_vae as gcn
import model.transformer_vae_batch as Trans_vae
import time
import pyvista as pv
import h5py


def save_model(mesh_vae, optimizer, epoch, train_loss, val_loss, checkpoint_name):
    checkpoint = {}
    checkpoint['state_dict'] = mesh_vae.state_dict()
    checkpoint['optimizer'] = optimizer.state_dict()
    checkpoint['epoch_num'] = epoch
    checkpoint['train_loss'] = train_loss
    checkpoint['val_loss'] = val_loss
    torch.save(checkpoint, checkpoint_name)


def train(mesheart, trainloader, optimizer, device, config, writer, epoch):
    mesheart.train()
    avg_loss = []

    for idx, data in enumerate(trainloader):

        heart_v, heart_f, heart_e, con, subid = data

        optimizer.zero_grad()
        heart_v, heart_f, heart_e, con = heart_v.to(device), heart_f.to(device), heart_e.to(device), con.to(device)

        # logvar and mu is from encoder block
        # v_out is the reconstructed mesh, from decoder block
        v_out, logvar, mu = mesheart(heart_v, heart_f, heart_e, con)

        loss, loss_recon = Loss.VAECELoss(v_out, heart_v, heart_f, logvar,
                                              mu, beta=config.beta, lambd=config.lambd, lambd_s=config.lambd_s, loss=config.loss)
        avg_loss.append(loss.item())
        loss.backward()
        optimizer.step()

    # logging.info('epoch:{}, loss:{}'.format(epoch, np.mean(avg_loss)))
    print('epoch:{}, loss:{}'.format(epoch, np.mean(avg_loss)))


    return np.mean(avg_loss)

def val(mesheart, validloader, optimizer, device, config, writer, epoch):
    # logging.info('-------------validation--------------')
    print('-------------validation--------------')
    mesheart.eval()
    with torch.no_grad():
        valid_error = []
        for idx, data in enumerate(validloader):
            myo_v, myo_f, myo_e, con, subid = data

            # debug
            # if subid[0]!='3703255',1890962:
            #     continue
            optimizer.zero_grad()
            myo_v, myo_f, myo_e, con = myo_v.to(device), myo_f.to(device), myo_e.to(device), con.to(device)

            v_out, logvar, mu = mesheart(myo_v, myo_f, myo_e, con)


            loss, loss_recon = Loss.VAECELoss(v_out, myo_v, myo_f, logvar,
                                              mu, beta=config.beta, lambd=config.lambd, lambd_s=config.lambd_s, loss=config.loss)
            valid_error.append(loss_recon)
        this_val_error = torch.mean(torch.stack(valid_error))

        print('epoch:{}, validation error:{}'.format(epoch, this_val_error))
        print('-------------------------------------')
        return this_val_error


def main(config):
    # --------------------------
    # load configuration
    # --------------------------
    os.environ['CUDA_VISIBLE_DEVICES'] = str(config.gpu)
    age_group = config.age_group
    print('age_group:', age_group)
    model_dir = config.model_dir
    device = config.device
    train_type = config.train_type
    tag = config.tag
    surf_type = config.surf_type

    n_epochs = config.n_epochs
    n_samples = config.n_samples
    lr = config.lr
    lr_decay = 0.99

    z_dim = config.z_dim
    channal = 3
    con_num_in = 10
    con_num_out = 32
    C = config.dim_h
    print('C:', con_num_in)

    model_name = f"{train_type}_z_dim{config.z_dim}_loss_{config.loss}_beta{config.beta}_" \
                 f"lambd{config.lambd}_lambds{config.lambd_s}_lr{config.lr}_wd{config.wd}_batch{config.batch}"


    start = timeit.default_timer()

    trainset = UKbiobankMesh(config, 'train')
    validset = UKbiobankMesh(config, 'val')


    trainloader = DataLoader(trainset, batch_size=config.batch, shuffle=True)
    validloader = DataLoader(validset, batch_size=config.batch, shuffle=False)

    # --------------------------
    # initialize models
    # --------------------------

    mesheart = Trans_vae.CAE(dim_in=channal,
                             dim_h=C, z_dim=z_dim,
                             c_dim_in=con_num_in, c_dim=con_num_out,
                             points=n_samples, seq_len=config.seq_len,
                             ff_size=config.ff_size, num_heads=config.num_heads,
                             activation=config.activation, num_layers=config.num_layers,
                             train_type=train_type).to(device)

    if config.wd:
        optimizer = optim.Adam(mesheart.parameters(), lr=lr, weight_decay=config.wd)
    else:
        optimizer = optim.Adam(mesheart.parameters(), lr=lr)

    # --------------------------
    # training
    # --------------------------

    logdir = f"{model_dir}/tb/{model_name}"
    util.setup_dir(logdir)
    writer = SummaryWriter(logdir)
    writer.add_hparams({'type': train_type, 'tag': tag}, {})
    cp_path = f"{model_dir}/model"
    util.setup_dir(cp_path)

    # load checkpoint_file
    checkpoint_file = config.checkpoint_file
    if checkpoint_file:
        ckptname = f'{cp_path}/{model_name}/epoch{checkpoint_file}.pt'
        checkpoint = torch.load(ckptname)
        start_epoch = checkpoint['epoch_num']
        mesheart.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
    mesheart.to(device)

    # logging.info("start training ...")
    print("start training ...")

    for epoch in tqdm(range(n_epochs + 1)):
        train_loss = train(mesheart, trainloader, optimizer, device, config, writer, epoch)
        if epoch % 20 == 0:
            val_loss = val(mesheart, validloader, optimizer, device, config, writer, epoch)
            # save model checkpoints
            if checkpoint_file:
                ckptname = f'{cp_path}/{model_name}/epoch{epoch+1+checkpoint_file}.pt'
            else:
                ckptname = f'{cp_path}/{model_name}/epoch{epoch}.pt'
            util.setup_dir(f'{cp_path}/{model_name}/')
            if epoch % 20 == 0:
                save_model(mesheart, optimizer, epoch, train_loss, val_loss, ckptname)
                # torch.save(mesheart.state_dict(), ckptname)

        writer.add_scalars('Train_Loss', {'train': train_loss}, epoch)
        writer.add_scalars('Val_Loss', {'val': val_loss}, epoch)

import warnings
if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    config = load_config()
    main(config)
