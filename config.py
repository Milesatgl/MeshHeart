import argparse
import torch


def load_config():
    # args
    parser = argparse.ArgumentParser(description="MeshHeart")

    # for training
    parser.add_argument('--data_dir', type=str,
                        help="directory of the dataset")
    parser.add_argument('--target_seg_dir', type=str,
                        help="directory of the dataset")
    parser.add_argument('--label_dir', type=str,
                        help="directory of the label file/training dataset file")
    parser.add_argument('--batch', default=1, type=int, help="batchsize")
    parser.add_argument('--model_dir', type=str, help="directory for saving the models")
    parser.add_argument('--data_name', default="ukbiobank", type=str, help="name of the dataset")
    parser.add_argument('--train_type', default="VAE_GCN", type=str, help="type of training: [AE-GCN,AE-Linear]")
    parser.add_argument('--surf_type', type=str, help="type of the surface: [sample,all]")
    parser.add_argument('--loss', type=str, help="loss_type")
    parser.add_argument('--device', default="gpu", type=str, help="gpu or cpu")
    parser.add_argument('--gpu', default=0, type=int, help="the gpu device index")
    parser.add_argument('--seq_len', default=50, type=int, help="length of the mesh sequence")
    parser.add_argument('--solver', default='euler', type=str, help="ODE solver: [euler, midpoint, rk4]")
    parser.add_argument('--step_size', default=0.2, type=float, help="step size of the ODE solver")
    parser.add_argument('--lambd', default=1.0, type=float, help="reconstruction weight")
    parser.add_argument('--beta', default=1e-2, type=float, help="beta-vae")
    parser.add_argument('--lambd_s', default=1.0, type=float, help="Laplacian smooth weight")
    parser.add_argument('--ff_size', default=1024, type=int, help="ff_size of transformer")
    parser.add_argument('--num_heads', default=4, type=int, help="number of heads")
    parser.add_argument('--activation', default='gelu', type=str, help="activation type")
    parser.add_argument('--num_layers', default=2, type=int, help="number of layers in transformer")
    parser.add_argument('--norm_first', default=False, type=bool, help="if True, encoder and decoder "
                                                                       "layers will perform LayerNorms "
                                                                       "before other attention and "
                                                                       "feedforward operations, otherwise after")

    parser.add_argument('--normalize', default=True, type=bool, help="normalize the input points")
    parser.add_argument('--n_epochs', type=int, help="num of training epochs")
    parser.add_argument('--n_samples', type=int, help="num of sampled points for training")
    parser.add_argument('--lr', default=0.0001, type=float, help="learning rate")
    parser.add_argument('--wd', default=None, type=float, help="learning rate")
    parser.add_argument('--z_dim', default=64, type=int, help="dimension of laten space")
    parser.add_argument('--kernel_size', default=5, type=int, help="kernel size of conv layers")
    parser.add_argument('--dim_h', default=128, type=int, help="dimension of hidden layers")
    parser.add_argument('--n_scale', default=3, type=int, help="num of scales for multi-scale inputs")

    parser.add_argument('--visualize', default=True)
    # # for testing
    parser.add_argument('--checkpoint_file', default=None, type=int, help="checkpoint_file(epoch number)")

    config = parser.parse_args()

    if config.device == "gpu":
        config.device = torch.device("cuda")
    elif config.device == "cpu":
        config.device = torch.device("cpu")
    else:
        config.device = torch.device(config.device)

    config.device = torch.device(config.device)
    return config