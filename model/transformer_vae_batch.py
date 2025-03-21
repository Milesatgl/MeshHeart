import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GraphConv, GCNConv
from torch_geometric.data import Batch
from pytorch3d.structures import Meshes
from util.mesh import transfer_face_to_edge
import trimesh
import pyvista as pv
from torch_geometric.data import Data, Batch

class MLP(nn.Module):
    def __init__(self, input_dim, out_dim, fc_dim, n_fc,
                 weight_norm=False, activation='relu', normalize_mlp=True):#, pixel_norm=False):
        super(MLP, self).__init__()
        # if weight_norm:
        #     linear = EqualLinear
        # else:
        #     linear = nn.Linear
        linear = nn.Linear
        if activation == 'lrelu':
            actvn = nn.LeakyReLU(0.2,True)
        # elif activation == 'blrelu':
        #     actvn = BidirectionalLeakyReLU()
        else:
            actvn = nn.ReLU(True)

        self.input_dim = input_dim
        self.model = []

        # normalize input
        if normalize_mlp:
            self.model += [PixelNorm()]

         # set the first layer
        self.model += [linear(input_dim, fc_dim),
                       actvn]
        if normalize_mlp:
            self.model += [PixelNorm()]

        # set the inner layers
        for i in range(n_fc - 2):
            self.model += [linear(fc_dim, fc_dim),
                           actvn]
            if normalize_mlp:
                self.model += [PixelNorm()]

        # set the last layer
        self.model += [linear(fc_dim, out_dim)] # no output activations

        # normalize output
        if normalize_mlp:
            self.model += [PixelNorm()]

        self.model = nn.Sequential(*self.model)

    def forward(self, input):
        out = self.model(input)
        return out

class PixelNorm(nn.Module):
    def __init__(self, num_channels=None):
        super().__init__()
        # num_channels is only used to match function signature with other normalization layers
        # it has no actual use

    def forward(self, input):
        return input / torch.sqrt(torch.mean(input ** 2, dim=1, keepdim=True) + 1e-5)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=60):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        # not used in the final model
        x = x + self.pe[:x.shape[0], :]
        return self.dropout(x)


# only for ablation / not used in the final model
class TimeEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(TimeEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, mask, lengths):
        time = mask * 1 / (lengths[..., None] - 1)
        time = time[:, None] * torch.arange(time.shape[1], device=x.device)[None, :]
        time = time[:, 0].T
        # add the time encoding
        x = x + time[..., None]
        return self.dropout(x)


class Encoder_TRANSFORMER(nn.Module):
    def __init__(self, dim_in=3, points=10000, seq_len=50, c_dim=12, z_dim=32, ff_size=1024, num_layers=4, num_heads=4, dropout=0.1,
                 ablation=None, activation="gelu"):
        super().__init__()


        self.points = points
        self.dim_in = dim_in
        self.num_frames = seq_len
        self.num_classes = c_dim


        self.latent_dim = z_dim

        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout

        self.ablation = ablation
        self.activation = activation
        self.use_bias = True

        self.input_feats = self.points * self.dim_in


        #### GCN

        self.skelEmbedding = nn.Sequential(
            nn.Conv1d(in_channels=dim_in, out_channels=64, kernel_size=1,
                      bias=self.use_bias),
            nn.ReLU(inplace=True),
        )
        self.gcn1 = GCNConv(64, 128)
        self.gcn2 = GCNConv(128, 256)

        self.fc = nn.Sequential(
            nn.Linear(256+c_dim, self.latent_dim, bias=True),
            nn.ReLU(inplace=True)
        )


        self.muQuery = nn.Parameter(torch.randn(self.latent_dim))
        self.sigmaQuery = nn.Parameter(torch.randn(self.latent_dim))


        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)

        # self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))

        seqTransEncoderLayer = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                          nhead=self.num_heads,
                                                          dim_feedforward=self.ff_size,
                                                          dropout=self.dropout,
                                                          activation=self.activation)
        self.seqTransEncoder = nn.TransformerEncoder(seqTransEncoderLayer,
                                                     num_layers=self.num_layers)

    def forward(self, v, f, edge_list, con):
        # x = v.permute((3, 0, 1, 2)).reshape(nframes, bs, njoints * nfeats)
        # gcn: input: node features (vertices, features)
        #   edge indices (2,edges)
        # embedding of the skeleton
        ### for gcn:
        batch_size = v.shape[0]
        se_length = v.shape[1]
        nodes = v.shape[2]

        v = v.reshape(batch_size * se_length, nodes, -1).permute((0, 2, 1))
        v = self.skelEmbedding(v)
        v = v.permute((0, 2, 1)).reshape((batch_size, se_length, nodes, -1))
        edge_list = edge_list.permute((0, 1, 3, 2))

        v_all = []
        for seq_len in range(se_length):
            data_list = []
            for b in range(batch_size):
                data = Data(x=v[b, seq_len], edge_index=edge_list[b, seq_len])
                data_list.append(data)
            batch = Batch.from_data_list(data_list)
            temp = F.leaky_relu(self.gcn1(batch.x, batch.edge_index), 0.15)
            temp = F.leaky_relu(self.gcn2(temp, batch.edge_index), 0.15).reshape(batch_size, 1, nodes, -1)
            v_all.append(temp)
        v = torch.cat(v_all, dim=1).permute((0, 1, 3, 2))

        v = v.max(dim=3)[0]
        ## concat with conditions
        con = con.unsqueeze(1).repeat(1, v.shape[1], 1)
        x = self.fc(torch.cat((v, con), dim=2))

        xseq = torch.cat((self.muQuery.unsqueeze(0).unsqueeze(0).repeat(batch_size, 1, 1),
                          self.sigmaQuery.unsqueeze(0).unsqueeze(0).repeat(batch_size, 1, 1), x), axis=1).permute(1, 0, 2)
        xseq = self.sequence_pos_encoder(xseq)

        # transformer layers
        # TODO: if not using MLP, try add random into mu and logvar
        xseq = self.seqTransEncoder(xseq)
        mu = xseq[0]
        logvar = xseq[1]
        return mu, logvar, xseq



class Decoder_TRANSFORMER(nn.Module):
    def __init__(self,  dim_in=3, points=10000, seq_len=50,
                 c_dim=16, z_dim=32, ff_size=1024, num_layers=4, num_heads=4, dropout=0.1,
                 ablation=None, activation="gelu"):
        super().__init__()


        self.njoints = points
        self.nfeats = dim_in
        self.num_frames = seq_len
        self.num_classes = c_dim

        self.latent_dim = z_dim

        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout

        self.ablation = ablation

        self.activation = activation

        self.input_feats = self.njoints * self.nfeats
        self.use_bias = True


        # if self.ablation == "zandtime":
        self.ztimelinear = nn.Linear(self.latent_dim + self.num_classes, self.latent_dim)
        # else:
        #     self.actionBiases = nn.Parameter(torch.randn(self.num_classes, self.latent_dim))
        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)



        seqTransDecoderLayer = nn.TransformerDecoderLayer(d_model=self.latent_dim,
                                                          nhead=self.num_heads,
                                                          dim_feedforward=self.ff_size,
                                                          dropout=self.dropout,
                                                          activation=activation)
        self.seqTransDecoder = nn.TransformerDecoder(seqTransDecoderLayer,
                                                     num_layers=self.num_layers)

        self.finallayer = nn.Sequential(
            nn.Linear(in_features=self.latent_dim, out_features=64, bias=self.use_bias),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=64, out_features=128, bias=self.use_bias),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=128, out_features=512, bias=self.use_bias),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=512, out_features=1024, bias=self.use_bias),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=1024, out_features=points * 3, bias=self.use_bias),
        )
    def forward(self, z, con):
        latent_dim = z.shape[1]
        bs = con.shape[0]
        nframes = self.num_frames
        njoints, nfeats = self.njoints, self.nfeats


        z = torch.cat((z, con), axis=1)
        z = self.ztimelinear(z)
        z = z[None]  # sequence of size 1

        timequeries = torch.zeros(nframes, bs, latent_dim, device=z.device)
        timequeries = self.sequence_pos_encoder(timequeries)

        output_seq = self.seqTransDecoder(tgt=timequeries, memory=z)

        output = self.finallayer(torch.squeeze(output_seq, 1)).reshape(nframes, bs, njoints, nfeats)
        return output.permute(1, 0, 2, 3), output_seq


class CAE(nn.Module):
    ## from ACTOR
    def __init__(self, dim_in=3,
                 dim_h=128, z_dim = 32,
                 c_dim_in=12, c_dim=16,
                 points=15000, seq_len=50,
                 ff_size=1024, num_heads=4,
                 activation="gelu", num_layers=4, train_type=None):
        super().__init__()
        self.latent_dim = z_dim
        self.train_type = train_type
        self.mapping = MLP(c_dim_in, c_dim, 64, 2, weight_norm=True)
        self.encoder = Encoder_TRANSFORMER(dim_in=dim_in, points=points,
                                           seq_len=seq_len, c_dim=c_dim,
                                           z_dim=z_dim, ff_size=ff_size,
                                           num_layers=4, num_heads=num_heads, dropout=0.1,
                                           ablation=train_type, activation="gelu")
        self.decoder = Decoder_TRANSFORMER(dim_in=dim_in, points=points,
                                           seq_len=seq_len, c_dim=c_dim,
                                           z_dim=z_dim, ff_size=ff_size,
                                           num_layers=4, num_heads=num_heads, dropout=0.1,
                                           ablation=train_type, activation="gelu")


    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)

        return eps.mul(std).add_(mu)

    def forward(self, v, f, edge_list, c):
        # v: batch, time_frames, points, channel
        con = self.mapping(c)
        # con = torch.argmax(c, axis=1)
        mu, logvar, xseq = self.encoder(v, f, edge_list, con)
        z = self.reparameterize(mu, logvar)

        # decode
        v_all, v_all_latent = self.decoder(z, con)# torch.Size([1, 50, 22043, 3])
        return v_all, logvar, mu

    def return_latent(self, v, f, edge_list, c):
        con = self.mapping(c)
        # con = torch.argmax(c, axis=1)
        mu, logvar, xseq = self.encoder(v, f, edge_list, con)
        z = self.reparameterize(mu, logvar)
        return z, xseq,con

    def generate_one(self, z, con):#, device, duration=50, fact=1):
        con = self.mapping(con)
        v_all, v_all_latent = self.decoder(z, con)
        return v_all, v_all_latent

    def generate_random_one(self, con):
        template_file_path = './template/heart_ES_imagespace.vtk'
        mesh_temp = trimesh.load(template_file_path)
        z = torch.randn(self.latent_dim, device=con.device)[None]#, device, duration=50, fact=1):
        con = self.mapping(con)
        v_all, _ = self.decoder(z, con)
        return v_all

class UnFlatten(nn.Module):
    def __init__(self, points, C):
        super(UnFlatten, self).__init__()
        self.points = points
        self.C = C

    def forward(self, input):
        return input.view(input.size(0), self.points, self.C * 2)