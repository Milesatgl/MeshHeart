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
    '''
    实现了原始 "Attention Is All You Need" 论文(作者 Vaswani 等人)中的公式。它为序列元素添加了位置信息，因为 Transformer 架构本身没有对序列顺序的内在理解。
    它对每个元素的序列位置(时间位置)进行编码。

    PositionalEncoding 类为序列中的每个位置(从 0 到 max_len-1)创建唯一的嵌入。每个位置都会获得一个长度为 d_model 的向量，其中：

    - 第一个维度(位置 0)在所有 d_model 维度上获得唯一的编码模式
    - 第二个维度(位置 1)获得不同的编码模式
    - 以此类推，直到位置 max_len-1

    关键特性是：
    1. 每个序列位置(0, 1, 2...)都会获得一个长度为 d_model 的唯一向量
    2. 使用不同频率的正弦和余弦函数在 d_model 维度上创建这种唯一模式

    这使得 Transformer 能够理解序列中元素的相对顺序，因为 Transformer 本身不像 RNN 那样具有跟踪位置的内在机制。

    当在 forward 方法中计算 `x + self.pe[:x.shape[0], :]` 时，它将位置信息添加到序列中的每个元素。
    '''
    def __init__(self, d_model, dropout=0.1, max_len=60):
        # d_model is the latent dimension, 32
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1) # Shape: [max_len, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)) # shape is [d_model/2]
        pe[:, 0::2] = torch.sin(position * div_term) # Even dimensions get sine，selects all even-indexed columns (0, 2, 4...) across all positions
        pe[:, 1::2] = torch.cos(position * div_term) # Odd dimensions get cosine，selects all odd-indexed columns (1, 3, 5...) across all positions
        pe = pe.unsqueeze(0).transpose(0, 1) # Shape: [max_len, 1, d_model]

        # position * div_term shape is [max_len, d_model/2]; 60*16
        # pe[:, 0::2] shape is [60, d_model/2]; 60*16

        self.register_buffer('pe', pe)

    def forward(self, x):
        # input shape: (se_length+2, batch_size, latent_dim)，pe shape is (max_len, 1, d_model), so se_len+2 corresponds to max_len
        # not used in the final model
        x = x + self.pe[:x.shape[0], :] # output shape is (se_length+2, batch_size, latent_dim)
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

        # self.input_feats = self.points * self.dim_in


        #### GCN
        # (batch_size * se_length, dim_in, nodes)
        # conv1d input is N,C,L; N is a batch size, C denotes a number of channels, L is length of signal sequence., 卷积在最后一个维度，也就是 nodes 上面, 对象是 C;
        # [C_1, C_2, C_3] @ [K_1, K_2, K_3] = out
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


        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout) # 增加序列位置编码，不改变数据维度

        # self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        # d_model: the number of expected features in the input
        seqTransEncoderLayer = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                          nhead=self.num_heads,
                                                          dim_feedforward=self.ff_size,
                                                          dropout=self.dropout,
                                                          activation=self.activation)
        # 4 transformer encoder layers
        self.seqTransEncoder = nn.TransformerEncoder(seqTransEncoderLayer,
                                                     num_layers=self.num_layers)

    def forward(self, v, f, edge_list, con):
        # v: batch, time_frames, points, channel
        # x = v.permute((3, 0, 1, 2)).reshape(nframes, bs, njoints * nfeats)
        # gcn: input: node features (vertices, features)
        #   edge indices (2,edges)
        # embedding of the skeleton
        ### for gcn:
        batch_size = v.shape[0]
        se_length = v.shape[1]
        nodes = v.shape[2]
        # v: batch, time_frames, num_points, channel; channel is x, y, z coords
        v = v.reshape(batch_size * se_length, nodes, -1).permute((0, 2, 1)) # (batch_size * se_length, dim_in, num_nodes), change coords to dim_in
        v = self.skelEmbedding(v) # (batch_size * se_length, 64, num_nodes), 将三维坐标投射到 64 维空间
        v = v.permute((0, 2, 1)).reshape((batch_size, se_length, nodes, -1)) # (batch_size, se_length, num_nodes, 64)，恢复 coords 维度
        edge_list = edge_list.permute((0, 1, 3, 2))

        v_all = []
        for seq_len in range(se_length):
            data_list = []
            for b in range(batch_size):
                #  v[b_0, seq_len_0] shape is (num_nodes, 64)
                data = Data(x=v[b, seq_len], edge_index=edge_list[b, seq_len])
                # so each data object is a graph with num_nodes nodes, each node has 64 features
                data_list.append(data)
            batch = Batch.from_data_list(data_list)
            temp = F.leaky_relu(self.gcn1(batch.x, batch.edge_index), 0.15)
            temp = F.leaky_relu(self.gcn2(temp, batch.edge_index), 0.15).reshape(batch_size, 1, nodes, -1)
            # gcn 邻域聚合特征+投射特征 从64维到256维，temp shape is (batch_size, 1, num_nodes, 256)
            v_all.append(temp)
        # v_all shape is (batch_size, se_length, num_nodes, 256)
        v = torch.cat(v_all, dim=1).permute((0, 1, 3, 2)) # geometry convolution
        # v shape is (batch_size, se_length, 256, num_nodes)

        v = v.max(dim=3)[0] # (batch_size, se_length, 256), max pooling over num_nodes
        ## concat with conditions
        con = con.unsqueeze(1).repeat(1, v.shape[1], 1)
        x = self.fc(torch.cat((v, con), dim=2)) # fc输入是256+c_dim，投射到 latent_dim 32 维空间
        # x shape is (batch_size, se_length, latent_dim)
        xseq = torch.cat((
            self.muQuery.unsqueeze(0).unsqueeze(0).repeat(batch_size, 1, 1),
            self.sigmaQuery.unsqueeze(0).unsqueeze(0).repeat(batch_size, 1, 1),
            x
        ), axis=1).permute(1, 0, 2)
        # 在 se_length 维度上添加 muQuery 和 sigmaQuery, 然后permute, xseq shape is (se_length+2, batch_size, latent_dim), 前两个维度是 mu 和 sigma
        xseq = self.sequence_pos_encoder(xseq)

        # transformer layers
        # TODO: if not using MLP, try add random into mu and logvar
        xseq = self.seqTransEncoder(xseq) # (se_length+2, batch_size, latent_dim), TransformerEncoder does not change the shape?
        mu = xseq[0] # mu shape is (batch_size, latent_dim)
        logvar = xseq[1] # logvar shape is (batch_size, latent_dim)
        return mu, logvar, xseq



class Decoder_TRANSFORMER(nn.Module):
    def __init__(self,  dim_in=3, points=10000, seq_len=50,
                 c_dim=16, z_dim=32, ff_size=1024, num_layers=4, num_heads=4, dropout=0.1,
                 ablation=None, activation="gelu"):
        super().__init__()


        self.njoints = points
        self.nfeats = dim_in # 3D coordinates
        self.num_frames = seq_len
        self.num_classes = c_dim # projected conditioning vector dim

        self.latent_dim = z_dim

        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout

        self.ablation = ablation

        self.activation = activation

        # self.input_feats = self.njoints * self.nfeats
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
        ) # 利用输入 的 32 维 latent vector 生成 3D mesh 的坐标点，输出是 points * 3 维的向量
    def forward(self, z, con):
        latent_dim = z.shape[1] # z shape is (batch_size, latent_dim)
        bs = con.shape[0] # con shape is (batch_size, c_dim)
        nframes = self.num_frames
        njoints, nfeats = self.njoints, self.nfeats

        # so the logic here, is concatenate the latent vector z with the conditioning vector con
        # then project it to the latent_dim
        # this process is similar to the encoder, but the encoder is from 256+c_dim to latent_dim
        # here is from latent_dim + c_dim to latent_dim
        z = torch.cat((z, con), axis=1) # z shape is (batch_size, latent_dim + c_dim)
        z = self.ztimelinear(z) # project to latent_dim
        z = z[None]  # sequence of size 1, change dim to (1, batch_size, latent_dim)

        timequeries = torch.zeros(nframes, bs, latent_dim, device=z.device) # (se_length, batch_size, latent_dim)
        timequeries = self.sequence_pos_encoder(timequeries) # encoder 中，pose encoder 的输入是 (se_length+2, batch_size, latent_dim)，similar here
        # tgt: the sequence to the decoder.
        # memory: the sequence from the last layer of the encoder.
        output_seq = self.seqTransDecoder(tgt=timequeries, memory=z) # shape is (se_length, batch_size, latent_dim), 经过decoder 解码，不改变形状，理论上应该返回的是可以直接用于生成点云的 latent vector

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
        # 根据 均值 mu 和 logVar 生成符合目标正态分布的随机变量
        return eps.mul(std).add_(mu) # shape is (batch_size, latent_dim), 32 维的 latent vector

    def forward(self, v, f, edge_list, c):
        # v: batch, time_frames, points, channel
        con = self.mapping(c) # con shape is (batch_size, c_dim)
        # con = torch.argmax(c, axis=1)
        mu, logvar, xseq = self.encoder(v, f, edge_list, con)
        z = self.reparameterize(mu, logvar) # 生成符合目标正态分布的随机变量

        # decode
        v_all, v_all_latent = self.decoder(z, con)# torch.Size([1, 50, 22043, 3])
        return v_all, logvar, mu

    def return_latent(self, v, f, edge_list, c):
        # 训练完成之后，给定任意的 v, f, edge_list 和 c，可以返回 latent vector
        con = self.mapping(c)
        # con = torch.argmax(c, axis=1)
        mu, logvar, xseq = self.encoder(v, f, edge_list, con)
        z = self.reparameterize(mu, logvar)
        return z, xseq,con

    def generate_one(self, z, con):#, device, duration=50, fact=1):
        # 利用 return_latent 返回的 latent vector 和条件向量 con 生成新的 mesh
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