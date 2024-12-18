from __future__ import division
import torch
from torch.autograd import Variable
import torch.nn as nn
from utils_gat_features02 import outputActivation
from torch_geometric.utils import erdos_renyi_graph, is_undirected
from torch_geometric.data import Data
from torch_geometric.nn import GATConv
from torch_geometric.nn import global_mean_pool, global_max_pool
import numpy as np
from more_itertools import flatten


class TA_LSTM(nn.Module):

    def __init__(self,in_dim,sequence_length,out_dim,use_gpu=False):

        super(TA_LSTM, self).__init__()

        # 参数导入部分
        self.in_dim = in_dim
        self.sequence_length = sequence_length

        # self.lstm_in_dim = lstm_in_dim
        # self.lstm_hidden_dim = lstm_hidden_dim
        self.out_dim = out_dim
        self.use_gpu = use_gpu

        # 网络结构部分

        # batch_norm layer
        # self.batch_norm = nn.BatchNorm1d(in_dim)

        # input layer
        # self.layer_in = nn.Linear(in_dim, in_dim, bias=False)

        # lstmcell
        self.lstmcell = nn.LSTMCell(in_dim, out_dim)

        # temporal atteention module, 产生sequence_length个时间权重, 维度1 ×（lstm_hidden_dim + lstm_in_dim）-> 1 × sequence_length
        self.T_A = nn.Linear(sequence_length * out_dim, sequence_length)

        # # output layer, 维度 1 × lstm_hiddendim -> 1 × 1
        # self.layer_out = nn.Linear(lstm_hidden_dim, out_dim, bias=False)

        # activate functions
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim = -1)

    def forward(self, input):

        # 批归一化处理输入
        # out = self.batch_norm(input)
        # print('batch_norm',out.size())

        # 经过输入层处理
        out = input
        B = out.size(1)
        # print('layer_in',out.size())

        # 初始化隐藏状态与记忆单元状态
        h_t_1 = torch.zeros(out.size(1), self.out_dim).cuda()  # batch, hidden_size
        c_t_1 = torch.zeros(out.size(1), self.out_dim).cuda()  # batch, hidden_size

        # 创建一个列表，存储ht
        h_list = []

        for i in range(self.sequence_length):
            # x_t = out[i, :, i * self.lstm_in_dim:(i + 1) * (self.lstm_in_dim)]
            x_t = out[i, :, :]
            # x_t = out[i]

            # print(x_t.shape)    # (128 * 16)
            h_t, c_t = self.lstmcell(x_t, (h_t_1, c_t_1))

            h_list.append(h_t)

            h_t_1, c_t_1 = h_t, c_t

        output_using_lstm_cell = torch.stack(h_list)
        # print(output_using_lstm_cell.shape)

        total_ht = h_list[0]
        for i in range(1, len(h_list)):
            total_ht = torch.cat((total_ht, h_list[i]), 1)
        # print(total_ht.shape)               # (128*256)

        beta_t = self.relu(self.T_A(total_ht))
        beta_t = self.softmax(beta_t)
        # print(beta_t.shape)     # (128 * 16)
        out = torch.zeros(out.size(1), self.out_dim).cuda()
        # print(h_list[0].size(),beta_t[:,1].size())

        for i in range(len(h_list)):
            out = out + h_list[i] * beta_t[:, i].reshape(B, 1)
        # print(out.shape)            # 128*16


        return out


class highwayNet(nn.Module):

    ## Initialization
    def __init__(self, args):
        super(highwayNet, self).__init__()

        ## Unpack arguments
        self.args = args

        ## Use gpu flag
        self.use_cuda = args['use_cuda']

        # Flag for maneuver based (True) vs uni-modal decoder (False)
        self.use_maneuvers = args['use_maneuvers']

        # Flag for train mode (True) vs test-mode (False)
        self.train_flag = args['train_flag']

        ## Sizes of network layers
        self.encoder_size = args['encoder_size']
        self.decoder_size = args['decoder_size']
        self.gat_in_length = args['gat_in_length']
        self.gat_hide_length = args['gat_hide_length']
        self.gat_out_length = args['gat_out_length']
        #         self.gat_head = args['gat_head']
        self.in_length = args['in_length']
        self.out_length = args['out_length']
        self.grid_size = args['grid_size']
        self.soc_conv_depth = args['soc_conv_depth']
        self.conv_3x1_depth = args['conv_3x1_depth']
        self.dyn_embedding_size = args['dyn_embedding_size']
        self.input_embedding_size = args['input_embedding_size']
        self.dis_embedding_size = args['dis_embedding_size']

        self.num_lat_classes = args['num_lat_classes']
        self.num_lon_classes = args['num_lon_classes']
        self.soc_embedding_size = (((args['grid_size'][0] - 4) + 1) // 2) * self.conv_3x1_depth

        ## Define network weights
        # GAT layers
        self.gat1 = GATConv(self.gat_in_length, self.gat_hide_length)
        self.gat2 = GATConv(self.gat_hide_length, self.gat_out_length)
        # self.gat = GATConv(self.gat_in_length, self.gat_out_length, dropout=0.6)
        #         self.fc = torch.nn.Linear(self.gat_out_length,self.input_embedding_size)
        # Input embedding layer
        self.ip_emb = torch.nn.Linear(4, self.gat_in_length)
        self.ip_emb_pos = torch.nn.Linear(2, self.input_embedding_size)  ##将输入的2维数据转换成指定大小的嵌入向量，
        self.ip_emb_va = torch.nn.Linear(2, self.input_embedding_size)
        self.ip_emb_dis = torch.nn.Linear(1, self.dis_embedding_size)
        self.ip_emb_lane = torch.nn.Linear(2, self.input_embedding_size)
        self.dropout = torch.nn.Dropout(0.3)
        # Encoder LSTM

        total_embedding_size = self.input_embedding_size * 3 + self.dis_embedding_size + self.gat_out_length  # self.dis_embedding_size + 1+self.area_embedding_size
        self.enc_lstm = torch.nn.LSTM(total_embedding_size, self.encoder_size,1)  # self.encoder_size为隐藏状态的维度，1是代表lstm的层数，输出为隐藏状态
        self.TA_lstm = TA_LSTM(total_embedding_size,16, self.encoder_size)
        # self.enc_lstm_gat = torch.nn.LSTM(self.input_embedding_size, self.encoder_size,1)
        self.enc_lstm_gat2 = torch.nn.LSTM(self.encoder_size * 2, self.encoder_size * 2, 1)
        # Vehicle dynamics embedding
        self.dyn_emb = torch.nn.Linear(self.encoder_size, self.dyn_embedding_size)
        self.dyn_emb1 = torch.nn.Linear(self.gat_out_length, self.encoder_size)
        # Convolutional social pooling layer and social embedding layer
        self.soc_conv = torch.nn.Conv2d(self.encoder_size, self.soc_conv_depth, 3)
        self.conv_3x1 = torch.nn.Conv2d(self.soc_conv_depth, self.conv_3x1_depth, (3, 1))
        self.soc_maxpool = torch.nn.MaxPool2d((2, 1), padding=(1, 0))
        self.soc_avgpool = torch.nn.AvgPool2d((2, 1), padding=(1, 0))
        # FC social pooling layer (for comparison):
        # self.soc_fc = torch.nn.Linear(self.soc_conv_depth * self.grid_size[0] * self.grid_size[1], (((args['grid_size'][0]-4)+1)//2)*self.conv_3x1_depth)

        # Decoder LSTM
        if self.use_maneuvers:
            self.dec_lstm = torch.nn.LSTM(
                2 * self.soc_embedding_size + self.dyn_embedding_size + self.num_lat_classes + self.num_lon_classes,
                self.decoder_size)
        else:
            self.dec_lstm = torch.nn.LSTM(2 * self.soc_embedding_size + self.dyn_embedding_size, self.decoder_size)

        # Output layers:
        self.op = torch.nn.Linear(self.decoder_size, 5)
        self.op_lat = torch.nn.Linear(2 * self.soc_embedding_size + self.dyn_embedding_size, self.num_lat_classes)
        self.op_lon = torch.nn.Linear(2 * self.soc_embedding_size + self.dyn_embedding_size, self.num_lon_classes)

        # Activations:
        self.leaky_relu = torch.nn.LeakyReLU(0.1)
        self.relu = torch.nn.ReLU()
        self.tanh = torch.nn.Tanh()
        self.softmax = torch.nn.Softmax(dim=1)

    ## Forward Pass
    def forward(self, data_figure_x, data_figure_edge_index, batch_gat, hist, hist_va,  hist_dis, hist_lane, hist_all, nbrs, nbrs_va, nbrs_dis, nbrs_lane, nbrs_len, nbrs_all, masks, lat_enc, lon_enc):
        #         print(hist_all.shape)
        #         print(nbrs_all.shape)
        ##Apply GAT layers,GAT后需要增加激活函数，池化层后面不需要激活函数
        # data_gat = data_figure_x.view(16, -1, 9)
        # data_gat, (_, _) = self.enc_lstm_gat(self.leaky_relu(self.ip_emb(data_gat)))
        # data_gat = data_gat.view(-1, 64)
        data_gat = self.leaky_relu(self.ip_emb(data_figure_x))
        # data_gat = self.gat(data_gat, data_figure_edge_index)
        data_gat = self.gat1(data_gat, data_figure_edge_index)
        data_gat = self.leaky_relu(data_gat)
        data_gat = self.gat2(data_gat, data_figure_edge_index)
        data_gat = self.leaky_relu(data_gat)

        unique_elements, inverse_indices = torch.unique(batch_gat, return_inverse=True)
        indexes = [torch.where(inverse_indices == g)[0] for g in unique_elements]

        hist_f_indexes = torch.stack([mm[0] for mm in indexes])
        hist_gat = data_gat[hist_f_indexes]
        hist_gat = hist_gat.view(16, -1, self.gat_out_length)
        hist_gat = self.dropout(hist_gat)


        nbrs_f_indexes = [mm[1:] for mm in indexes]
        nbrs_f_indexes = torch.cat(nbrs_f_indexes, dim=0)
        nbrs_gat = data_gat[nbrs_f_indexes]
        nbrs_gat = nbrs_gat.view(16, -1, self.gat_out_length)
        nbrs_gat = self.dropout(nbrs_gat)

        hist_emb = self.leaky_relu(self.ip_emb_pos(hist))
        hist_va_emb = self.leaky_relu(self.ip_emb_va(hist_va))
        hist_dis_emb = self.leaky_relu(self.ip_emb_dis(hist_dis))  ######应用全连接层到换道压力
        hist_lane_emb = self.leaky_relu(self.ip_emb_lane(hist_lane))

        nbrs_emb = self.leaky_relu(self.ip_emb_pos(nbrs))
        nbrs_va_emb = self.leaky_relu(self.ip_emb_va(nbrs_va))
        nbrs_dis_emb = self.leaky_relu(self.ip_emb_dis(nbrs_dis))
        nbrs_lane_emb = self.leaky_relu(self.ip_emb_lane(nbrs_lane))

        combined_emb_hist = torch.cat((hist_emb, hist_va_emb, hist_dis_emb, hist_lane_emb, hist_gat), dim=-1)
        combined_emb_nbrs = torch.cat((nbrs_emb, nbrs_va_emb, nbrs_dis_emb, nbrs_lane_emb, nbrs_gat), dim=-1)

        ## Forward pass hist:
        # _, (hist_enc, _) = self.enc_lstm(combined_emb_hist)
        hist_enc = self.TA_lstm(combined_emb_hist)
        hist_enc = self.leaky_relu(self.dyn_emb(hist_enc))
        #         hist_enc = hist_enc.view(hist_enc.shape[1],hist_enc.shape[2])
        # hist_enc = self.leaky_relu(self.dyn_emb(hist_enc.view(hist_enc.shape[1], hist_enc.shape[2])))

        ## Forward pass nbrs
        _, (nbrs_enc, _) = self.enc_lstm(combined_emb_nbrs)
        # nbrs_enc = self.enc_lstm(combined_emb_nbrs)
        nbrs_enc = nbrs_enc.view(nbrs_enc.shape[1], nbrs_enc.shape[2])

        ## Masked scatter
        soc_enc = torch.zeros_like(masks).float()
        soc_enc = soc_enc.masked_scatter_(masks, nbrs_enc)
        soc_enc = soc_enc.permute(0, 3, 2, 1)

        ## Apply convolutional social pooling:
        soc_enc = self.leaky_relu(self.conv_3x1(self.leaky_relu(self.soc_conv(soc_enc))))
        soc_enc1 = self.soc_maxpool(soc_enc).view(-1, self.soc_embedding_size)
        soc_enc2 = self.soc_avgpool(soc_enc).view(-1, self.soc_embedding_size)
        soc_enc = torch.cat((soc_enc1, soc_enc2), 1)
        #         soc_enc = soc_enc.view(-1,self.soc_embedding_size)

        ## Apply fc soc pooling
        # soc_enc = soc_enc.contiguous()
        # soc_enc = soc_enc.view(-1, self.soc_conv_depth * self.grid_size[0] * self.grid_size[1])
        # soc_enc = self.leaky_relu(self.soc_fc(soc_enc))

        ## Concatenate encodings:
        enc = torch.cat((soc_enc, hist_enc), 1)
        # enc = torch.cat((enc_hr, gat_enc),1)

        if self.use_maneuvers:
            ## Maneuver recognition:
            lat_pred = self.softmax(self.op_lat(enc))
            lon_pred = self.softmax(self.op_lon(enc))

            if self.train_flag:
                ## Concatenate maneuver encoding of the true maneuver
                enc = torch.cat((enc, lat_enc, lon_enc), 1)
                fut_pred = self.decode(enc)
                return fut_pred, lat_pred, lon_pred
            else:
                fut_pred = []
                ## Predict trajectory distributions for each maneuver class
                for k in range(self.num_lon_classes):
                    for l in range(self.num_lat_classes):
                        lat_enc_tmp = torch.zeros_like(lat_enc)
                        lon_enc_tmp = torch.zeros_like(lon_enc)
                        lat_enc_tmp[:, l] = 1
                        lon_enc_tmp[:, k] = 1
                        enc_tmp = torch.cat((enc, lat_enc_tmp, lon_enc_tmp), 1)
                        fut_pred.append(self.decode(enc_tmp))
                return fut_pred, lat_pred, lon_pred
        else:
            fut_pred = self.decode(enc)
            return fut_pred

    def decode(self, enc):
        enc = enc.repeat(self.out_length, 1, 1)
        h_dec, _ = self.dec_lstm(enc)
        h_dec = h_dec.permute(1, 0, 2)
        fut_pred = self.op(h_dec)
        fut_pred = fut_pred.permute(1, 0, 2)
        fut_pred = outputActivation(fut_pred)
        return fut_pred
