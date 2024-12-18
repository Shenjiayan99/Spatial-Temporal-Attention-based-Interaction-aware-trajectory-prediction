from __future__ import print_function, division
from torch.utils.data import Dataset, DataLoader
import scipy.io as scp
import numpy as np
import torch
from torch_geometric.data import Data,Batch
from torch_geometric.utils import erdos_renyi_graph,is_undirected,to_undirected
#___________________________________________________________________________________________________________________________

### Dataset class for the NGSIM dataset
class ngsimDataset(Dataset):

    def __init__(self, mat_file, file, t_h=30, t_f=50, d_s=2, enc_size=64, grid_size=(13, 3)):  #
        self.T = scp.loadmat(mat_file)['tracks']
        self.t_h = t_h  # length of track history
        self.t_f = t_f  # length of predicted trajectory
        self.d_s = d_s  # down sampling rate of all sequences
        self.enc_size = enc_size  # size of encoder LSTM
        self.grid_size = grid_size  # size of social context grid
        self.vehicle_on_ramp_id = np.load(file)  # 匝道上的车辆

        D = scp.loadmat(mat_file)['traj']
        vehicle_id = D[:, 1].squeeze()
        ds_id = D[:, 0].squeeze()
        mask = ~np.logical_and(np.isin(ds_id, self.vehicle_on_ramp_id[:, 0]),np.isin(vehicle_id, self.vehicle_on_ramp_id[:, 1]))
        index = np.where(mask)[0]
        self.D = D[index, :]


        # # 样本均衡
        # LK = np.where(self.D[:, 16] == 1)[0]
        # LLC = np.where(self.D[:, 16] == 2)[0]
        # # RLC = np.where(self.D[:, 16] == 3)[0]
        # # min_ = np.min([len(LK), len(LLC), len(RLC)])
        # # LK_index = np.random.choice(LK, min_, replace=False)
        # # LLC_index = np.random.choice(LLC, min_, replace=False)
        # # RLC_index = np.random.choice(RLC, min_, replace=False)
        # # index = LK_index.tolist() + LLC_index.tolist() + RLC_index.tolist()
        # self.D = self.D[LLC, :]
        print(len(self.D))

    def __len__(self):
        return len(self.D)

    def __getitem__(self, idx):#####按索引获取数据

        dsId = self.D[idx, 0].astype(int)
        vehId = self.D[idx, 1].astype(int)

        # if vehId in self.vehicle_on_ramp_id:#判断车辆是否是匝道车辆
        #     return None
        # else:
        t = self.D[idx, 2]
        grid = self.D[idx, 18:]
        neighbors = []
        neighbors_va = []
        neighbors_type = []
        neighbors_dis = []
        neighbors_lane = []

        # Get track history 'hist' = ndarray, and future track 'fut' = ndarray
        hist = self.getHistory(vehId, t, vehId, dsId)  # 提取自车的历史轨迹
        hist_va = self.get_hist_VA(vehId, t, dsId)
        hist_type = self.get_hist_type(vehId, t, dsId)
        hist_dis = self.get_hist_dis(vehId, t, dsId)
        hist_lane = self.get_hist_lane(vehId, t, dsId)
        fut = self.getFuture(vehId, t, dsId)

        # Get track histories of all neighbours 'neighbors' = [ndarray,[],ndarray,ndarray]
        for i in grid:
            neighbors.append(self.getHistory(i.astype(int), t, vehId, dsId))
            neighbors_va.append(self.get_hist_VA(i.astype(int), t, dsId))
            neighbors_type.append(self.get_hist_type(i.astype(int), t, dsId))
            neighbors_dis.append(self.get_hist_dis(i.astype(int), t, dsId))
            neighbors_lane.append(self.get_hist_lane(i.astype(int), t, dsId))

        neighbors_len = sum([len(neighbors[i]) != 0 for i in range(len(neighbors))])
        
        # Maneuvers 'lon_enc' = one-hot vector, 'lat_enc = one-hot vector
        # traj列名 ： 0：Dataset Id     1: Vehicle Id    2: Frame Number   3: Local X    4: Local Y     5: Length    # 6: Width  7: Class                     8: Vel    9: Acc   10: Lane Id    11:匝道标签   12：换道压力   13:区域标签  #14: Lateral maneuver   15: Longitudinal maneuver                16-55 Neighbor Car Ids at grid location
        lon_enc = np.zeros([2])
        lon_enc[int(self.D[idx, 17] - 1)] = 1
        lat_enc = np.zeros([3])
        lat_enc[int(self.D[idx, 16] - 1)] = 1
        return hist, hist_va, hist_type, hist_dis, hist_lane, fut, neighbors, neighbors_va, neighbors_type,  neighbors_dis, neighbors_lane, neighbors_len, lat_enc, lon_enc
    #####加特征需要修改

    # trackl列名：0: Frame Number  1: Local X  2: Local Y   3: Length   4: Width  5: Class  6: Vel    7: Acc  8:匝道标签   9：换道压力  10:区域标签
 ## Helper function to get v and a
    def get_hist_VA(self, vehId, t, dsId):
        if vehId == 0:
            return np.empty([0, 2])
        else:
            if self.T.shape[1] <= vehId - 1:
                return np.empty([0, 2])
            vehTrack = self.T[dsId - 1][vehId - 1].transpose()

            if vehTrack.size == 0 or np.argwhere(vehTrack[:, 0] == t).size == 0:
                return np.empty([0, 2])
            else:
                stpt = np.maximum(0, np.argwhere(vehTrack[:, 0] == t).item() - self.t_h)
                enpt = np.argwhere(vehTrack[:, 0] == t).item() + 1
                hist_va = vehTrack[stpt:enpt:self.d_s, 6:8]

            if len(hist_va) < self.t_h // self.d_s + 1:
                return np.empty([0, 2])
            return hist_va

    ## Helper function to get length and width and class
    def get_hist_type(self, vehId, t, dsId):
        if vehId == 0:
            return np.empty([0, 3])
        else:
            if self.T.shape[1] <= vehId - 1:
                return np.empty([0, 3])
            vehTrack = self.T[dsId - 1][vehId - 1].transpose()

            if vehTrack.size == 0 or np.argwhere(vehTrack[:, 0] == t).size == 0:
                return np.empty([0, 3])
            else:
                stpt = np.maximum(0, np.argwhere(vehTrack[:, 0] == t).item() - self.t_h)
                enpt = np.argwhere(vehTrack[:, 0] == t).item() + 1
                hist_type = vehTrack[stpt:enpt:self.d_s, 3:6]

            if len(hist_type) < self.t_h // self.d_s + 1:
                return np.empty([0, 3])
            return hist_type

        
    def get_hist_lab(self, vehId, t, dsId):
        if vehId == 0:
            return np.empty([0, 1])
        else:
            if self.T.shape[1] <= vehId - 1:
                return np.empty([0, 1])
            vehTrack = self.T[dsId - 1][vehId - 1].transpose()

            if vehTrack.size == 0 or np.argwhere(vehTrack[:, 0] == t).size == 0:
                return np.empty([0, 1])
            else:
                stpt = np.maximum(0, np.argwhere(vehTrack[:, 0] == t).item() - self.t_h)
                enpt = np.argwhere(vehTrack[:, 0] == t).item() + 1
                hist_lab = vehTrack[stpt:enpt:self.d_s, 8:9]

            if len(hist_lab) < self.t_h // self.d_s + 1:
                return np.empty([0, 1])
            return hist_lab

    def get_hist_dis(self, vehId, t, dsId):
        if vehId == 0:
            return np.empty([0, 1])
        else:
            if self.T.shape[1] <= vehId - 1:
                return np.empty([0, 1])
            vehTrack = self.T[dsId - 1][vehId - 1].transpose()

            if vehTrack.size == 0 or np.argwhere(vehTrack[:, 0] == t).size == 0:
                return np.empty([0, 1])
            else:
                stpt = np.maximum(0, np.argwhere(vehTrack[:, 0] == t).item() - self.t_h)
                enpt = np.argwhere(vehTrack[:, 0] == t).item() + 1
                hist_dis = vehTrack[stpt:enpt:self.d_s, 9:10]

            if len(hist_dis) < self.t_h // self.d_s + 1:
                return np.empty([0, 1])
            return hist_dis

    def get_hist_area(self, vehId, t, dsId):
        if vehId == 0:
            return np.empty([0, 3])
        else:
            if self.T.shape[1] <= vehId - 1:
                return np.empty([0, 3])
            vehTrack = self.T[dsId - 1][vehId - 1].transpose()

            if vehTrack.size == 0 or np.argwhere(vehTrack[:, 0] == t).size == 0:
                return np.empty([0, 3])
            else:
                stpt = np.maximum(0, np.argwhere(vehTrack[:, 0] == t).item() - self.t_h)
                enpt = np.argwhere(vehTrack[:, 0] == t).item() + 1
                hist_area = vehTrack[stpt:enpt:self.d_s, 10:11].astype(int)
                # print('hist_area',hist_area)
                hist_area_ = np.zeros((len(hist_area),3))  ####将hist_area改成独热编码
                hist_area_[np.arange(len(hist_area)), hist_area.flatten()-1] = 1
                # print('hist_area_',hist_area_)
                # for i in range(len(hist_area)):
                #     hist_area_[i, hist_area[i,0]-1] = 1
                hist_area = hist_area_

            if len(hist_area) < self.t_h // self.d_s + 1:
                return np.empty([0, 3])

            return hist_area

    def get_hist_lane(self, vehId, t, dsId):
        if vehId == 0:
            return np.empty([0, 2])
        else:
            if self.T.shape[1] <= vehId - 1:
                return np.empty([0, 2])
            vehTrack = self.T[dsId - 1][vehId - 1].transpose()

            if vehTrack.size == 0 or np.argwhere(vehTrack[:, 0] == t).size == 0:
                return np.empty([0, 2])
            else:
                stpt = np.maximum(0, np.argwhere(vehTrack[:, 0] == t).item() - self.t_h)
                enpt = np.argwhere(vehTrack[:, 0] == t).item() + 1
                hist_lane = vehTrack[stpt:enpt:self.d_s, 11:13]
            if len(hist_lane) < self.t_h // self.d_s + 1:
                return np.empty([0, 2])
            return hist_lane
    
#     # 提取周围车辆的匝道标签以及换道压力,在tracks中处于第9、10列
#     def get_hist_lab_dis_area(self, vehId, t, dsId):
#         if vehId == 0:
#             return np.empty([0, 3])
#         else:
#             if self.T.shape[1] <= vehId - 1:
#                 return np.empty([0, 3])
#             vehTrack = self.T[dsId - 1][vehId - 1].transpose()

#             if vehTrack.size == 0 or np.argwhere(vehTrack[:, 0] == t).size == 0:
#                 return np.empty([0, 3])
#             else:
#                 stpt = np.maximum(0, np.argwhere(vehTrack[:, 0] == t).item() - self.t_h)
#                 enpt = np.argwhere(vehTrack[:, 0] == t).item() + 1
#                 hist_lab_dis_area = vehTrack[stpt:enpt:self.d_s, 8:11]

#             if len(hist_lab_dis_area) < self.t_h // self.d_s + 1:
#                 return np.empty([0, 3])
#             return hist_lab_dis_area
        
    ## Helper function to get track history
    def getHistory(self, vehId, t, refVehId, dsId):
        if vehId == 0:
            return np.empty([0, 2])
        else:
            if self.T.shape[1] <= vehId - 1:  # 以上是筛选掉不存在或者id不正确的车辆
                return np.empty([0, 2])
            refTrack = self.T[dsId - 1][refVehId - 1].transpose()  # 从tracks中提取自身车辆轨迹，每一行代表一个时间点的轨迹信息
            vehTrack = self.T[dsId - 1][vehId - 1].transpose()  # 提取周围车辆轨迹
            refPos = refTrack[np.where(refTrack[:, 0] == t)][0, 1:3]  # 获取自身车辆在时间点t时的位置（localx，localy）

            if vehTrack.size == 0 or np.argwhere(vehTrack[:, 0] == t).size == 0:
                return np.empty([0, 2])
            else:
                stpt = np.maximum(0, np.argwhere(vehTrack[:, 0] == t).item() - self.t_h)  # 获取前三秒的索引
                enpt = np.argwhere(vehTrack[:, 0] == t).item() + 1  # 以上两行是为了提取时间点t前3秒到t这段时间的轨迹数据
                hist = vehTrack[stpt:enpt:self.d_s, 1:3] - refPos  # 提取并计算周围车辆相对目标车辆位置的前三秒历史轨迹，相对位置的变化，

            if len(hist) < self.t_h // self.d_s + 1:  # 小于16个采样点的话，也就是历史轨迹不足三秒，则返回空数组
                return np.empty([0, 2])
            return hist

    ## Helper function to get track future
    def getFuture(self, vehId, t, dsId):  #######同上，获取车辆未来三秒的轨迹
        vehTrack = self.T[dsId - 1][vehId - 1].transpose()
        refPos = vehTrack[np.where(vehTrack[:, 0] == t)][0, 1:3]
        stpt = np.argwhere(vehTrack[:, 0] == t).item() + self.d_s
        enpt = np.minimum(len(vehTrack), np.argwhere(vehTrack[:, 0] == t).item() + self.t_f + 1)
        fut = vehTrack[stpt:enpt:self.d_s, 1:3] - refPos
        return fut

    ## Collate function for dataloader
    def collate_fn(self, samples):

        # Initialize neighbors and neighbors length batches:
        nbr_batch_size = 0
        for  _, _, _, _, _, _, nbrs, nbrs_va,nbrs_type,nbrs_dis, nbrs_lane, _, _, _ in samples:
            nbr_batch_size += sum([len(nbrs[i]) != 0 for i in range(len(nbrs))])
        maxlen = self.t_h // self.d_s + 1
        nbrs_batch = torch.zeros(maxlen, nbr_batch_size, 2)
        nbrs_va_batch = torch.zeros(maxlen, nbr_batch_size, 2)
        nbrs_type_batch = torch.zeros(maxlen, nbr_batch_size, 3)
        nbrs_dis_batch = torch.zeros(maxlen, nbr_batch_size, 1)
        nbrs_lane_batch = torch.zeros(maxlen, nbr_batch_size, 2)

        # Initialize social mask batch:
        pos = [0, 0]
        mask_batch = torch.zeros(len(samples), self.grid_size[1], self.grid_size[0], self.enc_size)
        mask_batch = mask_batch.bool()

        # Initialize history, history lengths, future, output mask, lateral maneuver and longitudinal maneuver batches:
        hist_batch = torch.zeros(maxlen, len(samples), 2)
        hist_va_batch = torch.zeros(maxlen, len(samples), 2)
        hist_type_batch = torch.zeros(maxlen, len(samples), 3)
        hist_dis_batch = torch.zeros(maxlen, len(samples), 1)
        hist_lane_batch = torch.zeros(maxlen, len(samples), 2)
        nbrs_len_batch = torch.zeros(len(samples), 1)

        fut_batch = torch.zeros(self.t_f // self.d_s, len(samples), 2)
        op_mask_batch = torch.zeros(self.t_f // self.d_s, len(samples), 2)
        lat_enc_batch = torch.zeros(len(samples), 3)
        lon_enc_batch = torch.zeros(len(samples), 2)

        count = 0
        count_va = 0
        count_type = 0
        count_dis = 0
        count_lane = 0
#         
        for sampleId, (hist, hist_va, hist_type, hist_dis, hist_lane, fut, nbrs, nbrs_va, nbrs_type, nbrs_dis, nbrs_lane, nbrs_len, lat_enc, lon_enc) in enumerate(samples):

            # Set up history, future, lateral maneuver and longitudinal maneuver batches:
            hist_batch[0:len(hist), sampleId, 0] = torch.from_numpy(hist[:, 0])
            hist_batch[0:len(hist), sampleId, 1] = torch.from_numpy(hist[:, 1])
            
            hist_va_batch[0:len(hist_va), sampleId, 0] = torch.from_numpy(hist_va[:, 0])
            hist_va_batch[0:len(hist_va), sampleId, 1] = torch.from_numpy(hist_va[:, 1])

            hist_type_batch[0:len(hist_type), sampleId, 0] = torch.from_numpy(hist_type[:, 0])
            hist_type_batch[0:len(hist_type), sampleId, 1] = torch.from_numpy(hist_type[:, 1])
            hist_type_batch[0:len(hist_type), sampleId, 2] = torch.from_numpy(hist_type[:, 2])

            hist_dis_batch[0:len(hist_dis), sampleId, 0] = torch.from_numpy(hist_dis[:, 0])

            hist_lane_batch[0:len(hist_lane), sampleId, 0] = torch.from_numpy(hist_lane[:, 0])
            hist_lane_batch[0:len(hist_lane), sampleId, 1] = torch.from_numpy(hist_lane[:, 1])

            nbrs_len_batch[sampleId, 0] = nbrs_len

            fut_batch[0:len(fut), sampleId, 0] = torch.from_numpy(fut[:, 0])
            fut_batch[0:len(fut), sampleId, 1] = torch.from_numpy(fut[:, 1])
            op_mask_batch[0:len(fut), sampleId, :] = 1
            lat_enc_batch[sampleId, :] = torch.from_numpy(lat_enc)
            lon_enc_batch[sampleId, :] = torch.from_numpy(lon_enc)

            # Set up neighbor, neighbor sequence length, and mask batches:
            for id, nbr in enumerate(nbrs):
                if len(nbr) != 0:
                    nbrs_batch[0:len(nbr), count, 0] = torch.from_numpy(nbr[:, 0])
                    nbrs_batch[0:len(nbr), count, 1] = torch.from_numpy(nbr[:, 1])
                    pos[0] = id % self.grid_size[0]
                    pos[1] = id // self.grid_size[0]
                    mask_batch[sampleId, pos[1], pos[0], :] = torch.ones(self.enc_size).bool()
                    count += 1
            for id, nbr_va in enumerate(nbrs_va):
                if len(nbr_va) != 0:
                    nbrs_va_batch[0:len(nbr_va), count_va, 0] = torch.from_numpy(nbr_va[:, 0])
                    nbrs_va_batch[0:len(nbr_va), count_va, 1] = torch.from_numpy(nbr_va[:, 1])
                    count_va += 1
            for id, nbr_type in enumerate(nbrs_type):
                if len(nbr_type) != 0:
                    nbrs_type_batch[0:len(nbr_type), count_type, 0] = torch.from_numpy(nbr_type[:, 0])
                    nbrs_type_batch[0:len(nbr_type), count_type, 1] = torch.from_numpy(nbr_type[:, 1])
                    nbrs_type_batch[0:len(nbr_type), count_type, 2] = torch.from_numpy(nbr_type[:, 2])
                    count_type += 1
            for id, nbr_dis in enumerate(nbrs_dis):
                if len(nbr_dis) != 0:
                    nbrs_dis_batch[0:len(nbr_dis), count_dis, 0] = torch.from_numpy(nbr_dis[:, 0])
                    count_dis += 1
            for id, nbr_lane in enumerate(nbrs_lane):
                if len(nbr_lane) != 0:
                    nbrs_lane_batch[0:len(nbr_lane), count_lane, 0] = torch.from_numpy(nbr_lane[:, 0])
                    nbrs_lane_batch[0:len(nbr_lane), count_lane, 1] = torch.from_numpy(nbr_lane[:, 1])
                    count_lane += 1

        hist_all_batch = torch.cat((hist_batch, hist_va_batch, hist_dis_batch), 2)
        nbrs_all_batch = torch.cat((nbrs_batch, nbrs_va_batch, nbrs_dis_batch), 2)

        hist_all_batch_ = torch.cat((hist_batch, hist_va_batch), 2)
        nbrs_all_batch_ = torch.cat((nbrs_batch, nbrs_va_batch), 2)

        def create_graph(num_nodes, data):  ##根据节点数num_nodes和节点特征数据data构建单个图结构
            nodes = np.repeat(np.arange(num_nodes), repeats=num_nodes - 1)
            nodes = np.append(nodes, 0)
            targets = np.tile(np.arange(num_nodes), num_nodes)[np.arange(num_nodes * num_nodes) % num_nodes != np.repeat(np.arange(num_nodes), num_nodes)]
            targets = np.append(targets, 0)
            edge_index = np.array([nodes, targets])

            # edge_index = [list(range(1, num_nodes)),[0]*(num_nodes-1)]  #存储图中所有边的索引，每个边由一对节点索引组成，采用COO格式（坐标列表）：两个子列表分别存储边的起终点索引
            # for i in range(num_nodes):  #循环嵌套
            #     for j in range(i + 1, num_nodes):      #因为是无向图，节点与节点之间计算一次即可，避免重复计算
            #         x, y = data[i, 0:2], data[j, 0:2]  #根据第i个节点和第j个节点特征，计算节点间距离，hist_all的1、2列为hist，即x，y位置
            #         dis = pow(sum(pow((x-y),2)),0.5)*0.3048   #计算节点i和j之间的欧几里得距离，转化成米为单位
            #         if dis <= 30:  #如果两点之间的距离小于30米，则认为这两个节点之间存在一条边
            #             edge_index[0].append(i)
            #             edge_index[1].append(j)  ##将这条边的起点i和终点j分别添加到edge_index的两个子列表中

            edge_index = torch.LongTensor(edge_index)  #将edge_index中列表转换为pytorch的longtensor类型，以便用于图神经网络中
            graph = Data(x=data, edge_index=edge_index)  #创建一个Data对象，其中x是节点特征
            # graph.edge_index = to_undirected(graph.edge_index, num_nodes=num_nodes)  #使用to_undirected确保创建的图是无向的，即每条边无向性
            return graph
        
        #搭建16*128个图结构
        data = []  #构建批量处理图数据的过程，初始化一个data存放所有的图结构
        for m in range(16):  #遍历每个时间步
            j = 0  #初始化j，用于追踪每个样本中邻居数据的起始位置
            for i in range(len(samples)):  ##构建一个时间步内所有样本的图结构
                x_hist = hist_all_batch_[m][i].view(1,len(hist_all_batch_[m][i]))
                x_nbr = nbrs_all_batch_[m][j:j+int(nbrs_len_batch[i].item()),:]  #提取第m个时间步中第i个样本的历史信息和邻居信息
                j = j + int(nbrs_len_batch[i].item())
                x = torch.cat((x_hist,x_nbr),0)  #拼接当前样本的历史信息和邻居信息，形成完整的特征矩阵x，
                data.append(create_graph(len(x),x))  #调用以上函数，为当前的特征矩阵创建一个图结构，一个batch是128个样本
        data_figure_batch = Batch.from_data_list(data)  #利用Batch.from_data_list函数，将所有创建的图结构data批量化处理
        batch_gat = data_figure_batch.batch
        # batch_gat = data_figure_batch.batch[0:int(len(data_figure_batch.batch)/16)] #从批量化的图结构中提取出批次信息，并按照实际需要进行切片处理。这里假设每16个图结构组成一个完整的批次，因此需要按照这个比例来调整。
        # batch_gat = batch_gat.repeat(1, 16)  #将批次信息重复16次，以匹配原始的时间步长
        # batch_gat = batch_gat.view(-1)  #将批次信息展平，用于后续与模型输出对齐
#         data_x = data_figure_batch.x[:,:2]
#         data_edge = data_figure_batch.edge_index
        
#这段代码的目的是利用历史轨迹信息和周围车辆的邻居信息构建一系列图结构，每个图代表了某个时间步长内的车辆及其相互之间的关系。通过图神经网络，模型可以学习到车辆间的相互作用，进而更准确地预测未来的轨迹。这个过程首先是单独为每个样本创建图结构，然后将这些图结构批量化处理，使其适用于批处理操 作，最终得到的batch_gat用于指示每个图结构在批次中的位置，以便模型能够识别并正确处理。
        
        return data_figure_batch.x, data_figure_batch.edge_index, batch_gat, hist_batch, hist_va_batch, hist_type_batch, hist_dis_batch, hist_lane_batch, hist_all_batch, nbrs_batch, nbrs_va_batch, nbrs_type_batch, nbrs_dis_batch, nbrs_lane_batch, nbrs_len_batch, nbrs_all_batch, mask_batch, lat_enc_batch, lon_enc_batch, fut_batch, op_mask_batch

        

    #________________________________________________________________________________________________________________________________________

## Custom activation for output layer (Graves, 2015)
def outputActivation(x):
    muX = x[:,:,0:1]
    muY = x[:,:,1:2]
    sigX = x[:,:,2:3]
    sigY = x[:,:,3:4]
    rho = x[:,:,4:5]
    sigX = torch.exp(sigX)
    sigY = torch.exp(sigY)
    rho = torch.tanh(rho)
    out = torch.cat([muX, muY, sigX, sigY, rho],dim=2)
    return out

## Batchwise NLL loss, uses mask for variable output lengths
def maskedNLL(y_pred, y_gt, mask):
    acc = torch.zeros_like(mask)
    muX = y_pred[:,:,0]
    muY = y_pred[:,:,1]
    sigX = y_pred[:,:,2]
    sigY = y_pred[:,:,3]
    rho = y_pred[:,:,4]
    ohr = torch.pow(1-torch.pow(rho,2),-0.5)
    x = y_gt[:,:, 0]
    y = y_gt[:,:, 1]
    # If we represent likelihood in feet^(-1):
    out = 0.5*torch.pow(ohr, 2)*(torch.pow(sigX, 2)*torch.pow(x-muX, 2) + torch.pow(sigY, 2)*torch.pow(y-muY, 2) - 2*rho*torch.pow(sigX, 1)*torch.pow(sigY, 1)*(x-muX)*(y-muY)) - torch.log(sigX*sigY*ohr) + 1.8379
    # If we represent likelihood in m^(-1):
    # out = 0.5 * torch.pow(ohr, 2) * (torch.pow(sigX, 2) * torch.pow(x - muX, 2) + torch.pow(sigY, 2) * torch.pow(y - muY, 2) - 2 * rho * torch.pow(sigX, 1) * torch.pow(sigY, 1) * (x - muX) * (y - muY)) - torch.log(sigX * sigY * ohr) + 1.8379 - 0.5160
    acc[:,:,0] = out
    acc[:,:,1] = out
    acc = acc*mask
    lossVal = torch.sum(acc)/torch.sum(mask)
    return lossVal

## NLL for sequence, outputs sequence of NLL values for each time-step, uses mask for variable output lengths, used for evaluation
def maskedNLLTest(fut_pred, lat_pred, lon_pred, fut, op_mask, num_lat_classes=3, num_lon_classes = 2,use_maneuvers = True, avg_along_time = False):
    if use_maneuvers:
        acc = torch.zeros(op_mask.shape[0],op_mask.shape[1],num_lon_classes*num_lat_classes).cuda()
        count = 0
        for k in range(num_lon_classes):
            for l in range(num_lat_classes):
                wts = lat_pred[:,l]*lon_pred[:,k]
                wts = wts.repeat(len(fut_pred[0]),1)
                y_pred = fut_pred[k*num_lat_classes + l]
                y_gt = fut
                muX = y_pred[:, :, 0]
                muY = y_pred[:, :, 1]
                sigX = y_pred[:, :, 2]
                sigY = y_pred[:, :, 3]
                rho = y_pred[:, :, 4]
                ohr = torch.pow(1 - torch.pow(rho, 2), -0.5)
                x = y_gt[:, :, 0]
                y = y_gt[:, :, 1]
                # If we represent likelihood in feet^(-1):
                out = -(0.5*torch.pow(ohr, 2)*(torch.pow(sigX, 2)*torch.pow(x-muX, 2) + 0.5*torch.pow(sigY, 2)*torch.pow(y-muY, 2) - rho*torch.pow(sigX, 1)*torch.pow(sigY, 1)*(x-muX)*(y-muY)) - torch.log(sigX*sigY*ohr) + 1.8379)
                # If we represent likelihood in m^(-1):
                # out = -(0.5 * torch.pow(ohr, 2) * (torch.pow(sigX, 2) * torch.pow(x - muX, 2) + torch.pow(sigY, 2) * torch.pow(y - muY, 2) - 2 * rho * torch.pow(sigX, 1) * torch.pow(sigY, 1) * (x - muX) * (y - muY)) - torch.log(sigX * sigY * ohr) + 1.8379 - 0.5160)
                acc[:, :, count] =  out + torch.log(wts)
                count+=1
        acc = -logsumexp(acc, dim = 2)
        acc = acc * op_mask[:,:,0]
        if avg_along_time:
            lossVal = torch.sum(acc) / torch.sum(op_mask[:, :, 0])
            return lossVal
        else:
            lossVal = torch.sum(acc,dim=1)
            counts = torch.sum(op_mask[:,:,0],dim=1)
            return lossVal,counts
    else:
        acc = torch.zeros(op_mask.shape[0], op_mask.shape[1], 1).cuda()
        y_pred = fut_pred
        y_gt = fut
        muX = y_pred[:, :, 0]
        muY = y_pred[:, :, 1]
        sigX = y_pred[:, :, 2]
        sigY = y_pred[:, :, 3]
        rho = y_pred[:, :, 4]
        ohr = torch.pow(1 - torch.pow(rho, 2), -0.5)
        x = y_gt[:, :, 0]
        y = y_gt[:, :, 1]
        # If we represent likelihood in feet^(-1):
        out = 0.5*torch.pow(ohr, 2)*(torch.pow(sigX, 2)*torch.pow(x-muX, 2) + torch.pow(sigY, 2)*torch.pow(y-muY, 2) - 2 * rho*torch.pow(sigX, 1)*torch.pow(sigY, 1)*(x-muX)*(y-muY)) - torch.log(sigX*sigY*ohr) + 1.8379
        # If we represent likelihood in m^(-1):
        # out = 0.5 * torch.pow(ohr, 2) * (torch.pow(sigX, 2) * torch.pow(x - muX, 2) + torch.pow(sigY, 2) * torch.pow(y - muY, 2) - 2 * rho * torch.pow(sigX, 1) * torch.pow(sigY, 1) * (x - muX) * (y - muY)) - torch.log(sigX * sigY * ohr) + 1.8379 - 0.5160
        acc[:, :, 0] = out
        acc = acc * op_mask[:, :, 0:1]
        if avg_along_time:
            lossVal = torch.sum(acc[:, :, 0]) / torch.sum(op_mask[:, :, 0])
            return lossVal
        else:
            lossVal = torch.sum(acc[:,:,0], dim=1)
            counts = torch.sum(op_mask[:, :, 0], dim=1)
            return lossVal,counts

## Batchwise MSE loss, uses mask for variable output lengths
def maskedMSE(y_pred, y_gt, mask):
    acc = torch.zeros_like(mask)
    muX = y_pred[:,:,0]
    muY = y_pred[:,:,1]
    x = y_gt[:,:, 0]
    y = y_gt[:,:, 1]
    out = torch.pow(x-muX, 2) + torch.pow(y-muY, 2)
    acc[:,:,0] = out
    acc[:,:,1] = out
    acc = acc*mask
    lossVal = torch.sum(acc)/torch.sum(mask)
    return lossVal

## MSE loss for complete sequence, outputs a sequence of MSE values, uses mask for variable output lengths, used for evaluation
def maskedMSETest(y_pred, y_gt, mask):
    acc = torch.zeros_like(mask)
    muX = y_pred[:, :, 0] * 0.3048
    muY = y_pred[:, :, 1] * 0.3048
    x = y_gt[:, :, 0] * 0.3048
    y = y_gt[:, :, 1] * 0.3048
    out = torch.pow(x - muX, 2) + torch.pow(y - muY, 2)
    # print(out.shape)
    acc[:, :, 0] = out
    acc[:, :, 1] = out
    acc = acc * mask   #mask的形状是（25，128，2），里面的值是（1/0，1/0），表示是否是有效样本，acc就是根据mask有效的值就保留，无效的值就为0.[ [[out,out],[],……[]],[[....]],...,[[..]] ]

    lossVal = torch.sum(acc[:,:,0],dim=1)
    counts = torch.sum(mask[:,:,0],dim=1)

    ade = torch.sum(torch.pow(out, 0.5)) / torch.sum(counts)

    fde = torch.sum(torch.pow(out[-1, :], 0.5)) / torch.sum(counts[-1])
    return lossVal, counts, ade, fde



## Helper function for log sum exp calculation:
def logsumexp(inputs, dim=None, keepdim=False):
    if dim is None:
        inputs = inputs.view(-1)
        dim = 0
    s, _ = torch.max(inputs, dim=dim, keepdim=True)
    outputs = s + (inputs - s).exp().sum(dim=dim, keepdim=True).log()
    if not keepdim:
        outputs = outputs.squeeze(dim)
    return outputs
