import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.modules import TimeEncoder, MergeLayer, MultiHeadAttention
from utils.utils import NeighborSampler


class TGAT(nn.Module):

    def __init__(self, node_raw_features: np.ndarray, edge_raw_features: np.ndarray, neighbor_sampler: NeighborSampler,
                 time_feat_dim: int, num_layers: int = 2, num_heads: int = 2, dropout: float = 0.1, device: str = 'cpu'):
        """
        TGAT model.
        :param node_raw_features: ndarray, shape (num_nodes + 1, node_feat_dim)
        :param edge_raw_features: ndarray, shape (num_edges + 1, edge_feat_dim)
        :param neighbor_sampler: neighbor sampler
        :param time_feat_dim: int, dimension of time features (encodings)
        :param num_layers: int, number of temporal graph convolution layers
        :param num_heads: int, number of attention heads
        :param dropout: float, dropout rate
        :param device: str, device
        """
        super(TGAT, self).__init__()

        #self.node_raw_features = nn.Parameter(torch.from_numpy(node_raw_features.astype(np.float32)), requires_grad = True).to(device)
        #self.edge_raw_features = nn.Parameter(torch.from_numpy(edge_raw_features.astype(np.float32)), requires_grad = True).to(device)

        self.node_raw_features = torch.from_numpy(node_raw_features.astype(np.float32)).to(device)
        self.edge_raw_features = torch.from_numpy(edge_raw_features.astype(np.float32)).to(device)

        self.neighbor_sampler = neighbor_sampler
        self.node_feat_dim = self.node_raw_features.shape[1]
        self.edge_feat_dim = self.edge_raw_features.shape[1]
        self.time_feat_dim = time_feat_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.device = device

        self.time_encoder = TimeEncoder(time_dim=time_feat_dim)

        self.temporal_conv_layers = nn.ModuleList([MultiHeadAttention(node_feat_dim=self.node_feat_dim,
                                                                      edge_feat_dim=self.edge_feat_dim,
                                                                      time_feat_dim=self.time_feat_dim,
                                                                      num_heads=self.num_heads,
                                                                      dropout=self.dropout) for _ in range(num_layers)])
        # follow the TGAT paper, use merge layer to combine the attention results and node original feature
        self.merge_layers = nn.ModuleList([MergeLayer(input_dim1=self.node_feat_dim + self.time_feat_dim, input_dim2=self.node_feat_dim,
                                                      hidden_dim=self.node_feat_dim, output_dim=self.node_feat_dim) for _ in range(num_layers)])

        
#         self.merge_layers_time =  nn.Linear(self.time_feat_dim, self.node_feat_dim, bias=True)
#         self.merge_layers_text =  nn.Linear(self.node_feat_dim, self.node_feat_dim-self.time_feat_dim, bias=True)
        self.time_global_dim = 128
#         self.output_layer = nn.Linear(in_features=self.node_feat_dim + self.time_global_dim*2, out_features=self.node_feat_dim, bias=True)
#         self.node_time_projector =nn.Linear(self.time_feat_dim, self.time_global_dim)
        self.time_global = nn.Sequential(
            nn.Linear(self.time_feat_dim, self.time_global_dim),
            nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=self.time_global_dim,
                    nhead=8,
                    dim_feedforward=256,
                    dropout= self.dropout,
                    activation= 'gelu',
                    batch_first=True,
                ),
                num_layers= 2, #args.layer_num,
#                 norm=nn.LayerNorm(self.time_global_dim)
            )  
        )
        self.text_global = nn.Sequential(
            nn.Linear(self.node_feat_dim, self.time_global_dim),
            nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=self.time_global_dim,
                    nhead=8,
                    dim_feedforward=256,
                    dropout= self.dropout,
                    activation= 'gelu',  #'gelu',
                    batch_first=True,
                ),
                num_layers= 2, #args.layer_num,
#                 norm=nn.LayerNorm(self.time_global_dim)
            )
        )
        self.output_layer = nn.Linear(in_features=self.node_feat_dim  + self.time_global_dim*2, out_features=self.node_feat_dim, bias=True)
#         self.align_text =  nn.Linear(in_features=self.node_feat_dim-self.time_feat_dim + self.time_global_dim, out_features=self.node_feat_dim, bias=True)
                                     
#         self.align_time = nn.Linear(in_features=self.time_feat_dim+self.time_global_dim , out_features=self.time_feat_dim, bias=True)

    def symmetric_kl_divergence(self, p, q):
        # 数值稳定性优化：避免多次 softmax 调用
        p_log = F.log_softmax(p, dim=1)  # 使用 log_softmax 计算 log(p)
        q_log = F.log_softmax(q, dim=1)  # 使用 log_softmax 计算 log(q)
        p_prob = F.softmax(p, dim=1)     # 保持 p 的概率分布
        q_prob = F.softmax(q, dim=1)     # 保持 q 的概率分布

        # KL 散度计算：KL(p || q) 和 KL(q || p)
        kl_pq = torch.sum(p_prob * (p_log - q_log), dim=1)  # KL(p || q)
        kl_qp = torch.sum(q_prob * (q_log - p_log), dim=1)  # KL(q || p)

        # 对称 KL 散度
        symmetric_kl = kl_pq + kl_qp

        return torch.sigmoid(symmetric_kl.mean())
    
    
    def compute_src_dst_node_temporal_embeddings(self, src_node_ids: np.ndarray, dst_node_ids: np.ndarray,
                                                 node_interact_times: np.ndarray, num_neighbors: int = 20):
        """
        compute source and destination node temporal embeddings
        :param src_node_ids: ndarray, shape (batch_size, )
        :param dst_node_ids: ndarray, shape (batch_size, )
        :param node_interact_times: ndarray, shape (batch_size, )
        :param num_neighbors: int, number of neighbors to sample for each node
        :return:
        """
        # Tensor, shape (batch_size, node_feat_dim)
        src_node_embeddings = self.compute_node_temporal_embeddings(node_ids=src_node_ids, node_interact_times=node_interact_times,
                                                                    current_layer_num=self.num_layers, num_neighbors=num_neighbors)
        # Tensor, shape (batch_size, node_feat_dim)
        dst_node_embeddings = self.compute_node_temporal_embeddings(node_ids=dst_node_ids, node_interact_times=node_interact_times,
                                                                    current_layer_num=self.num_layers, num_neighbors=num_neighbors)
        

        src_raw_features = self.node_raw_features[torch.from_numpy(src_node_ids)]
        dst_raw_features = self.node_raw_features[torch.from_numpy(dst_node_ids)]
#         self.node_raw_features[torch.from_numpy(node_ids)]
        
        node_raw_time_input = torch.mean(self.time_encoder(timestamps=torch.from_numpy(node_interact_times[:, np.newaxis]).float().to(self.device)), dim =1)

        node_time_token = self.time_global(node_raw_time_input)
    
        node_text_input = torch.cat([src_raw_features, dst_raw_features], dim =0)
        node_text_token = self.text_global(node_text_input)
        node_time_token_sd = torch.cat([node_time_token, node_time_token], dim =0)
     
        
        src_node_embeddings = self.output_layer(torch.cat([src_node_embeddings, node_text_token[:len(src_raw_features)], node_time_token], dim =1))
        dst_node_embeddings = self.output_layer(torch.cat([dst_node_embeddings, node_text_token[len(src_raw_features):], node_time_token], dim =1))
        
        align_loss_term = self.symmetric_kl_divergence(node_text_token, node_time_token_sd)
        

        return src_node_embeddings, dst_node_embeddings, align_loss_term

    def compute_node_temporal_embeddings(self, node_ids: np.ndarray, node_interact_times: np.ndarray,
                                         current_layer_num: int, num_neighbors: int = 20):
        """
        given node ids node_ids, and the corresponding time node_interact_times,
        return the temporal embeddings after convolution at the current_layer_num
        :param node_ids: ndarray, shape (batch_size, ) or (*, ), node ids
        :param node_interact_times: ndarray, shape (batch_size, ) or (*, ), node interaction times
        :param current_layer_num: int, current layer number
        :param num_neighbors: int, number of neighbors to sample for each node
        :return:
        """
        assert current_layer_num >= 0
        device = self.node_raw_features.device

        # query (source) node always has the start time with time interval == 0
        # Tensor, shape (batch_size, 1, time_feat_dim)
        node_time_features = self.time_encoder(timestamps=torch.zeros(node_interact_times.shape).unsqueeze(dim=1).to(device))
        # Tensor, shape (batch_size, node_feat_dim)
        node_raw_features = self.node_raw_features[torch.from_numpy(node_ids)]

        if current_layer_num == 0:
            return node_raw_features
        else:
            # get source node representations by aggregating embeddings from the previous (current_layer_num - 1)-th layer
            # Tensor, shape (batch_size, node_feat_dim)
            node_conv_features = self.compute_node_temporal_embeddings(node_ids=node_ids,
                                                                       node_interact_times=node_interact_times,
                                                                       current_layer_num=current_layer_num - 1,
                                                                       num_neighbors=num_neighbors)

            # get temporal neighbors, including neighbor ids, edge ids and time information
            # neighbor_node_ids, ndarray, shape (batch_size, num_neighbors)
            # neighbor_edge_ids, ndarray, shape (batch_size, num_neighbors)
            # neighbor_times, ndarray, shape (batch_size, num_neighbors)
            neighbor_node_ids, neighbor_edge_ids, neighbor_times = \
                self.neighbor_sampler.get_historical_neighbors(node_ids=node_ids,
                                                               node_interact_times=node_interact_times,
                                                               num_neighbors=num_neighbors)

            # get neighbor features from previous layers
            # shape (batch_size * num_neighbors, node_feat_dim)
            neighbor_node_conv_features = self.compute_node_temporal_embeddings(node_ids=neighbor_node_ids.flatten(),
                                                                                node_interact_times=neighbor_times.flatten(),
                                                                                current_layer_num=current_layer_num - 1,
                                                                                num_neighbors=num_neighbors)
            # shape (batch_size, num_neighbors, node_feat_dim)
            neighbor_node_conv_features = neighbor_node_conv_features.reshape(node_ids.shape[0], num_neighbors, self.node_feat_dim)

            # compute time interval between current time and historical interaction time
            # adarray, shape (batch_size, num_neighbors)
            neighbor_delta_times = node_interact_times[:, np.newaxis] - neighbor_times

            # shape (batch_size, num_neighbors, time_feat_dim)
            neighbor_time_features = self.time_encoder(timestamps=torch.from_numpy(neighbor_delta_times).float().to(device))

            # get edge features, shape (batch_size, num_neighbors, edge_feat_dim)
            neighbor_edge_features = self.edge_raw_features[torch.from_numpy(neighbor_edge_ids)]
            # temporal graph convolution
            # Tensor, output shape (batch_size, node_feat_dim + time_feat_dim)
            output, _ = self.temporal_conv_layers[current_layer_num - 1](node_features=node_conv_features,
                                                                         node_time_features=node_time_features,
                                                                         neighbor_node_features=neighbor_node_conv_features,
                                                                         neighbor_node_time_features=neighbor_time_features,
                                                                         neighbor_node_edge_features=neighbor_edge_features,
                                                                         neighbor_masks=neighbor_node_ids)

            # Tensor, output shape (batch_size, node_feat_dim)
            # follow the TGAT paper, use merge layer to combine the attention results and node original feature
            output = self.merge_layers[current_layer_num - 1](input_1=output, input_2=node_raw_features)

            return output
        
        
    def set_neighbor_sampler(self, neighbor_sampler: NeighborSampler):
        """
        set neighbor sampler to neighbor_sampler and reset the random state (for reproducing the results for uniform and time_interval_aware sampling)
        :param neighbor_sampler: NeighborSampler, neighbor sampler
        :return:
        """
        self.neighbor_sampler = neighbor_sampler
        if self.neighbor_sampler.sample_neighbor_strategy in ['uniform', 'time_interval_aware']:
            assert self.neighbor_sampler.seed is not None
            self.neighbor_sampler.reset_random_state()
