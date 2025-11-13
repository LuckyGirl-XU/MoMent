import numpy as np
import torch
import torch.nn as nn

from models.modules import TimeEncoder
from utils.utils import NeighborSampler
import torch.nn.functional as F

from torch import Tensor
from torch.fft import fft, ifft
import math
import torch.fft

from transformers import AutoModel, AutoTokenizer, AutoConfig

class MoMent(nn.Module):

    def __init__(self, node_raw_features: np.ndarray, edge_raw_features: np.ndarray, neighbor_sampler: NeighborSampler,
                 time_feat_dim: int, num_heads: int = 2, num_layers: int = 1, dropout: float = 0.1, device: str = 'cpu'):
        """
        MoMent model.
        :param node_raw_features: ndarray, shape (num_nodes + 1, node_feat_dim)
        :param edge_raw_features: ndarray, shape (num_edges + 1, edge_feat_dim)
        :param neighbor_sampler: neighbor sampler
        :param time_feat_dim: int, dimension of time features (encodings)
        :param num_tokens: int, number of tokens
        :param num_layers: int, number of transformer layers
        :param token_dim_expansion_factor: float, dimension expansion factor for tokens
        :param channel_dim_expansion_factor: float, dimension expansion factor for channels
        :param dropout: float, dropout rate
        :param device: str, device
        """
        super(MoMent, self).__init__()

        #self.node_raw_features = nn.Parameter(torch.from_numpy(node_raw_features.astype(np.float32)), requires_grad = True).to(device)
        #self.edge_raw_features = nn.Parameter(torch.from_numpy(edge_raw_features.astype(np.float32)), requires_grad = True).to(device)
        
        self.node_raw_features = torch.from_numpy(node_raw_features.astype(np.float32)).to(device)
        self.edge_raw_features = torch.from_numpy(edge_raw_features.astype(np.float32)).to(device)

        self.neighbor_sampler = neighbor_sampler
        self.node_feat_dim = self.node_raw_features.shape[1]
        self.edge_feat_dim = self.edge_raw_features.shape[1]
        self.time_feat_dim = time_feat_dim
        self.num_layers = num_layers
        self.num_head = num_heads
        
        self.dropout = dropout
        self.device = device

        self.num_channels = self.edge_feat_dim
        self.time_encoder = TimeEncoder(time_dim=self.time_feat_dim, parameter_requires_grad=False)
        self.projection_layer = nn.Linear(self.edge_feat_dim + self.time_feat_dim, self.num_channels)

        self.time_global = nn.Sequential(
#             nn.Linear(self.node_feat_dim, self.time_feat_dim),
            nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=self.time_feat_dim,
                    nhead= self.num_head,
                    dim_feedforward=512,
                    dropout=self.dropout,
                    activation= 'gelu',  #'gelu',
                    batch_first=True,
                ),
                num_layers= self.num_layers, 
#                 norm=nn.LayerNorm(self.time_global_dim)
            )
        )
        self.text_global = nn.Sequential(
            nn.Linear(self.node_feat_dim, self.time_feat_dim),
            nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=self.time_feat_dim,
                    nhead= self.num_head,
                    dim_feedforward=512,
                    dropout=self.dropout,
                    activation= 'gelu',  #'gelu',
                    batch_first=True,
                ),
                num_layers= self.num_layers, 
#                 norm=nn.LayerNorm(self.time_global_dim)
            )
        )
        self.Structure = nn.Sequential(
#             nn.Linear(self.node_feat_dim, self.time_feat_dim),
            nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=self.node_feat_dim,
                    nhead=self.num_head,
                    dim_feedforward=512,
                    dropout=self.dropout,
                    activation= 'gelu',  #'gelu',
                    batch_first=True,
                ),
                num_layers= self.num_layers, 
#                 norm=nn.LayerNorm(self.time_global_dim)
            )
        )


        self.output_layer = nn.Linear(in_features=self.time_feat_dim, out_features=self.node_feat_dim, bias=True)
        
        self.beta = nn.Parameter(torch.ones(1))  
        self.alpha = nn.Parameter(torch.ones(1)) 
        
        self.rff = RFFMap(self.time_feat_dim, self.node_feat_dim, 1)


    def mmd_rff(self, temp, textual, rff):
        
        z_s = rff(temp)            
        z_n = rff(textual)  
        return (z_s.mean(0) - z_n.mean(0)).pow(2).sum()
    
    
    def compute_src_dst_node_temporal_embeddings(self, src_node_ids: np.ndarray, dst_node_ids: np.ndarray,
                                                 node_interact_times: np.ndarray, edge_ids: np.ndarray, num_neighbors: int = 20, time_gap: int = 2000):
        """
        compute source and destination node temporal embeddings
        :param src_node_ids: ndarray, shape (batch_size, )
        :param dst_node_ids: ndarray, shape (batch_size, )
        :param node_interact_times: ndarray, shape (batch_size, )
        :param num_neighbors: int, number of neighbors to sample for each node
        :param time_gap: int, time gap for neighbors to compute node features
        :return:
        """

        node_ids_all = np.concatenate([src_node_ids, dst_node_ids],axis=0)
        
        node_interact_times_all = np.concatenate([node_interact_times, node_interact_times],axis=0)
        
        node_embeddings, align_loss_term = self.compute_node_temporal_embeddings(node_ids=node_ids_all, node_interact_times=node_interact_times_all, edge_ids = edge_ids,
                                                                    num_neighbors=num_neighbors, time_gap=time_gap)
        
        src_node_embeddings, dst_node_embeddings = node_embeddings[:len(src_node_ids),:], node_embeddings[len(src_node_ids):, :]
        
        
        
        return src_node_embeddings, dst_node_embeddings, align_loss_term



    
    def compute_node_temporal_embeddings(self, node_ids: np.ndarray, node_interact_times: np.ndarray, edge_ids: np.ndarray,
                                         num_neighbors: int = 20, time_gap: int = 2000):
        """
        given node ids node_ids, and the corresponding time node_interact_times, return the temporal embeddings of nodes in node_ids
        :param node_ids: ndarray, shape (batch_size, ), node ids
        :param node_interact_times: ndarray, shape (batch_size, ), node interaction times
        :param num_neighbors: int, number of neighbors to sample for each node
        :param time_gap: int, time gap for neighbors to compute node features
        :return:
        """
        # link encoder
        # get temporal neighbors, including neighbor ids, edge ids and time information
        # neighbor_node_ids, ndarray, shape (batch_size, num_neighbors)
        # neighbor_edge_ids, ndarray, shape (batch_size, num_neighbors)
        # neighbor_times, ndarray, shape (batch_size, num_neighbors)
        neighbor_node_ids, neighbor_edge_ids, neighbor_times = \
            self.neighbor_sampler.get_historical_neighbors(node_ids=node_ids,
                                                           node_interact_times=node_interact_times,
                                                           num_neighbors=num_neighbors)

        # Tensor, shape (batch_size, num_neighbors, edge_feat_dim)
        nodes_edge_raw_features = self.edge_raw_features[torch.from_numpy(neighbor_edge_ids)]
        # Tensor, shape (batch_size, num_neighbors, time_feat_dim)
        nodes_neighbor_time_features = self.time_encoder(timestamps=torch.from_numpy(node_interact_times[:, np.newaxis] - neighbor_times).float().to(self.device))
#         nodes_neighbor_time_features =  self.periodic_time(torch.from_numpy(node_interact_times[:, np.newaxis] - neighbor_times).float().to(self.device), 128)
        # ndarray, set the time features to all zeros for the padded timestamp
        nodes_neighbor_time_features[torch.from_numpy(neighbor_node_ids == 0)] = 0.0

#         # Tensor, shape (batch_size, num_neighbors, edge_feat_dim + time_feat_dim)
# #         combined_features = nodes_edge_raw_features
        combined_features = torch.cat([nodes_edge_raw_features, nodes_neighbor_time_features], dim=-1)
#         # Tensor, shape (batch_size, num_neighbors, num_channels)
        combined_features = self.projection_layer(combined_features)
    
        combined_features = self.Structure(combined_features)

        combined_features = torch.mean(combined_features, dim=1)
    
        node_text_input = self.node_raw_features[torch.from_numpy(node_ids)]  #nodes_time_gap_neighbor_node_agg_features + 
        node_text_embed = self.text_global(node_text_input)
        
        
        node_time_enc = self.time_encoder(timestamps=torch.from_numpy(node_interact_times[:, np.newaxis]).float().to(self.device))
        time_den_enc = self.time_encoder(timestamps=torch.from_numpy(neighbor_times).float().to(self.device))        
        root_node_time = torch.cat([node_time_enc, time_den_enc], dim=1)
        node_time_embed = self.time_global(root_node_time.mean(dim =1))
        
        node_embed = self.output_layer(self.alpha * node_text_embed + (1 - self.alpha) * node_time_embed)
        node_embeddings = combined_features + self.beta*node_embed
        
        
        if edge_ids is not None:
            sim_score_a = 1 - F.cosine_similarity(combined_features, node_embed, dim=1).mean()
            align_loss_term =  sim_score_a + self.mmd_rff(node_text_embed, node_time_embed, self.rff)
        else:
            align_loss_term = 0.0

        return node_embeddings, align_loss_term


    
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

    
class RFFMap(torch.nn.Module):
    """
    Random Fourier Feature mapping for RBF kernel.
    out_dim = D̂  (must be even)
    """
    def __init__(self, in_dim, out_dim=512, sigma=1.0, trainable=False):
        super().__init__()
        assert out_dim % 2 == 0, "out_dim must be even"
        self.out_dim = out_dim
        w = torch.randn(out_dim // 2, in_dim) / sigma      
        b = 2 * math.pi * torch.rand(out_dim // 2)         
        self.register_buffer("omega", w)
        self.register_buffer("bias",  b)
        if trainable:                                      
            self.omega = torch.nn.Parameter(self.omega)
            self.bias  = torch.nn.Parameter(self.bias)

    def forward(self, x):          # x: [B, d]
        proj = x @ self.omega.T + self.bias          
        z = torch.cat([proj.cos(), proj.sin()], dim=-1)
        return z * math.sqrt(2.0 / self.out_dim)     
        

class FeedForwardNet(nn.Module):

    def __init__(self, input_dim: int, dim_expansion_factor: float, dropout: float = 0.0):
        """
        two-layered MLP with GELU activation function.
        :param input_dim: int, dimension of input
        :param dim_expansion_factor: float, dimension expansion factor
        :param dropout: float, dropout rate
        """
        super(FeedForwardNet, self).__init__()

        self.input_dim = input_dim
        self.dim_expansion_factor = dim_expansion_factor
        self.dropout = dropout

        self.ffn = nn.Sequential(nn.Linear(in_features=input_dim, out_features=int(dim_expansion_factor * input_dim)),
                                 nn.GELU(),
                                 nn.Dropout(dropout),
                                 nn.Linear(in_features=int(dim_expansion_factor * input_dim), out_features=input_dim),
                                 nn.Dropout(dropout))

    def forward(self, x: torch.Tensor):
        """
        feed forward net forward process
        :param x: Tensor, shape (*, input_dim)
        :return:
        """
        return self.ffn(x)