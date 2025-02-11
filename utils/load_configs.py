import argparse
import sys
import torch


def get_link_prediction_args(is_evaluation: bool = False):
    """
    get the args for the link prediction task
    :param is_evaluation: boolean, whether in the evaluation process
    :return:
    """
    # arguments
    parser = argparse.ArgumentParser('Interface for the link prediction task')
    parser.add_argument('--dataset_name', type=str, help='dataset to be used', default='Enron',
                        choices=['Amazon_movies', 'Enron', 'Googlemap_CT', 'ICEWS1819', 'Stack_elec', 'Stack_ubuntu', 'Yelp'])
    parser.add_argument('--batch_size', type=int, default=256, help='batch size')
    parser.add_argument('--model_name', type=str, default='GraphMixer', help='name of the model, note that EdgeBank is only applicable for evaluation',
                        choices=['TGAT','GraphMixer', 'DyGFormer', 'CNEN']) # TGAT,CAWN (ICEWS1819) takes long time; TGN(ICEW1819) OOM
    parser.add_argument('--gpu', type=int, default=0, help='number of gpu to use')
    parser.add_argument('--num_neighbors', type=int, default=20, help='number of neighbors to sample for each node')
    parser.add_argument('--sample_neighbor_strategy', type=str, default='recent', choices=['uniform', 'recent', 'time_interval_aware'], help='how to sample historical neighbors')
    parser.add_argument('--time_scaling_factor', default=1e-6, type=float, help='the hyperparameter that controls the sampling preference with time interval, '
                        'a large time_scaling_factor tends to sample more on recent links, 0.0 corresponds to uniform sampling, '
                        'it works when sample_neighbor_strategy == time_interval_aware')
    parser.add_argument('--num_walk_heads', type=int, default=8, help='number of heads used for the attention in walk encoder')
    parser.add_argument('--num_heads', type=int, default=2, help='number of heads used in attention layer')
    parser.add_argument('--num_layers', type=int, default=2, help='number of model layers')
    parser.add_argument('--walk_length', type=int, default=1, help='length of each random walk')
    parser.add_argument('--time_gap', type=int, default=2000, help='time gap for neighbors to compute node features')
    parser.add_argument('--time_feat_dim', type=int, default=100, help='dimension of the time embedding')
    parser.add_argument('--position_feat_dim', type=int, default=172, help='dimension of the position embedding')
    parser.add_argument('--edge_bank_memory_mode', type=str, default='unlimited_memory', help='how memory of EdgeBank works',
                        choices=['unlimited_memory', 'time_window_memory', 'repeat_threshold_memory'])
    parser.add_argument('--time_window_mode', type=str, default='fixed_proportion', help='how to select the time window size for time window memory',
                        choices=['fixed_proportion', 'repeat_interval'])
    parser.add_argument('--patch_size', type=int, default=1, help='patch size')
    parser.add_argument('--channel_embedding_dim', type=int, default=50, help='dimension of each channel embedding')
    parser.add_argument('--max_input_sequence_length', type=int, default=48, help='maximal length of the input sequence of each node')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout rate')
    parser.add_argument('--num_epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--optimizer', type=str, default='Adam', choices=['SGD', 'Adam', 'RMSprop'], help='name of optimizer')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='weight decay')
    parser.add_argument('--patience', type=int, default=5, help='patience for early stopping')
    parser.add_argument('--val_ratio', type=float, default=0.15, help='ratio of validation set')
    parser.add_argument('--test_ratio', type=float, default=0.15, help='ratio of test set')
    parser.add_argument('--num_runs', type=int, default=5, help='number of runs')
    parser.add_argument('--test_interval_epochs', type=int, default=10, help='how many epochs to perform testing once')
    parser.add_argument('--negative_sample_strategy', type=str, default='random', choices=['random', 'historical', 'inductive'],
                        help='strategy for the negative edge sampling')
    parser.add_argument('--use_feature', type=str, default='', help='whether to use text embeddings as feature') # or Bert
    parser.add_argument('--load_best_configs', action='store_true', default=True, help='whether to load the best configurations')
    parser.add_argument('--memory_dim', type=int, default=64, help='dimension for neighbor hashtable')
    parser.add_argument('--update_neighbor', action='store_true', default=True, help='whether to update neighbor hashtable')
    parser.add_argument('--loss_weight', type=float, default=0.1, help='Loss weight')

    try:
        args = parser.parse_args()
        args.device = f'cuda:{args.gpu}' if torch.cuda.is_available() and args.gpu >= 0 else 'cpu'
    except:
        parser.print_help()
        sys.exit

    if args.load_best_configs:
        load_link_prediction_best_configs(args=args)
    
    if args.use_feature == 'Bert':
        args.time_dim = 768
        args.position_feat_dim = 768


    return args

def load_link_prediction_best_configs(args: argparse.Namespace):
    """
    load the best configurations for the link prediction task
    :param args: argparse.Namespace
    :return:
    """
    # model specific settings
    if args.model_name == 'TGAT':
        args.num_neighbors = 20
        args.num_layers = 2
        args.sample_neighbor_strategy = 'recent' # 'recent'
    elif args.model_name == 'GraphMixer':
        args.num_layers = 2
        args.patience = 5
        args.num_neighbors = 20
        args.dropout = 0.5
        args.sample_neighbor_strategy = 'recent'
    elif args.model_name == 'DyGFormer':
        args.num_layers = 2
        args.max_input_sequence_length = 48
        args.patch_size = 1
        args.dropout = 0.1
    elif args.model_name == 'CNEN':
        args.num_layers = 2
        args.dropout = 0.1
        args.max_input_sequence_length = 4
        args.memory_dim = 32
    else:
        raise ValueError(f"Wrong value for model_name {args.model_name}!")


def get_edge_classification_args(is_evaluation: bool = False):
    """
    get the args for the node classification task
    :return:
    """
    # arguments
    parser = argparse.ArgumentParser('Interface for the node classification task')
    parser.add_argument('--dataset_name', type=str, help='dataset to be used', default='Enron',
                        choices=['Amazon_movies', 'Enron', 'Googlemap_CT', 'ICEWS1819', 'Stack_elec', 'Stack_ubuntu', 'Yelp'])
    parser.add_argument('--batch_size', type=int, default=256, help='batch size') # GDELT, ICEWS1819, 
    parser.add_argument('--model_name', type=str, default='GraphMixer', help='name of the model',
                        choices=[ 'TGAT','GraphMixer', 'DyGFormer', 'CNEN'])
    parser.add_argument('--gpu', type=int, default=0, help='number of gpu to use')
    parser.add_argument('--num_neighbors', type=int, default=20, help='number of neighbors to sample for each node')
    parser.add_argument('--sample_neighbor_strategy', type=str, default='recent', choices=['uniform', 'recent', 'time_interval_aware'], help='how to sample historical neighbors')
    parser.add_argument('--time_scaling_factor', default=1e-6, type=float, help='the hyperparameter that controls the sampling preference with time interval, '
                        'a large time_scaling_factor tends to sample more on recent links, 0.0 corresponds to uniform sampling, '
                        'it works when sample_neighbor_strategy == time_interval_aware')
    parser.add_argument('--num_walk_heads', type=int, default=8, help='number of heads used for the attention in walk encoder')
    parser.add_argument('--num_heads', type=int, default=2, help='number of heads used in attention layer')
    parser.add_argument('--num_layers', type=int, default=2, help='number of model layers')
    parser.add_argument('--walk_length', type=int, default=1, help='length of each random walk')
    parser.add_argument('--time_gap', type=int, default=2000, help='time gap for neighbors to compute node features')
    parser.add_argument('--time_feat_dim', type=int, default=100, help='dimension of the time embedding')
    parser.add_argument('--position_feat_dim', type=int, default=172, help='dimension of the position embedding')
    parser.add_argument('--patch_size', type=int, default=1, help='patch size')
    parser.add_argument('--channel_embedding_dim', type=int, default=50, help='dimension of each channel embedding')
    parser.add_argument('--max_input_sequence_length', type=int, default=48, help='maximal length of the input sequence of each node')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout rate')
    parser.add_argument('--num_epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--optimizer', type=str, default='Adam', choices=['SGD', 'Adam', 'RMSprop'], help='name of optimizer')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='weight decay')
    parser.add_argument('--patience', type=int, default=5, help='patience for early stopping')
    parser.add_argument('--val_ratio', type=float, default=0.15, help='ratio of validation set')
    parser.add_argument('--test_ratio', type=float, default=0.15, help='ratio of test set')
    parser.add_argument('--num_runs', type=int, default=5, help='number of runs')
    parser.add_argument('--test_interval_epochs', type=int, default=10, help='how many epochs to perform testing once')
    parser.add_argument('--use_feature', type=str, default='', help='whether to use text embeddings as feature')
    parser.add_argument('--load_best_configs', action='store_true', default=False, help='whether to load the best configurations')
    parser.add_argument('--update_neighbor', action='store_true', default=True, help='whether to update neighbor hashtable')
    parser.add_argument('--memory_dim', type=int, default=64, help='dimension for neighbor hashtable')
#     parser.add_argument('--update_neighbor', action='store_true', default=True, help='whether to update neighbor hashtable')
    parser.add_argument('--loss_weight', type=float, default=0.1, help='Loss weight')
   

    try:
        args = parser.parse_args()
        args.device = f'cuda:{args.gpu}' if torch.cuda.is_available() and args.gpu >= 0 else 'cpu'
    except:
        parser.print_help()
        sys.exit()

    if args.load_best_configs:
        load_edge_classification_best_configs(args=args)
    
    if args.use_feature == True:
        args.time_dim = 768
        args.position_feat_dim = 768
     
    return args


def load_edge_classification_best_configs(args: argparse.Namespace):
    """
    load the best configurations for the node classification task
    :param args: argparse.Namespace
    :return:
    """
    # model specific settings
    if args.model_name == 'TGAT':
        args.num_neighbors = 20
        args.num_layers = 2
        args.dropout = 0.1
    elif args.model_name == 'GraphMixer':
        args.num_layers = 2
        args.dropout = 0.5
        args.sample_neighbor_strategy = 'recent'
    elif args.model_name == 'DyGFormer':
        args.num_layers = 2
        args.max_input_sequence_length = 48
        args.patch_size = 1
    elif args.model_name == 'CNEN':
        args.num_layers = 2
        args.dropout = 0.1
        args.max_input_sequence_length = 4
        args.memory_dim = 32
    else:
        raise ValueError(f"Wrong value for model_name {args.model_name}!")

