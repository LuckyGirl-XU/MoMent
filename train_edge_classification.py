import logging
import time
import sys
import os
from tqdm import tqdm
import numpy as np
import warnings
import shutil
import json
import torch
import torch.nn as nn

from models.TGAT import TGAT
from models.GraphMixer import GraphMixer
from models.DyGFormer import DyGFormer
from models.CNEN import CNEN
from models.modules import MergeLayer, MLPClassifier, MLPClassifier_edge
from utils.utils import set_random_seed, convert_to_gpu, get_parameter_sizes, create_optimizer
from utils.utils import get_neighbor_sampler
from evaluate_models_utils import evaluate_model_edge_classification
from utils.metrics import get_edge_classification_metrics
from utils.DataLoader import get_idx_data_loader, get_edge_classification_data
from utils.EarlyStopping import EarlyStopping
from utils.load_configs import get_edge_classification_args


if __name__ == "__main__":

    warnings.filterwarnings('ignore')

    # get arguments
    args = get_edge_classification_args(is_evaluation=False)

    # get data for training, validation and testing
    node_raw_features, edge_raw_features, full_data, train_data, val_data, test_data, new_node_val_data, new_node_test_data, cat_num = \
        get_edge_classification_data(dataset_name=args.dataset_name, val_ratio=args.val_ratio, test_ratio=args.test_ratio, args = args)

    # initialize training neighbor sampler to retrieve temporal graph
    train_neighbor_sampler = get_neighbor_sampler(data=train_data, sample_neighbor_strategy=args.sample_neighbor_strategy,
                                                  time_scaling_factor=args.time_scaling_factor, seed=0)

    # initialize validation and test neighbor sampler to retrieve temporal graph
    full_neighbor_sampler = get_neighbor_sampler(data=full_data, sample_neighbor_strategy=args.sample_neighbor_strategy,
                                                 time_scaling_factor=args.time_scaling_factor, seed=1)
    # get data loaders
    train_idx_data_loader = get_idx_data_loader(indices_list=list(range(len(train_data.src_node_ids))), batch_size=args.batch_size, shuffle=False)
    val_idx_data_loader = get_idx_data_loader(indices_list=list(range(len(val_data.src_node_ids))), batch_size=args.batch_size, shuffle=False)
    new_node_val_idx_data_loader = get_idx_data_loader(indices_list=list(range(len(new_node_val_data.src_node_ids))), batch_size=args.batch_size, shuffle=False)
    test_idx_data_loader = get_idx_data_loader(indices_list=list(range(len(test_data.src_node_ids))), batch_size=args.batch_size, shuffle=False)
    new_node_test_idx_data_loader = get_idx_data_loader(indices_list=list(range(len(new_node_test_data.src_node_ids))), batch_size=args.batch_size, shuffle=False)

    val_metric_all_runs, new_node_val_metric_all_runs, test_metric_all_runs, new_node_test_metric_all_runs = [], [], [], []

    for run in range(args.num_runs):

        set_random_seed(seed=(run))

        args.seed = run
        args.save_model_name = f'edge_classification_{args.model_name}_seed{args.seed}{args.use_feature}'

        # set up logger
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)
        os.makedirs(f"./logs/{args.model_name}/{args.dataset_name}/{args.save_model_name}/", exist_ok=True)
        # create file handler that logs debug and higher level messages
        fh = logging.FileHandler(f"./logs/{args.model_name}/{args.dataset_name}/{args.save_model_name}/{str(time.time())}.log")
        fh.setLevel(logging.DEBUG)
        # create console handler with a higher log level
        ch = logging.StreamHandler()
        ch.setLevel(logging.WARNING)
        # create formatter and add it to the handlers
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        # add the handlers to logger
        logger.addHandler(fh)
        logger.addHandler(ch)

        run_start_time = time.time()
        logger.info(f"********** Run {run + 1} starts. **********")

        logger.info(f'configuration is {args}')
        logger.info(f'Loss weight {args.loss_weight}')
        logger.info(f'node feature size {node_raw_features.shape}')
        logger.info(f'edge feature size {edge_raw_features.shape}')
        logger.info(f'node feature example {node_raw_features[1][:5]}')
        logger.info(f'edge feature example {edge_raw_features[1][:5]}')

        # create model
        if args.model_name == 'TGAT':
            dynamic_backbone = TGAT(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features, neighbor_sampler=train_neighbor_sampler,
                                    time_feat_dim=args.time_feat_dim, num_layers=args.num_layers, num_heads=args.num_heads, dropout=args.dropout, device=args.device)
        elif args.model_name == 'GraphMixer':
            dynamic_backbone = GraphMixer(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features, neighbor_sampler=train_neighbor_sampler,
                                          time_feat_dim=args.time_feat_dim, num_tokens=args.num_neighbors, num_layers=args.num_layers, dropout=args.dropout, device=args.device)
        elif args.model_name == 'DyGFormer':
            dynamic_backbone = DyGFormer(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features, neighbor_sampler=train_neighbor_sampler,
                                         time_feat_dim=args.time_feat_dim, channel_embedding_dim=args.channel_embedding_dim, patch_size=args.patch_size,
                                         num_layers=args.num_layers, num_heads=args.num_heads, dropout=args.dropout,
                                         max_input_sequence_length=args.max_input_sequence_length, device=args.device)           
        elif args.model_name == 'CNEN':
            dynamic_backbone = CNEN(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features,
                                         neighbor_sampler=train_neighbor_sampler,
                                         time_feat_dim=args.time_feat_dim,
                                         channel_embedding_dim=args.channel_embedding_dim, update_neighbor=args.update_neighbor,
                                         num_layers=args.num_layers, num_heads=args.num_heads, dropout=args.dropout,
                                             max_input_sequence_length=args.max_input_sequence_length, 
                                    memory_dim = args.memory_dim, 
                                    output_dim= node_raw_features.shape[1],
                                    device=args.device)
        else:
            raise ValueError(f"Wrong value for model_name {args.model_name}!")
        edge_classifier = MLPClassifier_edge(input_dim=node_raw_features.shape[1], dropout=args.dropout, cat_num=cat_num)
        model = nn.Sequential(dynamic_backbone, edge_classifier)
#         logger.info(f'model -> {model}')
#         logger.info(f'model name: {args.model_name}, #parameters: {get_parameter_sizes(model) * 4} B, '
#                     f'{get_parameter_sizes(model) * 4 / 1024} KB, {get_parameter_sizes(model) * 4 / 1024 / 1024} MB.')

        optimizer = create_optimizer(model=model, optimizer_name=args.optimizer, learning_rate=args.learning_rate, weight_decay=args.weight_decay)

        model = convert_to_gpu(model, device=args.device)

        save_model_folder = f"./saved_models/{args.model_name}/{args.dataset_name}/{args.save_model_name}/"
        shutil.rmtree(save_model_folder, ignore_errors=True)
        os.makedirs(save_model_folder, exist_ok=True)

        early_stopping = EarlyStopping(patience=args.patience, save_model_folder=save_model_folder,
                                       save_model_name=args.save_model_name, logger=logger, model_name=args.model_name)

        loss_func = nn.CrossEntropyLoss()

        for epoch in range(args.num_epochs):
            model.train()
            if args.model_name in ['TGAT', 'GraphMixer', 'DyGFormer']:
                # training, only use training graph
                model[0].set_neighbor_sampler(train_neighbor_sampler)

            # store train losses and metrics
            train_total_loss, train_y_trues, train_y_predicts = 0.0, [], []
            train_idx_data_loader_tqdm = tqdm(train_idx_data_loader, ncols=120)
            for batch_idx, train_data_indices in enumerate(train_idx_data_loader_tqdm):
                train_data_indices = train_data_indices.numpy()
                batch_src_node_ids, batch_dst_node_ids, batch_node_interact_times, batch_edge_ids, batch_labels = \
                    train_data.src_node_ids[train_data_indices], train_data.dst_node_ids[train_data_indices], \
                    train_data.node_interact_times[train_data_indices], train_data.edge_ids[train_data_indices], train_data.labels[train_data_indices]

                # we need to compute for positive and negative edges respectively, because the new sampling strategy (for evaluation) allows the negative source nodes to be
                # different from the source nodes, this is different from previous works that just replace destination nodes with negative destination nodes
                if args.model_name in ['TGAT']:
                    # get temporal embedding of source and destination nodes
                    # two Tensors, with shape (batch_size, node_feat_dim)
                    batch_src_node_embeddings, batch_dst_node_embeddings, batch_align_loss_term = \
                        model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                          dst_node_ids=batch_dst_node_ids,
                                                                          node_interact_times=batch_node_interact_times,
                                                                          num_neighbors=args.num_neighbors)

                elif args.model_name in ['GraphMixer']:
                    # get temporal embedding of source and destination nodes
                    # two Tensors, with shape (batch_size, node_feat_dim)
                    batch_src_node_embeddings, batch_dst_node_embeddings, batch_align_loss_term = \
                        model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                          dst_node_ids=batch_dst_node_ids,
                                                                          node_interact_times=batch_node_interact_times,
                                                                          num_neighbors=args.num_neighbors,
                                                                          time_gap=args.time_gap)

                elif args.model_name in ['DyGFormer']:
                    # get temporal embedding of source and destination nodes
                    # two Tensors, with shape (batch_size, node_feat_dim)
                    batch_src_node_embeddings, batch_dst_node_embeddings, batch_align_loss_term = \
                        model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                          dst_node_ids=batch_dst_node_ids,
                                                                          node_interact_times=batch_node_interact_times)
                elif args.model_name in ['CNEN']:
                    # get temporal embedding of source and destination nodes
                    # two Tensors, with shape (batch_size, node_feat_dim)
                    batch_src_node_embeddings, batch_dst_node_embeddings, batch_align_loss_term = \
                        model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                              dst_node_ids=batch_dst_node_ids,
                                                                              node_interact_times=batch_node_interact_times, 
                                                                              positive = True)

                else:
                    raise ValueError(f"Wrong value for model_name {args.model_name}!")

                # get predicted probabilities, shape (batch_size, )
                predicts = model[1](x_1=batch_src_node_embeddings, x_2 = batch_dst_node_embeddings, rel_embs = model[0].edge_raw_features)
                pred_labels = torch.max(predicts, dim=1)[1]
                labels = torch.from_numpy(batch_labels).long().to(predicts.device)
                
#                 align_loss = batch_align_loss_term
#                 loss_weight = args.loss_weight

                loss = loss_func(predicts, labels) + args.loss_weight * batch_align_loss_term

                train_y_trues.append(labels)
                train_y_predicts.append(pred_labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                

                train_idx_data_loader_tqdm.set_description(f'Epoch: {epoch + 1}, train for the {batch_idx + 1}-th batch, train loss: {loss.item()}')

            train_y_trues = torch.cat(train_y_trues, dim=0)
            train_y_predicts = torch.cat(train_y_predicts, dim=0)

            if (epoch + 1) % args.test_interval_epochs == 0:
                train_metrics = get_edge_classification_metrics(predicts=train_y_predicts, labels=train_y_trues)

                val_total_loss, val_metrics = evaluate_model_edge_classification(model_name=args.model_name,
                                                                                model=model,
                                                                                neighbor_sampler=full_neighbor_sampler,
                                                                                evaluate_idx_data_loader=val_idx_data_loader,
                                                                                evaluate_data=val_data,
                                                                                loss_func=loss_func,
                                                                                num_neighbors=args.num_neighbors,
                                                                                time_gap=args.time_gap,
                                                                                loss_weight= args.loss_weight)

                logger.info(f'Epoch: {epoch + 1}, learning rate: {optimizer.param_groups[0]["lr"]}, train loss: {train_total_loss:.4f}')
                for metric_name in train_metrics.keys():
                    logger.info(f'train {metric_name}, {train_metrics[metric_name]:.4f}')
                logger.info(f'validate loss: {val_total_loss:.4f}')
                for metric_name in val_metrics.keys():
                    logger.info(f'validate {metric_name}, {val_metrics[metric_name]:.4f}')

                # select the best model based on all the validate metrics
                val_metric_indicator = []
                for metric_name in val_metrics.keys():
                    val_metric_indicator.append((metric_name, val_metrics[metric_name], True))
                early_stop = early_stopping.step(val_metric_indicator, model)

                if early_stop:
                    break

            # perform testing once after test_interval_epochs
            if (epoch + 1) % args.test_interval_epochs == 0:
                

                test_total_loss, test_metrics = evaluate_model_edge_classification(model_name=args.model_name,
                                                                                   model=model,
                                                                                   neighbor_sampler=full_neighbor_sampler,
                                                                                   evaluate_idx_data_loader=test_idx_data_loader,
                                                                                   evaluate_data=test_data,
                                                                                   loss_func=loss_func,
                                                                                   num_neighbors=args.num_neighbors,
                                                                                   time_gap=args.time_gap)

                

                logger.info(f'test loss: {test_total_loss:.4f}')
                for metric_name in test_metrics.keys():
                    logger.info(f'test {metric_name}, {test_metrics[metric_name]:.4f}')

        # load the best model
        early_stopping.load_checkpoint(model)

        # evaluate the best model
        logger.info(f'get final performance on dataset {args.dataset_name}...')

        # the saved best model of memory-based models cannot perform validation since the stored memory has been updated by validation data
        if args.model_name not in ['JODIE', 'DyRep', 'TGN']:
            val_total_loss, val_metrics = evaluate_model_edge_classification(model_name=args.model_name,
                                                                             model=model,
                                                                             neighbor_sampler=full_neighbor_sampler,
                                                                             evaluate_idx_data_loader=val_idx_data_loader,
                                                                             evaluate_data=val_data,
                                                                             loss_func=loss_func,
                                                                             num_neighbors=args.num_neighbors,
                                                                             time_gap=args.time_gap,
                                                                             loss_weight= args.loss_weight)

        test_total_loss, test_metrics = evaluate_model_edge_classification(model_name=args.model_name,
                                                                           model=model,
                                                                           neighbor_sampler=full_neighbor_sampler,
                                                                           evaluate_idx_data_loader=test_idx_data_loader,
                                                                           evaluate_data=test_data,
                                                                           loss_func=loss_func,
                                                                           num_neighbors=args.num_neighbors,
                                                                           time_gap=args.time_gap,
                                                                           loss_weight= args.loss_weight)

        # store the evaluation metrics at the current run
        val_metric_dict, test_metric_dict = {}, {}

        if args.model_name not in ['JODIE', 'DyRep', 'TGN']:
            logger.info(f'validate loss: {val_total_loss:.4f}')
            for metric_name in val_metrics.keys():
                val_metric = val_metrics[metric_name]
                logger.info(f'validate {metric_name}, {val_metric:.4f}')
                val_metric_dict[metric_name] = val_metric

        logger.info(f'test loss: {test_total_loss:.4f}')
        for metric_name in test_metrics.keys():
            test_metric = test_metrics[metric_name]
            logger.info(f'test {metric_name}, {test_metric:.4f}')
            test_metric_dict[metric_name] = test_metric

        single_run_time = time.time() - run_start_time
        logger.info(f'Run {run + 1} cost {single_run_time:.2f} seconds.')

        if args.model_name not in ['JODIE', 'DyRep', 'TGN']:
            val_metric_all_runs.append(val_metric_dict)
        test_metric_all_runs.append(test_metric_dict)

        # avoid the overlap of logs
        if run < args.num_runs - 1:
            logger.removeHandler(fh)
            logger.removeHandler(ch)

        # save model result
        if args.model_name not in ['JODIE', 'DyRep', 'TGN']:
            result_json = {
                "validate metrics": {metric_name: f'{val_metric_dict[metric_name]:.4f}' for metric_name in val_metric_dict},
                "test metrics": {metric_name: f'{test_metric_dict[metric_name]:.4f}' for metric_name in test_metric_dict}
            }
        else:
            result_json = {
                "test metrics": {metric_name: f'{test_metric_dict[metric_name]:.4f}' for metric_name in test_metric_dict}
            }
        result_json = json.dumps(result_json, indent=4)

        save_result_folder = f"./saved_results/{args.model_name}/{args.dataset_name}"
        os.makedirs(save_result_folder, exist_ok=True)
        save_result_path = os.path.join(save_result_folder, f"{args.save_model_name}.json")

        with open(save_result_path, 'w') as file:
            file.write(result_json)

    # store the average metrics at the log of the last run
    logger.info(f'metrics over {args.num_runs} runs:')

    if args.model_name not in ['JODIE', 'DyRep', 'TGN']:
        for metric_name in val_metric_all_runs[0].keys():
            logger.info(f'validate {metric_name}, {[val_metric_single_run[metric_name] for val_metric_single_run in val_metric_all_runs]}')
            logger.info(f'average validate {metric_name}, {np.mean([val_metric_single_run[metric_name] for val_metric_single_run in val_metric_all_runs]):.4f} '
                        f'± {np.std([val_metric_single_run[metric_name] for val_metric_single_run in val_metric_all_runs], ddof=1):.4f}')

    for metric_name in test_metric_all_runs[0].keys():
        logger.info(f'test {metric_name}, {[test_metric_single_run[metric_name] for test_metric_single_run in test_metric_all_runs]}')
        logger.info(f'average test {metric_name}, {np.mean([test_metric_single_run[metric_name] for test_metric_single_run in test_metric_all_runs]):.4f} '
                    f'± {np.std([test_metric_single_run[metric_name] for test_metric_single_run in test_metric_all_runs], ddof=1):.4f}')

    sys.exit()
