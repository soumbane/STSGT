import os
import time
import argparse
import configparser
import numpy as np
import torch
import torch.nn as nn
import tqdm

from engine import trainer
from utils import *
import ast

DATASET = 'COVID_JHU'  # COVID Time-Series Dataset from John Hopkins University
# DATASET = 'COVID_NYT'  # COVID Time-Series Dataset from New York Times

config_file = './config/{}.conf'.format(DATASET)
config = configparser.ConfigParser()
config.read(config_file)


parser = argparse.ArgumentParser(description='arguments')
parser.add_argument('--no_cuda', action="store_true", help="NO GPU")
parser.add_argument('--data', type=str, default=config['data']['data'], help='data path')
parser.add_argument('--sensors_distance', type=str, default=config['data']['sensors_distance'],
                    help='Node Distance File')
parser.add_argument('--batch_size', type=int, default=config['data']['batch_size'],
                    help="Training Batch Size")
parser.add_argument('--valid_batch_size', type=int, default=config['data']['valid_batch_size'],
                    help="Validation Batch Size")
parser.add_argument('--test_batch_size', type=int, default=config['test']['test_batch_size'],
                    help="Test Batch Size")
parser.add_argument('--fill_zeros', type=eval, default=config['data']['fill_zeros'],
                    help="whether to fill zeros in data with average")

parser.add_argument('--num_of_vertices', type=int, default=config['model']['num_of_vertices'],
                    help='Number of sensors')
parser.add_argument('--in_dim', type=int, default=config['model']['in_dim'], help='input dimension')
parser.add_argument('--hidden_dims', type=list, default=ast.literal_eval(config['model']['hidden_dims']),
                    help='Convolution operation dimension of each STSGCL layer in the middle')
parser.add_argument('--first_layer_embedding_size', type=int,
                    default=config['model']['first_layer_embedding_size'],
                    help='The dimension of the first input layer')
parser.add_argument('--out_layer_dim', type=int, default=config['model']['out_layer_dim'],
                    help='Output module middle layer dimension')
parser.add_argument('--d_model', type=int, default=config['model']['d_model'],
                    help='Embedding dimension for the ST Synchronous Transformer')
parser.add_argument('--n_heads', type=int, default=config['model']['n_heads'],
                    help='Number of heads for the Multi-Head Attention')
parser.add_argument('--dropout', type=float, default=config['model']['dropout'],
                    help='dropout for the ST Synchronous Transformer')
parser.add_argument('--forward_expansion', type=int, default=config['model']['forward_expansion'],
                    help='Hidden Layer Dimension for the ST Synchronous Transformer')
parser.add_argument("--history", type=int, default=config['model']['history'],
                    help="The discrete time series of each sample input")
parser.add_argument("--horizon", type=int, default=config['model']['horizon'],
                    help="The discrete time series of each sample output (forecast)")
parser.add_argument("--strides", type=int, default=config['model']['strides'],
                    help="The step size of the sliding window, "
                         "the local spatio-temporal graph is constructed using several time steps, "
                         "the default is 3")
parser.add_argument("--temporal_emb", type=eval, default=config['model']['temporal_emb'],
                    help="Whether to use temporal embedding vector")
parser.add_argument("--spatial_emb", type=eval, default=config['model']['spatial_emb'],
                    help="Whether to use spatial embedding vector")

parser.add_argument("--use_transformer", type=eval, default=config['model']['use_transformer'],
                    help="Whether to use the Spatio-Temporal Transformer or not")
parser.add_argument("--use_informer", type=eval, default=config['model']['use_informer'],
                    help="Whether to use the Spatio-Temporal Informer or not")
parser.add_argument("--factor", type=int, default=config['model']['factor'],
                    help="The amount of self-attentions needed")
parser.add_argument("--attention_dropout", type=float, default=config['model']['attention_dropout'],
                    help="The amount of dropout for sparse self-attentions")
parser.add_argument("--output_attention", type=eval, default=config['model']['output_attention'],
                    help="Whether to output the self-attentions or not")

parser.add_argument("--use_mask", type=eval, default=config['model']['use_mask'],
                    help="Whether to use the mask matrix to optimize adj")
parser.add_argument("--activation", type=str, default=config['model']['activation'],
                    help="Activation Function {ReLU, GLU}")

parser.add_argument('--seed', type=int, default=config['train']['seed'], help='Seed Settings')
parser.add_argument("--learning_rate", type=float, default=config['train']['learning_rate'],
                    help="Initial Learning Rate")
parser.add_argument("--weight_decay", type=float, default=config['train']['weight_decay'],
                    help="Weight Decay Rate")
parser.add_argument("--lr_decay", type=eval, default=config['train']['lr_decay'],
                    help="Whether to enable the initial learning rate decay strategy")
parser.add_argument("--lr_decay_rate", type=float, default=config['train']['lr_decay_rate'],
                    help="Learning rate decay rate")
parser.add_argument('--epochs', type=int, default=config['train']['epochs'],
                    help="Number of training epochs")
parser.add_argument('--print_every', type=int, default=config['train']['print_every'],
                    help='Print losses and metrics after print_every iterations')
parser.add_argument('--save', type=str, default=config['train']['save'], help='Save Path')
parser.add_argument('--save_loss', type=str, default=config['train']['save_loss'], help='Save Loss Path')
parser.add_argument('--expid', type=int, default=config['train']['expid'], help='Experiment ID')
parser.add_argument('--max_grad_norm', type=float, default=config['train']['max_grad_norm'],
                    help="Gradient Threshold")

parser.add_argument('--patience', type=int, default=config['train']['patience'],
                    help='Patience during training')
parser.add_argument('--log_file', default=config['train']['log_file'], help='log file')

args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

log = open(args.log_file, 'w')
log_string(log, str(args))


def main():
    # load data
    adj = get_adjacency_matrix(distance_df_filename=args.sensors_distance)

    # local_adj is only required for synchronous adjacency matrix - for STSI/STST/STSGT (our model) ONLY
    local_adj = construct_adj(A=adj, steps=args.strides)
    local_adj = torch.FloatTensor(local_adj)

    dataloader = load_dataset(dataset_dir=args.data,
                              batch_size=args.batch_size,
                              valid_batch_size=args.valid_batch_size,
                              test_batch_size=args.test_batch_size,
                              fill_zeros=args.fill_zeros
                              )

    scaler = dataloader['scaler']

    log_string(log, 'Loading Data ...')

    log_string(log, "The shape of original spatial adjacency matrix: {}".format(adj.shape))

    log_string(log, f'x_train: {torch.tensor(dataloader["train_loader"].xs).shape}\t\t '
                    f'y_train: {torch.tensor(dataloader["train_loader"].ys).shape}')
    log_string(log, f'x_val:   {torch.tensor(dataloader["val_loader"].xs).shape}\t\t'
                    f'y_val:   {torch.tensor(dataloader["val_loader"].ys).shape}')
    log_string(log, f'x_test:   {torch.tensor(dataloader["test_loader"].xs).shape}\t\t'
                    f'y_test:   {torch.tensor(dataloader["test_loader"].ys).shape}')
    log_string(log, f'mean:   {scaler.mean:.4f}\t\tstd:   {scaler.std:.4f}')
    log_string(log, 'Data Loaded !!')

    engine = trainer(scaler=scaler,
                     adj=local_adj,
                     history=args.history,
                     num_of_vertices=args.num_of_vertices,
                     in_dim=args.in_dim,
                     hidden_dims=args.hidden_dims,
                     first_layer_embedding_size=args.first_layer_embedding_size,
                     out_layer_dim=args.out_layer_dim,
                     d_model=args.d_model,
                     n_heads=args.n_heads,
                     factor=args.factor,
                     attention_dropout=args.attention_dropout,
                     output_attention=args.output_attention,
                     dropout=args.dropout,
                     forward_expansion=args.forward_expansion,
                     log=log,
                     lrate=args.learning_rate,
                     w_decay=args.weight_decay,
                     l_decay_rate=args.lr_decay_rate,
                     device=device,
                     activation=args.activation,
                     use_mask=args.use_mask,
                     max_grad_norm=args.max_grad_norm,
                     lr_decay=args.lr_decay,
                     temporal_emb=args.temporal_emb,
                     spatial_emb=args.spatial_emb,
                     use_transformer=args.use_transformer,
                     use_informer=args.use_informer,
                     horizon=args.horizon,
                     strides=args.strides)

    # Start Training
    if args.use_informer:
        log_string(log, 'Using Spatio-Temporal Synchronous Informer\'s Prob-sparse Self-Attention ...')
    else:
        log_string(log, 'Using Spatio-Temporal Synchronous Transformer\'s Full Self-Attention ...')

    log_string(log, 'Training Model ...')
    his_loss = []
    val_time = []
    train_time = []

    wait = 0
    val_mae_min = float('inf')
    best_model_wts = None

    train_loss = []
    train_mae = []
    train_rmse = []
    train_rmsle = []

    valid_loss = []
    valid_mae = []
    valid_rmse = []
    valid_rmsle = []

    for i in tqdm.tqdm(range(1, args.epochs + 1)):
        if wait >= args.patience:
            log_string(log, f'early stop at epoch: {i:04d}')
            break

        train_loss.clear()
        train_mae.clear()
        train_rmse.clear()
        train_rmsle.clear()

        t1 = time.time()
        dataloader['train_loader'].shuffle()

        for ix, (x, y) in enumerate(dataloader['train_loader'].get_iterator()):

            x_train = torch.Tensor(x).to(device)  # [B, T, N, C]
            # print("Train Data Batch Size: ", x_train.shape)

            y_train = torch.Tensor(y[:, :, :, 0]).to(device)  # [B, T, N]
            # print("YTrain Label Batch Size: ", y_train.shape)

            loss, tmae, trmse, trmsle = engine.train_model(x_train, y_train)

            train_loss.append(loss)
            train_mae.append(tmae)
            train_rmse.append(trmse)
            train_rmsle.append(trmsle)

            if ix % args.print_every == 0:
                logs = 'Iter: {:03d}, Train Loss: {:.4f}, Train MAE: {:.4f}, ' \
                       'Train RMSE: {:.4f}, Train RMSLE: {:.4f}, lr: {}'
                print(logs.format(ix, train_loss[-1], train_mae[-1], train_rmse[-1], train_rmsle[-1],
                                  engine.optimizer.param_groups[0]['lr']), flush=True)

        if args.lr_decay:
            engine.lr_scheduler.step()

        t2 = time.time()
        train_time.append(t2 - t1)

        valid_loss.clear()
        valid_mae.clear()
        valid_rmse.clear()
        valid_rmsle.clear()

        s1 = time.time()
        for ix, (x, y) in enumerate(dataloader['val_loader'].get_iterator()):
            x_val = torch.Tensor(x).to(device)  # [B, T, N, C]
            # print("Val Data Batch Size: ", x_val.shape)

            y_val = torch.Tensor(y[:, :, :, 0]).to(device)  # [B, T, N]
            # print("Val Label Batch Size: ", y_val.shape)

            vloss, vmae, vrmse, vrmsle = engine.eval_model(x_val, y_val)
            valid_loss.append(vloss)
            valid_mae.append(vmae)
            valid_rmse.append(vrmse)
            valid_rmsle.append(vrmsle)

        s2 = time.time()
        logs = 'Epoch: {:03d}, Inference Time: {:.4f} secs'
        log_string(log, logs.format(i, (s2-s1)))

        val_time.append(s2 - s1)

        mtrain_loss = np.mean(train_loss)
        mtrain_mae = np.mean(train_mae)
        mtrain_rmse = np.mean(train_rmse)
        mtrain_rmsle = np.mean(train_rmsle)

        mvalid_loss = np.mean(valid_loss)
        mvalid_mae = np.mean(valid_mae)
        mvalid_rmse = np.mean(valid_rmse)
        mvalid_rmsle = np.mean(valid_rmsle)
        his_loss.append(mvalid_loss)

        logs = 'Epoch: {:03d}, Train Loss: {:.4f}, Train MAE {:.4f}, ' \
               'Train RMSE: {:.4f}, Train RMSLE: {:.4f}, Valid Loss: {:.4f}, Valid MAE: {:.4f}, ' \
               'Valid RMSE: {:.4f}, Valid RMSLE: {:.4f}, Training Time: {:.4f}/epoch'
        log_string(log, logs.format(i, mtrain_loss, mtrain_mae, mtrain_rmse,
                                    mtrain_rmsle, mvalid_loss, mvalid_mae,
                                    mvalid_rmse, mvalid_rmsle, (t2 - t1)))

        if not os.path.exists(args.save):
            os.makedirs(args.save)

        if val_mae_min >= mvalid_mae > 0:
            log_string(
                log,
                f'Validation MAE decreases from {val_mae_min:.4f} to {mvalid_mae:.4f}, '
                f'save model to '
                f'{args.save + "exp_" + str(args.expid) + "_" + str(round(mvalid_mae, 2)) + "_best_model.pth"}'
            )
            wait = 0
            val_mae_min = mvalid_mae
            best_model_wts = engine.model.state_dict()
            torch.save(best_model_wts,
                       args.save + "exp_" + str(args.expid) + "_" + str(round(val_mae_min, 2)) + "_best_model.pth")
        else:
            wait += 1

        np.save(f'{args.save_loss}' + 'history_loss' + f'_{args.expid}', his_loss)

    log_string(log, 'Training Completed ...')
    log_string(log, "The Validation MAE of the best model is " + str(round(val_mae_min, 2)))

    log_string(log, "Average Training Time: {:.4f} secs/epoch".format(np.mean(train_time)))
    log_string(log, "Average Inference Time: {:.4f} secs".format(np.mean(val_time)))

    # Test
    log_string(log, 'Testing Model ...')
    engine.model.load_state_dict(
        torch.load(args.save + "exp_" + str(args.expid) + "_" + str(round(val_mae_min, 2)) + "_best_model.pth"))

    outputs = []
    y_real = torch.Tensor(dataloader['y_test'][:, :, :, 0]).to(device)  # (no_test_samples, T, N)
    # print("y_real shape: ", y_real.shape)

    for ix, (x, y) in tqdm.tqdm(enumerate(dataloader['test_loader'].get_iterator())):
        x_test = torch.Tensor(x).to(device)  # [B, T, N, C]
        # print("TestX shape: ", x_test.shape)
        with torch.no_grad():
            y_pred = engine.model(x_test)  # [B, T, N] - For our STSI, GraphWaveNet, ASTGCN-r, STTN
            # y_pred = engine.model(local_adj, x_test)  # [B, T, N] - For STGCN model ONLY
            # print("y_pred shape: ", y_pred.shape)

        outputs.append(y_pred)

    y_hat = torch.cat(outputs, dim=0)  # [B, T, N]
    # print("y_hat shape: ", y_hat.shape)

    # The following is done because when you are doing batch,
    # you can pad out a new sample to meet the batch_size requirements
    # y_hat = y_hat[:y_real.size(0), ...]  # [B, T, N]
    y_real = y_real[:y_hat.size(0), ...]  # [B, T, N]
    # print("y_real shape: ", y_real.shape)

    amae = []
    armse = []
    armsle = []

    for t in range(args.horizon):
        pred = scaler.inverse_transform(y_hat[:, t, :])
        real = y_real[:, t, :]

        mae, rmse, rmsle = metric(pred, real)
        logs = 'The best model on the test set for horizon: {:d}, ' \
               'Test MAE: {:.4f}, Test RMSE: {:.4f}, Test RMSLE: {:.4f}'

        log_string(log, logs.format(t+1, mae, rmse, rmsle))
        amae.append(mae)
        armse.append(rmse)
        armsle.append(rmsle)

    logs = 'On average over 12 horizons, Test MAE: {:.4f}, Test RMSE: {:.4f}, Test RMSLE: {:.4f}'
    log_string(log, logs.format(np.mean(amae), np.mean(armse), np.mean(armsle)))
    log_string(log, 'Testing Completed ...')


if __name__ == '__main__':
    start = time.time()
    main()
    end = time.time()

    log_string(log, 'total time: %.2fhours' % ((end - start) / 3600))
    log.close()



