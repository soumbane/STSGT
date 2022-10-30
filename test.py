from utils import *
import argparse
from models.model import *
import tqdm
import numpy as np
import os
import configparser
import ast
from engine import trainer

DATASET = 'COVID_JHU'  # COVID Dataset from Johns Hopkins University
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
                    help="The discrete time series of each sample output")
parser.add_argument("--strides", type=int, default=config['model']['strides'],
                    help="The step size of the sliding window, the local spatio-temporal graph "
                         "is constructed using several time steps, the default is 3")
parser.add_argument("--temporal_emb", type=eval, default=config['model']['temporal_emb'],
                    help="Whether to use temporal embedding vector")
parser.add_argument("--spatial_emb", type=eval, default=config['model']['spatial_emb'],
                    help="Whether to use spatial embedding vector")
parser.add_argument("--use_mask", type=eval, default=config['model']['use_mask'],
                    help="Whether to use the mask matrix to optimize adj")
parser.add_argument("--activation", type=str, default=config['model']['activation'],
                    help="Activation Function {relu, GlU}")

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

parser.add_argument("--learning_rate", type=float, default=config['train']['learning_rate'],
                    help="Initial Learning Rate")
parser.add_argument("--weight_decay", type=float, default=config['train']['weight_decay'],
                    help="Weight Decay Rate")
parser.add_argument("--lr_decay", type=eval, default=config['train']['lr_decay'],
                    help="Whether to enable the initial learning rate decay strategy")
parser.add_argument("--lr_decay_rate", type=float, default=config['train']['lr_decay_rate'],
                    help="Learning rate decay rate")
parser.add_argument('--max_grad_norm', type=float, default=config['train']['max_grad_norm'],
                    help="Gradient Threshold")

parser.add_argument('--log_file', default=config['test']['log_file'], help='log file')
parser.add_argument('--checkpoint', type=str, help='')

args = parser.parse_args()


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


log = open(args.log_file, 'w')
log_string(log, str(args))


def main():
    # load data
    adj = get_adjacency_matrix(distance_df_filename=args.sensors_distance)

    # local_adj for STSI/STST/STSGT (our model) ONLY
    local_adj = construct_adj(A=adj, steps=args.strides)
    local_adj = torch.FloatTensor(local_adj)

    dataloader = load_dataset(dataset_dir=args.data,
                              batch_size=args.batch_size,
                              valid_batch_size=args.valid_batch_size,
                              test_batch_size=args.test_batch_size
                              )

    scaler = dataloader['scaler']

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

    # Load Pre-trained Model
    engine.model.load_state_dict(torch.load(args.checkpoint))

    if args.use_informer:
        log_string(log, 'Using Spatio-Temporal Synchronous Informer\'s Prob-sparse Self-Attention ...')
    else:
        log_string(log, 'Using Spatio-Temporal Synchronous Transformer\'s Full Self-Attention ...')

    log_string(log, 'Model loaded successfully ...')

    engine.model.eval()

    outputs = []
    y_real = torch.Tensor(dataloader['y_test'][:, :, :, 0]).to(device)  # (no_test_samples, T, N)

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

        log_string(log, logs.format(t + 1, mae, rmse, rmsle))
        amae.append(mae)
        armse.append(rmse)
        armsle.append(rmsle)

    logs = 'On average over 12 horizons, Test MAE: {:.4f}, Test RMSE: {:.4f}, Test RMSLE: {:.4f}'
    log_string(log, logs.format(np.mean(amae), np.mean(armse), np.mean(armsle)))
    log_string(log, 'Testing Completed ...')

    '''# The following is for plotting MI Wayne County only:
    y_real_81 = y_real[11:46, 0, 81].cpu().detach().numpy()

    y_real_81_non_zero = y_real_81[np.where(y_real_81 != 0)]
    print(y_real_81_non_zero)
    # np.save(os.path.join('data/COVID_JHU/MI_Wayne_County_Infected_Cases_Plots', 'y_real'), y_real_81_non_zero)

    y_hat_81 = scaler.inverse_transform(y_hat[11:46, 0, 81]).cpu().detach().numpy()
    y_hat_81_non_zero = y_hat_81[np.where(y_real_81 != 0)]
    print(y_hat_81_non_zero)
    # np.save(os.path.join('data/COVID_JHU/MI_Wayne_County_Infected_Cases_Plots', 'y_pred_STST'), y_hat_81_non_zero)
    # np.save(os.path.join('data/COVID_JHU/MI_Wayne_County_Infected_Cases_Plots', 'y_pred_GWNet'), y_hat_81_non_zero)
    # np.save(os.path.join('data/COVID_JHU/MI_Wayne_County_Infected_Cases_Plots', 'y_pred_ASTGCN'), y_hat_81_non_zero)
    # np.save(os.path.join('data/COVID_JHU/MI_Wayne_County_Infected_Cases_Plots', 'y_pred_STTN'), y_hat_81_non_zero)
    # np.save(os.path.join('data/COVID_JHU/MI_Wayne_County_Infected_Cases_Plots', 'y_pred_STGCN'), y_hat_81_non_zero)'''

    '''# The following is for plotting MI Oakland County only:
    y_real_62 = y_real[11:46, 0, 62].cpu().detach().numpy()

    y_real_62_non_zero = y_real_62[np.where(y_real_62 != 0)]
    print(y_real_62_non_zero)
    # np.save(os.path.join('data/COVID_JHU/MI_Oakland_County_Infected_Cases_Plots', 'y_real'), y_real_62_non_zero)

    y_hat_62 = scaler.inverse_transform(y_hat[11:46, 0, 62]).cpu().detach().numpy()
    y_hat_62_non_zero = y_hat_62[np.where(y_real_62 != 0)]
    print(y_hat_62_non_zero)
    np.save(os.path.join('data/COVID_JHU/MI_Oakland_County_Infected_Cases_Plots', 'y_pred_STST'), y_hat_62_non_zero)
    # np.save(os.path.join('data/COVID_JHU/MI_Oakland_County_Infected_Cases_Plots', 'y_pred_GWNet'), y_hat_62_non_zero)
    # np.save(os.path.join('data/COVID_JHU/MI_Oakland_County_Infected_Cases_Plots', 'y_pred_ASTGCN'), y_hat_62_non_zero)
    # np.save(os.path.join('data/COVID_JHU/MI_Oakland_County_Infected_Cases_Plots', 'y_pred_STTN'), y_hat_62_non_zero)
    # np.save(os.path.join('data/COVID_JHU/MI_Oakland_County_Infected_Cases_Plots', 'y_pred_STGCN'), y_hat_62_non_zero)'''

    '''# The following is for plotting NY State only:
    y_real_32 = y_real[11:46, 0, 32].cpu().detach().numpy()
    # print(y_real_32)
    # print(len(y_real_32))

    y_real_32_non_zero = y_real_32[np.where(y_real_32 != 0)]
    print(y_real_32_non_zero)
    # np.save(os.path.join('data/COVID_JHU/NY_State_Infected_Cases_Plots', 'y_real'), y_real_32_non_zero)

    y_hat_32 = scaler.inverse_transform(y_hat[11:46, 0, 32]).cpu().detach().numpy()
    y_hat_32_non_zero = y_hat_32[np.where(y_real_32 != 0)]
    print(y_hat_32_non_zero)
    # np.save(os.path.join('data/COVID_JHU/NY_State_Infected_Cases_Plots', 'y_pred_STST'), y_hat_32_non_zero)
    # np.save(os.path.join('data/COVID_JHU/NY_State_Infected_Cases_Plots', 'y_pred_GWNet'), y_hat_32_non_zero)
    # np.save(os.path.join('data/COVID_JHU/NY_State_Infected_Cases_Plots', 'y_pred_ASTGCN'), y_hat_32_non_zero)
    # np.save(os.path.join('data/COVID_JHU/NY_State_Infected_Cases_Plots', 'y_pred_STTN'), y_hat_32_non_zero)
    # np.save(os.path.join('data/COVID_JHU/NY_State_Infected_Cases_Plots', 'y_pred_STGCN'), y_hat_32_non_zero)'''

    # The following is for plotting NY(ind. 32), CA(ind. 4), FL (9) and TX (43) State Bar Graph only:
    # ind 11: Oct 2, 2021/ind 24: Oct 15, 2021/ind 46: Nov 6, 2021
    # Nov 6 - Nov 17, 2021: Ground Truth Infected Cases
    # y_real_32 = y_real[46, :, 32].cpu().detach().numpy()  # for NY
    # y_real_32 = y_real[46, :, 4].cpu().detach().numpy()  # for CA
    # y_real_32 = y_real[46, :, 9].cpu().detach().numpy()  # for FL
    y_real_32 = y_real[46, :, 43].cpu().detach().numpy()  # for TX
    # print(y_real_32)
    print(len(y_real_32))

    y_real_32_non_zero = y_real_32[np.where(y_real_32 != 0)]
    print(y_real_32_non_zero)
    print(np.mean(y_real_32_non_zero))
    # print(len(y_real_32_non_zero))

    # Nov 6 - Nov 17, 2021: Predicted Infected Cases
    # y_hat_32 = scaler.inverse_transform(y_hat[46, :, 32]).cpu().detach().numpy()  # for NY
    # y_hat_32 = scaler.inverse_transform(y_hat[46, :, 4]).cpu().detach().numpy()  # for CA
    # y_hat_32 = scaler.inverse_transform(y_hat[46, :, 9]).cpu().detach().numpy()  # for FL
    y_hat_32 = scaler.inverse_transform(y_hat[46, :, 43]).cpu().detach().numpy()  # for TX
    y_hat_32_non_zero = y_hat_32[np.where(y_real_32 != 0)]
    print(y_hat_32_non_zero)
    print(np.mean(y_hat_32_non_zero))


if __name__ == "__main__":
    main()
    log.close()





