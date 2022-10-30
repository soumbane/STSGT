import os
import torch
import random
import numpy as np
import pandas as pd

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def log_string(log, string):
    """Print log"""
    log.write(string + '\n')
    log.flush()
    print(string)


def count_parameters(model):
    """Statistical Model Parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def init_seed(seed):
    """Disable cudnn to maximize reproducibility"""
    torch.cuda.cudnn_enabled = False
    """
    cuDNN uses a non-deterministic algorithm and can be disabled using torch.backends.cudnn.enabled = False
    If it is set to torch.backends.cudnn.enabled=True, it means it is set to use a non-deterministic 
    algorithm
    Then set: torch.backends.cudnn.benchmark = True, when this flag is True, 
    it will make the program spend a little extra time at the beginning,
    Search for the most suitable convolution implementation algorithm for each convolutional layer of the 
    entire network, thereby achieving network acceleration
    But because it uses a non-deterministic algorithm, this will make the network feedforward 
    results slightly different each time. If you want to avoid this kind of result fluctuation, 
    you can set the following flag to True
    """
    # torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


"""Read and Construct Localized Spatio-Temporal Adjacency Matrix"""


def get_adjacency_matrix(distance_df_filename):
    """
    distance_df_filename: str, adj csv file path
    """
    df = pd.read_csv(distance_df_filename, header=None)
    adj_mx = df.values

    # Scale the adjacency matrix
    adj_mx_final = adj_mx / np.max(adj_mx)

    # Threshold for the 51 States or 83 counties
    thresh = 0.3
    adj_mx_final[adj_mx_final > thresh] = 0.

    return adj_mx_final


def construct_adj(A, steps):
    """
    Build a spatial-temporal graph
    A: np.ndarray, adjacency matrix, shape is (N, N)
    steps: select a few time steps to build the graph
    return: new adjacency matrix: csr_matrix, shape is (N * steps, N * steps)
    """
    N = len(A)  # Get the number of rows
    adj = np.zeros((N * steps, N * steps))

    for i in range(steps):
        """The diagonal represents the space map of each time step, which is A"""
        adj[i * N: (i + 1) * N, i * N: (i + 1) * N] = A

    for i in range(N):
        for k in range(steps - 1):
            """Each node will only connect to itself in adjacent time steps"""
            adj[k * N + i, (k + 1) * N + i] = 1.
            adj[(k + 1) * N + i, k * N + i] = 1.

    return adj


class DataLoader(object):
    def __init__(self, xs, ys, batch_size, pad_with_last_sample=False):
        """
        Data loader
        :param xs: training data
        :param ys: label data
        :param batch_size:batch size
        :param pad_with_last_sample: When the remaining data is not enough,
        whether to copy the last sample to reach the batch size
        """
        self.batch_size = batch_size
        self.current_ind = 0
        if pad_with_last_sample:
            num_padding = (batch_size - (len(xs) % batch_size)) % batch_size
            x_padding = np.repeat(xs[-1:], num_padding, axis=0)
            y_padding = np.repeat(ys[-1:], num_padding, axis=0)
            xs = np.concatenate([xs, x_padding], axis=0)
            ys = np.concatenate([ys, y_padding], axis=0)

        self.size = len(xs)
        self.num_batch = int(self.size // self.batch_size)
        self.xs = xs
        self.ys = ys

    def shuffle(self):
        """Shuffle Dataset"""
        permutation = np.random.permutation(self.size)
        xs, ys = self.xs[permutation], self.ys[permutation]
        self.xs = xs
        self.ys = ys

    def get_iterator(self):
        self.current_ind = 0

        def _wrapper():
            while self.current_ind < self.num_batch:
                start_ind = self.batch_size * self.current_ind
                end_ind = min(self.size, self.batch_size * (self.current_ind + 1))
                x_i = self.xs[start_ind:end_ind, ...]
                y_i = self.ys[start_ind:end_ind, ...]
                yield x_i, y_i
                self.current_ind += 1

        return _wrapper()


class StandardScaler:
    """Standardize the input using standard mean sub and std div"""
    def __init__(self, mean, std, fill_zeros=False):
        self.mean = mean
        self.std = std
        self.fill_zeros = fill_zeros

    def transform(self, data):
        if self.fill_zeros:
            mask = (data == 0)
            data[mask] = self.mean
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def load_dataset(dataset_dir, batch_size, valid_batch_size, test_batch_size, fill_zeros=False):
    """
    Load data set
    dataset_dir: dataset directory
    batch_size: Training batch size
    valid_batch_size: validation set batch size
    test_batch_size: test set batch size
    fill_zeros: (bool) whether to fill zeros in data with average
    """
    data = {}
    for category in ['train', 'val', 'test']:
        cat_data = np.load(os.path.join(dataset_dir, category + '.npz'))
        data['x_' + category] = cat_data['x']
        data['y_' + category] = cat_data['y']

    scaler = StandardScaler(mean=data['x_train'][..., 0].mean(),
                            std=data['x_train'][..., 0].std(),
                            fill_zeros=fill_zeros)

    for category in ['train', 'val', 'test']:
        data['x_' + category][..., 0] = scaler.transform(data['x_' + category][..., 0])

    data['train_loader'] = DataLoader(data['x_train'], data['y_train'], batch_size)
    data['val_loader'] = DataLoader(data['x_val'], data['y_val'], valid_batch_size)
    data['test_loader'] = DataLoader(data['x_test'], data['y_test'], test_batch_size)
    data['scaler'] = scaler

    return data


def masked_mse(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)

    # Calculate Mask
    mask = mask.float()
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)

    # Masked MSE Loss
    loss = (preds-labels)**2
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)

    return torch.mean(loss)


def masked_rmse(preds, labels, null_val=np.nan):
    # Masked RMSE Loss
    return torch.sqrt(masked_mse(preds=preds, labels=labels, null_val=null_val))


def masked_rmsle(preds, labels, null_val=np.nan):
    # Masked RMSLE Loss
    # loss = (torch.log(torch.abs(preds) + 1) - torch.log(torch.abs(labels) + 1)) ** 2

    return torch.sqrt(masked_mse(preds=torch.log(torch.abs(preds) + 1),
                                 labels=torch.log(torch.abs(labels) + 1),
                                 null_val=null_val))


def masked_mae(preds, labels, null_val=np.nan):

    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)

    # Calculate Mask
    mask = mask.float()
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)

    # Masked MAE Loss
    loss = torch.abs(preds-labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)

    return torch.mean(loss)


def metric(pred, real):
    mae = masked_mae(pred, real, 0.0).item()
    rmse = masked_rmse(pred, real, 0.0).item()
    rmsle = masked_rmsle(pred, real, 0.0).item()

    return mae, rmse, rmsle

