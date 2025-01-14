'''
Utility functions for data processing.
'''

import copy
from scipy.sparse import coo_matrix
import numpy as np
import pandas as pd
import torch
import torch_geometric
# import SPD
import networkx as nx
from tqdm import tqdm

from config import Config

def load_dataset_pytorch():
    '''Loads the data for the given population into a list of Pytorch Geometric
    Data objects, which then can be used to create DataLoaders.
    '''
    connectomes = torch.load(f"{Config.DATA_FOLDER}connectome.ts")
    scores = torch.load(f"{Config.DATA_FOLDER}score.ts")

    connectomes[connectomes < 0] = 0

    pyg_data = []
    for subject in range(scores.shape[0]):
        sparse_mat = to_sparse(connectomes[:, :, subject])
        pyg_data.append(torch_geometric.data.Data(x=torch.eye(Config.ROI, dtype=torch.float),
                                                  y=scores[subject].float(), edge_index=sparse_mat._indices(),
                                                  edge_attr=sparse_mat._values().float()))

    return pyg_data


def to_sparse(mat):
    '''Transforms a square matrix to torch.sparse tensor

    Methods ._indices() and ._values() can be used to access to
    edge_index and edge_attr while generating Data objects
    '''
    coo = coo_matrix(mat, dtype='float64')
    row = torch.from_numpy(coo.row.astype(np.int64))
    col = torch.from_numpy(coo.col.astype(np.int64))
    coo_index = torch.stack([row, col], dim=0)
    coo_values = torch.from_numpy(coo.data.astype(np.float64).reshape(-1, 1)).reshape(-1)
    sparse_mat = torch.sparse.LongTensor(coo_index, coo_values)
    return sparse_mat


def load_dataset_cpm():
    '''Loads the data for given population in the upper triangular matrix form
    as required by CPM functions.
    '''
    connectomes = np.array(torch.load(f"{Config.DATA_FOLDER}connectome.ts"))
    scores = np.array(torch.load(f"{Config.DATA_FOLDER}score.ts"))

    fc_data = {}
    behav_data = {}
    for subject in range(scores.shape[0]):  # take upper triangular part of each matrix
        fc_data[subject] = connectomes[:, :, subject][np.triu_indices_from(connectomes[:, :, subject], k=1)]
        behav_data[subject] = {'score': scores[subject].item()}
    return pd.DataFrame.from_dict(fc_data, orient='index'), pd.DataFrame.from_dict(behav_data, orient='index')


def get_loaders(train, test, batch_size=1):
    '''Returns data loaders for given data lists
    '''
    # train_loader = torch_geometric.data.DataLoader(train, batch_size=batch_size)
    # test_loader = torch_geometric.data.DataLoader(test, batch_size=batch_size)
    train_loader = torch_geometric.loader.DataLoader(train, batch_size=batch_size)
    test_loader = torch_geometric.loader.DataLoader(test, batch_size=batch_size)
    return train_loader, test_loader


def load_dataset_tensor(pop="NT"):
    '''Loads dataset as tuple of (tensor of connectomes,
       tensor of fiq scores, tensor of viq scores)
    '''
    connectomes = torch.load(f"{Config.DATA_FOLDER}connectome.ts")
    scores = torch.load(f"{Config.DATA_FOLDER}score.ts")
    return connectomes, scores


def to_dense(data):
    '''Returns a copy of the data object in Dense form.
    '''
    denser = torch_geometric.transforms.ToDense()
    copy_data = denser(copy.deepcopy(data))
    return copy_data
