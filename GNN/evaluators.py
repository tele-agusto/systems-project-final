'''
Functions for k-fold evaluation of models.
'''

import random
import pickle
import numpy as np
from sklearn.model_selection import KFold
import torch

import data_utils
from reggnn import RegGNN
from reggnn import RegGNN3

from config import Config
import UTR as utr
# import UTR_full as utr
import pandas as pd
import torch_geometric
from sklearn.decomposition import PCA
from numpy.random import shuffle
from torch_geometric.transforms import FeaturePropagation

def return_h5_path(file):
    path = f'/content/drive/MyDrive/{file}'
    return path


def evaluate_RegGNN(sample_selection=False, shuffle=False, random_state=None,
                    dropout=0.1, k_list=list(range(2, 16)), lr=3e-5, wd=5e-4,
                    device=torch.device('cpu'), num_epoch=150, n_select_splits=10):
    if sample_selection is False:
        k_list = [0]

    overall_preds = {k: [] for k in k_list}
    overall_scores = {k: [] for k in k_list}
    train_mae = {k: [] for k in k_list}
    
    # 5' utr adjacency matrix
    adj_t = torch.tensor(utr.adjacency)
    
    ## fully-connected adjacency matrix
    # adj_t = torch.ones(110,110)
    # adj_t = torch.ones(126,126)
    # print(len(adj_t))

    ## random adjacency matrix
    # adj_t = np.random.choice([0,1],12100,p=[0.99083,0.00917]).reshape(110,110)

    adj_t = torch.tensor(adj_t)
   
    # convert adjacency matrix to edge index
    edge_index = adj_t.nonzero().contiguous()
    edge_index = edge_index
    edge_index = edge_index.t()

    # Load data and metadata
    pd.options.display.precision = 5
    df_cite_targets = pd.read_hdf(return_h5_path('train_cite_targets.h5'))
    # df_cite_targets = df_cite_targets.iloc[0:1000,:]
    df_cite_inputs = pd.read_hdf(return_h5_path('train_cite_inputs.h5'))
    # df_cite_inputs = df_cite_inputs.iloc[0:1000,:]
    df_meta = pd.read_csv(return_h5_path('metadata.csv'), index_col='cell_id')

    
    # get indexes for samples collected from each donor on a given day
    df_meta = df_meta[df_meta.technology == 'citeseq']
    df_meta = df_meta[df_meta.day != 7]
    df_meta = df_meta[df_meta.donor != 27678]
    df_meta_2 = df_meta[df_meta.day == 2]
    df_meta_3 = df_meta[df_meta.day == 3]
    df_meta_4 = df_meta[df_meta.day == 4]
    df_meta_2_32606 = df_meta_2[df_meta_2.donor == 32606]
    df_meta_3_32606 = df_meta_3[df_meta_3.donor == 32606]
    df_meta_4_32606 = df_meta_4[df_meta_4.donor == 32606]
    df_meta_2_13176 = df_meta_2[df_meta_2.donor == 13176]
    df_meta_3_13176 = df_meta_3[df_meta_3.donor == 13176]
    df_meta_4_13176 = df_meta_4[df_meta_4.donor == 13176]
    df_meta_2_31800 = df_meta_2[df_meta_2.donor == 31800]
    df_meta_3_31800 = df_meta_3[df_meta_3.donor == 31800]
    df_meta_4_31800 = df_meta_4[df_meta_4.donor == 31800]

    
    # order inputs and targets by donor and day + drop columns of genes w/o corresponding proteins
    inputs = df_cite_inputs.loc[df_meta.index]
    # inputs_gnn = inputs.reindex(indexes)
    targets = df_cite_targets.loc[df_meta.index]
    to_drop = [column for column in inputs.columns if column not in list(utr.utrs.Full_gene)]
    print(len(to_drop))
    # to_drop = np.hstack((to_drop, ['ENSG00000185896_LAMP1','ENSG00000169442_CD52','ENSG00000213949_ITGA1','ENSG00000160791_CCR5','ENSG00000134460_IL2RA','ENSG00000117091_CD48','ENSG00000100031_GGT1','ENSG00000197405_C5AR1','ENSG00000010278_CD9','ENSG00000150337_FCGR1A','ENSG00000211898_IGHD','ENSG00000121807_CCR2','ENSG00000211899_IGHM','ENSG00000185291_IL3RA','ENSG00000116824_CD2','ENSG00000090339_ICAM1','ENSG00000143226_FCGR2A','ENSG00000026508_CD44','ENSG00000110848_CD69','ENSG00000150093_ITGB1','ENSG00000110651_CD81','ENSG00000204592_HLA-E','ENSG00000170458_CD14','ENSG00000101017_CD40','ENSG00000137101_CD72','ENSG00000012124_CD22','ENSG00000149294_NCAM1','ENSG00000114013_CD86','ENSG00000177455_CD19','ENSG00000197635_DPP4','ENSG00000138185_ENTPD1','ENSG00000166825_ANPEP','ENSG00000102245_CD40LG','ENSG00000206503_HLA-A','ENSG00000089692_LAG3']))
    # print(len(to_drop))
    inputs_gnn = inputs.drop(inputs[to_drop], axis=1)
    print(inputs.columns)
    print(inputs_gnn.columns)
    print(to_drop)

   
    neurons = inputs_gnn.shape[1]

    inputs_gnn_1 = inputs_gnn.loc[df_meta_2_32606.index]
    inputs_gnn_2 = inputs_gnn.loc[df_meta_3_32606.index]
    inputs_gnn_3 = inputs_gnn.loc[df_meta_4_32606.index]
    inputs_gnn_4 = inputs_gnn.loc[df_meta_2_13176.index]
    inputs_gnn_5 = inputs_gnn.loc[df_meta_3_13176.index]
    inputs_gnn_6 = inputs_gnn.loc[df_meta_4_13176.index]
    inputs_gnn_7 = inputs_gnn.loc[df_meta_2_31800.index]
    inputs_gnn_8 = inputs_gnn.loc[df_meta_3_31800.index]
    inputs_gnn_9 = inputs_gnn.loc[df_meta_4_31800.index]
    
 
    
    print(inputs_gnn.columns)
    print(utr.utrs.Full_gene)
    print(utr.utrs_sorted.Full_gene)
    

    # day cv
    inputs_day1 = np.vstack((inputs_gnn_1,inputs_gnn_4,inputs_gnn_7))
    inputs_day2 = np.vstack((inputs_gnn_2,inputs_gnn_5,inputs_gnn_8))
    inputs_day3 = np.vstack((inputs_gnn_3,inputs_gnn_6,inputs_gnn_9))
    inputs_gnn = np.vstack((inputs_day1, inputs_day2, inputs_day3))
    # inputs_gnn = np.vstack((inputs_gnn_1,inputs_gnn_4,inputs_gnn_7,inputs_gnn_2,inputs_gnn_5,inputs_gnn_8,inputs_gnn_3,inputs_gnn_6,inputs_gnn_9))


  
    to_drop_2 = [column for column in targets.columns if column not in list(utr.utrs.Protein)]
    print(len(to_drop_2))
    print(inputs_gnn.shape)
    # print(utr.utrs.Full_gene)
    # print(utr.utrs_sorted.Full_gene)
    targets_gnn = targets.drop(targets[to_drop_2], axis=1)
    print(targets_gnn.shape)
    pd.DataFrame(targets_gnn.columns).to_csv('gnnproteinorder.csv')
    targets_gnn.round(decimals=3)
    targets_gnn_1 = targets_gnn.loc[df_meta_2_32606.index]
    targets_gnn_2 = targets_gnn.loc[df_meta_3_32606.index]
    targets_gnn_3 = targets_gnn.loc[df_meta_4_32606.index]
    targets_gnn_4 = targets_gnn.loc[df_meta_2_13176.index]
    targets_gnn_5 = targets_gnn.loc[df_meta_3_13176.index]
    targets_gnn_6 = targets_gnn.loc[df_meta_4_13176.index]
    targets_gnn_7 = targets_gnn.loc[df_meta_2_31800.index]
    targets_gnn_8 = targets_gnn.loc[df_meta_3_31800.index]
    targets_gnn_9 = targets_gnn.loc[df_meta_4_31800.index]
    # donor cv
    # targets_gnn = np.vstack((targets_gnn_1,targets_gnn_2,targets_gnn_3,targets_gnn_4,targets_gnn_5,targets_gnn_6,targets_gnn_7,targets_gnn_8,targets_gnn_9))
    # day cv
    targets_gnn = np.vstack((targets_gnn_1,targets_gnn_4,targets_gnn_7,targets_gnn_2,targets_gnn_5,targets_gnn_8,targets_gnn_3,targets_gnn_6,targets_gnn_9))
    targets_gnn = targets_gnn[0:70656,:]
    
    inputs_gnn = torch.tensor(inputs_gnn)
    inputs_gnn = torch.t(inputs_gnn)
    targets_gnn = torch.tensor(targets_gnn)
    targets_gnn = torch.t(targets_gnn)
    print(targets_gnn[0:2, 0:20])


    # load batches into Data object for gnn training
    pyg_data = []
    for i in range(552):
        inputs_batch = inputs_gnn[:, 128 * i:128 * (i + 1)]
        pyg_data.append(torch_geometric.data.Data(x=inputs_batch,
                                                  y=torch.round(targets_gnn[:, 128 * i:128 * (i + 1)].float(),
                                                                decimals=3), edge_index=edge_index,
                                                  edge_attr=None))


    data = pyg_data
    

    # donor cv
    train_test_split = [[list(range(0,361)), list(range(361,552))], [list(range(187,552)), list(range(0,187))], [np.hstack((list((range(0,187))), list(range(361,552)))), list(range(187,361))]]
    
    # day cv
    # train_test_split = [[list(range(0,334)), list(range(334,552))], [list(range(171,552)), list(range(0,171))], [np.hstack((list((range(0,171))), list(range(334,552)))), list(range(171,334))]]

    fold = -1
    # for train_idx, test_idx in KFold(Config.K_FOLDS, shuffle=shuffle,
    #                                  random_state=random_state).split(data):
    for train_idx, test_idx in train_test_split:
        print(train_idx)
        print(test_idx)
        fold += 1
        print(f"Cross Validation Fold {fold + 1}/{Config.K_FOLDS}")
        if sample_selection:
            sample_atlas = sampsel.select_samples(train_idx, n_select_splits, k_list,
                                                  data_dict, score_dict, shuffle,
                                                  random_state)

        for k in k_list:
            if sample_selection:
                selected_train_data = [data[subject] for subject in sample_atlas[k]]
            else:
                selected_train_data = [data[i] for i in train_idx]
            test_data = [data[i] for i in test_idx]

            # candidate_model = RegGNN(7476, 512, 110, dropout).float().to(device)
            candidate_model = RegGNN(110,110,128).float().to(device)
            # candidate_model = RegGNN3(70).float().to(device)
            optimizer = torch.optim.Adam(candidate_model.parameters(), lr=lr, weight_decay=wd)
            train_loader, test_loader = data_utils.get_loaders(selected_train_data, test_data)
            candidate_model.train()
            for epoch in range(num_epoch):
                preds = []
                scores = []
                for batch in train_loader:
                    out = candidate_model(batch.x.to(device, dtype = torch.float32), data_utils.to_dense(batch).adj.to(device, dtype=torch.int64))
                    # out = candidate_model(batch.x.to(device), data_utils.to_dense(batch).edge_index.to(device, dtype=torch.int32))
                    # loss = candidate_model.loss(out.view(-1, 1), batch.y.to(device).view(-1, 1))
                    loss = candidate_model.loss(out.reshape(-1, 1), batch.y.to(device).reshape(-1, 1))
                    if epoch % 50 == 0:
                        print(loss)

                    candidate_model.zero_grad()
                    loss.backward()
                    optimizer.step()
                    preds.append(out.cpu().data.numpy())
                    scores.append(batch.y.numpy())
                
                preds = np.hstack(preds)
                scores = np.hstack(scores)
                epoch_mae = np.mean(np.abs(preds.reshape(-1, 1) - scores.reshape(-1, 1)))
                train_mae[k].append(epoch_mae)

                for batch in test_loader:
                    out2 = candidate_model(batch.x.to(device, dtype = torch.float32),
                                          data_utils.to_dense(batch).adj.to(device, dtype=torch.int64))
                    
                    loss2 = candidate_model.loss(out2.reshape(-1, 1), batch.y.to(device).reshape(-1, 1))
                    if epoch % 50 == 0:
                      print(f'test:{loss2}')


            candidate_model.eval()
            with torch.no_grad():
                preds = []
                scores = []
                for batch in test_loader:
                    out = candidate_model(batch.x.to(device, dtype = torch.float32),
                                          data_utils.to_dense(batch).adj.to(device, dtype=torch.int64))
                   
                    loss = candidate_model.loss(out.reshape(-1, 1), batch.y.to(device).reshape(-1, 1))
                    preds.append(out.cpu().data.numpy())
                    scores.append(batch.y.cpu().numpy())

                preds = np.hstack(preds)
                scores = np.hstack(scores)

            overall_preds[k].extend(preds)
            overall_scores[k].extend(scores)

    for k in k_list:
        overall_preds[k] = np.hstack(overall_preds[k]).ravel()
        overall_scores[k] = np.hstack(overall_scores[k]).ravel()

    if sample_selection is False:
        overall_preds = overall_preds[k_list[0]]
        overall_scores = overall_scores[k_list[0]]
        overall_scores = np.around(overall_scores, decimals=3)
        print(overall_scores[0:20])

    return overall_preds, overall_scores, train_mae
