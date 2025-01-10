#Load packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import gc
import h5py
import hdf5plugin
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
#from sklearn.model_selection import RandomizedSearchCV
from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.model_selection import HalvingGridSearchCV
from numpy import random
from scipy.stats import uniform
import math
from sklearn.model_selection import GroupKFold


#function to load dataset
def return_h5_path(file):
    path = f'/open-problems-multimodal/{file}'
    return path

#Load data and metadata
df_cite_targets = pd.read_hdf(return_h5_path('train_cite_targets.h5'))
df_cite_inputs = pd.read_hdf(return_h5_path('train_cite_inputs.h5'))
df_meta = pd.read_csv('/open-problems-multimodal/metadata.csv', index_col='cell_id')

df_meta = df_meta[df_meta.technology=='citeseq']
df_meta = df_meta[df_meta.day!=7]
df_meta = df_meta[df_meta.donor!=27678]
df_meta_train = df_meta[df_meta.donor!=31800]
df_meta_test = df_meta[df_meta.donor==31800]

train_inputs = df_cite_inputs.loc[df_meta_train.index]
test_inputs = df_cite_inputs.loc[df_meta_test.index]
train_targets = df_cite_targets.loc[df_meta_train.index]
test_targets = df_cite_targets.loc[df_meta_test.index]

#Grids to search within to optimise random forest
estimator_grid = [0]*1
feature_grid = [0]*1
for i in range(0,1):
    estimator_grid[i] = 200*i + 200

for i in range(0,1):
    feature_grid[i] = 0.2*i +0.2

print(estimator_grid)
print(feature_grid)


#train random forest
forest = RandomForestRegressor(random_state=10, min_samples_leaf=10, n_jobs=1)
#forest.fit(train_cite_inputs_data[0:56000, :], train_cite_targets_data[0:56000, :])
distributions = dict(n_estimators=estimator_grid, max_features=feature_grid)



#clf = RandomizedSearchCV(forest, distributions, random_state=10)
clf = HalvingGridSearchCV(forest, distributions, resource='n_samples', random_state=10)



search = clf.fit(train_inputs, train_targets, groups=df_meta_train.day)

search.best_params_