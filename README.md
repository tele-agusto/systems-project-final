# systems-project

Comparing a random forest baseline to GNN models for predicting cell surface protein abundance in single cells from mRNA expression data.

Three different GNNs were used to connect genes in the network based on sequence similarity in the promoter, 5' UTR or 3' UTR. The perfomrance of each GNN was compared with an identical GNN with a shuffled adjacency matrix to determine if they had captured useful relationships. Only the promoter GNN outperformed its shuffled version indicating it was the strongest of the three GNNs.

The random forest had the best overall performance and predicted genes with high mRNA expression to protein abundance correlation well. However, when this correlation was weak, the promoter GNN outperformed the random forest - indicating the network structure of the model had captured useful relationships. 
