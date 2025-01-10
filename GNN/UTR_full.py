from Bio import SeqIO
import pandas as pd
import numpy as np
import scipy.signal.convolve

seq_dict = {rec.id : rec.seq for rec in SeqIO.parse("/content/drive/MyDrive/martquery_geneutrscanonical.txt", "fasta")}
df_cite_inputs = pd.read_hdf(return_h5_path('train_cite_inputs.h5'))

gutrs = []
for key in seq_dict.keys():
    gutrs += [(key.split('|')[0],str(seq_dict[key]))]

gutrs = pd.DataFrame(gutrs, columns=['gene', 'UTR'])

gene_names = list(df_cite_inputs.columns)

Genes = []
for gene in gutrs.gene:
    for Gene in gene_names:
        if gene == Gene.split('_')[0]:
            Genes.append(Gene)

gutrs['Full_gene'] = Genes
Gene_protein_name =[]
for i in range(len(gutrs)):
    Gene_protein_name.append(gutrs.Full_gene[i].split('_')[1])

gutrs['Gene_protein_name'] = Gene_protein_name
gutrs_sorted = gutrs.sort_values('Gene_protein_name')
gutrs_sorted = gutrs_sorted.set_index('UTR')
gutrs_sorted = gutrs_sorted.drop('Sequenceunavailable')
gutrs_sorted = gutrs_sorted.reset_index()
gutrs_sorted

def one_hot_encode(seq):
    mapping = dict(zip("ACGT", range(4)))
    seq2 = [mapping[i] for i in seq]
    return np.eye(4)[seq2]

convolutions = np.array([0]*206985769).reshape(14387,14387)
#the max convolution for each base might come at a different index so cannot really sum them, need to find index at whihc sum is max instead - also need to divide by  length of shorter sequence
#might want to normalise for what you would expect due to random chance??
#second sequence reversed as convolution function automatically reverses one sequence so we want to cancel that out
# multiply by length of smallest sequence to penalise short sequences
for i in range(len(gutrs_sorted.UTR)):
    for j in range(len(gutrs_sorted.UTR)):
        length = min(len(gutrs_sorted.UTR[gutrs_sorted.index[i]]), len(gutrs_sorted.UTR[gutrs_sorted.index[j]]))
        diff = abs(len(gutrs_sorted.UTR[gutrs_sorted.index[i]]) - len(gutrs_sorted.UTR[gutrs_sorted.index[j]]))
        convolutions_A = scipy.signal.convolve(one_hot_encode(gutrs_sorted['UTR'][gutrs_sorted.index[i]])[:,0], one_hot_encode(gutrs_sorted['UTR'][gutrs_sorted.index[j]])[::-1][:,0])
        convolutions_C = scipy.signal.convolve(one_hot_encode(gutrs_sorted['UTR'][gutrs_sorted.index[i]])[:,1], one_hot_encode(gutrs_sorted['UTR'][gutrs_sorted.index[j]])[::-1][:,1])
        convolutions_G = scipy.signal.convolve(one_hot_encode(gutrs_sorted['UTR'][gutrs_sorted.index[i]])[:,2], one_hot_encode(gutrs_sorted['UTR'][gutrs_sorted.index[j]])[::-1][:,2])
        convolutions_T = scipy.signal.convolve(one_hot_encode(gutrs_sorted['UTR'][gutrs_sorted.index[i]])[:,3], one_hot_encode(gutrs_sorted['UTR'][gutrs_sorted.index[j]])[::-1][:,3])

        convolutions[i,j] = (((max(convolutions_G + convolutions_T + convolutions_C + convolutions_A)*1.0)/length)/0.0625) - diff/10

adjacency = np.array([0]*206985769).reshape(14387,14387)

for i in range(len(convolutions)):
    for j in range(len(convolutions)):
        if convolutions[i,j] > 4:
                adjacency[i,j] = 1
