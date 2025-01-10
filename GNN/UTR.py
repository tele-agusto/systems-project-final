from Bio import SeqIO
import pandas as pd
import numpy as np

def return_h5_path(file):
    path = f'/content/systems-project/GNN/{file}'
    return path

seq_dict = {rec.id : rec.seq for rec in SeqIO.parse(return_h5_path("mart_export.txt"), "fasta")}
full_matching_names = pd.read_csv(return_h5_path('CITEseq22_112_prot_RNA_pairs_13gr.csv'))

matching_names = []
for i in range(112):
    matching_names += [(full_matching_names['protein'][i], full_matching_names['RNA'][i].split('(')[1].strip(')'))]

matching_names = pd.DataFrame(matching_names, columns=['Protein', 'Gene'])

utrs = []
for key in seq_dict.keys():
    utrs += [(key.split('|')[0],str(seq_dict[key]))]

utrs = pd.DataFrame(utrs, columns=['gene', 'UTR'])


Genes = []
for gene in utrs.gene:
    for Gene in matching_names.Gene:
        if gene == Gene.split('_')[0]:
            Genes.append(Gene)

#delete higher index first
del Genes[79]
del Genes[78]
# del Genes[95]
# del Genes[94]
# print(Genes[70:74])
# del Genes[72]
# del Genes[71]
# print(Genes[70:74])

utrs['Full_gene'] = Genes

Proteins = []
for gene in utrs.Full_gene:
    for rna in full_matching_names.RNA:
        if gene.upper() == rna.split(' ')[2].strip('(').strip(')').upper():
            Proteins.append(rna.split(' ')[0])
Proteins = list(dict.fromkeys(Proteins))
del Proteins[79]
del Proteins[78]
# del Proteins[95]
# del Proteins[94]
# print(Proteins[70:74])
# del Proteins[72]
# del Proteins[71]
# print(Proteins[70:74])
# print(Proteins)

utrs['Protein'] = Proteins

Gene_protein_name =[]
for i in range(len(utrs)):
    Gene_protein_name.append(utrs.Full_gene[i].split('_')[1])

utrs['Gene_protein_name'] = Gene_protein_name

utrs_sorted = utrs.sort_values('Gene_protein_name')
# utrs_sorted = utrs
## for 5' utr
utrs_sorted['UTR'][76] = 'A'
utrs_sorted['UTR'][80] = 'A'


def one_hot_encode(seq):
    mapping = dict(zip("ACGT", range(4)))
    seq2 = [mapping[i] for i in seq]
    return np.eye(4)[seq2]


convolutions = np.array([0]*12100).reshape(110,110)
#the max convolution for each base might come at a different index so cannot really sum them, need to find index at whihc sum is max instead - also need to divide by  length of shorter sequence
#might want to normalise for what you would expect due to random chance??
#second sequence reversed as convolution function automatically reverses one sequence so we want to cancel that out
# multiply by length of smallest sequence to penalise short sequences
for i in range(len(utrs_sorted.UTR)):
    for j in range(len(utrs_sorted.UTR)):
        length = min(len(utrs_sorted.UTR[utrs_sorted.index[i]]), len(utrs_sorted.UTR[utrs_sorted.index[j]]))
        diff = abs(len(utrs_sorted.UTR[utrs_sorted.index[i]]) - len(utrs_sorted.UTR[utrs_sorted.index[j]]))
        convolutions_A = np.convolve(one_hot_encode(utrs_sorted['UTR'][utrs_sorted.index[i]])[:,0], one_hot_encode(utrs_sorted['UTR'][utrs_sorted.index[j]])[::-1][:,0])
        convolutions_C = np.convolve(one_hot_encode(utrs_sorted['UTR'][utrs_sorted.index[i]])[:,1], one_hot_encode(utrs_sorted['UTR'][utrs_sorted.index[j]])[::-1][:,1])
        convolutions_G = np.convolve(one_hot_encode(utrs_sorted['UTR'][utrs_sorted.index[i]])[:,2], one_hot_encode(utrs_sorted['UTR'][utrs_sorted.index[j]])[::-1][:,2])
        convolutions_T = np.convolve(one_hot_encode(utrs_sorted['UTR'][utrs_sorted.index[i]])[:,3], one_hot_encode(utrs_sorted['UTR'][utrs_sorted.index[j]])[::-1][:,3])

        # change convolutions calculation
        convolutions[i,j] = (((max(convolutions_G + convolutions_T + convolutions_C + convolutions_A)*1.0)/length)/0.0625) - diff/10
        # convolutions[i,j] = max(convolutions_G + convolutions_T + convolutions_C + convolutions_A)
print(sum(convolutions))
# 5' utr normal order
for i in range(len(convolutions)):
    convolutions[63, i] = 0
    convolutions[64,i] = 0
    convolutions[i,63] = 0
    convolutions[i,64] = 0
convolutions[63,63] = 16
convolutions[64,64] = 16

# 5' utr shuffled order
# for i in range(len(convolutions)):
#     convolutions[76, i] = 0
#     convolutions[80,i] = 0
#     convolutions[i,76] = 0
#     convolutions[i,80] = 0
# convolutions[76,76] = 16
# convolutions[80,80] = 16

adjacency = np.array([0]*12100).reshape(110,110)

#change threshhold
# for i in range(len(convolutions)):
#     for j in range(len(convolutions)):
#         if convolutions[i,j] > 5:
#                 adjacency[i,j] = 1

for i in range(len(convolutions)):
  for j in range(len(convolutions)):
    if convolutions[i,j] > sorted(convolutions[i,:],reverse=True)[4]:
      adjacency[i,j] = 1
      adjacency[j,i] = 1

