from xclib.data import data_utils
import numpy as np
import os
from scipy.sparse import csr_matrix
from scipy.sparse import csc_matrix
from tqdm import tqdm
import time
import sys
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import pairwise_kernels

def findTrainNN(Y, X, k=1,batch_size = 1024):
    X = normalize(X, norm='l2',axis=1)
    Y = normalize(Y, norm='l2',axis=1) 
    num_examples = Y.shape[0]
    idxs = np.zeros((num_examples, k))
    scores = np.zeros((num_examples, k))
    batch_num = int(num_examples/batch_size) + 1
    for batch_iter in tqdm(range(batch_num)):
        if batch_iter == batch_num - 1:
            left = num_examples - batch_iter * batch_size
        else:
            left = batch_size
        start_point = batch_iter*batch_size
        Y_batch = Y[start_point:start_point+left, :]
        cos_sim = pairwise_kernels(Y_batch, X, metric='linear', n_jobs=-1)
        topK_idxs = np.argpartition(-cos_sim,range(k+1),axis = 1)[:,1:k+1]
        for i in range(left):
            scores[start_point+i,:] = cos_sim[i,topK_idxs[i]]
        idxs[start_point:start_point+left,:] = topK_idxs
    return idxs

def convert_lbl(labels):
    num_labels = labels.shape[1]
    label_wise_save = [ [] for _ in range(num_labels) ]
    rows, cols = labels.nonzero()
    for i in range(len(rows)):
        row, col = rows[i], cols[i]
        label_wise_save[col].append(row)
    return label_wise_save


def computeNNPP(nn_idxs, train_lbl, up_bound):
    label_wise_save = convert_lbl(train_lbl)
    nn_percents = []
    label_counts = []
    num_labels = len(label_wise_save)
    for i in tqdm(range(num_labels)):
        idxs = label_wise_save[i]
        nn_pos_count = 0
        label_counts.append(len(idxs))
        if len(idxs) > 0 and len(idxs) < up_bound:
            for idx in idxs:
                nearest_neigh = nn_idxs[idx]
                if (nearest_neigh in label_wise_save[i]):
                    nn_pos_count += 1
            nn_percents.append(nn_pos_count/len(idxs))
        else:
            if len(idxs) == 0:
                nn_percents.append(0)
            else:
                nn_percents.append(-1)
    return nn_percents, label_counts

def get_cut_idx(NN_percent, label_counts, cut_rate, tail_bound):
    NN_percent = np.array(NN_percent)
    label_counts = np.array(label_counts)
    cut_idx = np.logical_and((NN_percent < cut_rate),(label_counts < tail_bound))
    cut_count = np.sum(cut_idx)
    cut_idx = np.nonzero(cut_idx)[0]
    return cut_idx

def cut_lbl(labels, cut_idx):
    labels = labels.tocsc()
    mask = np.ones(labels.shape[1], dtype=bool)
    mask[cut_idx] = False
    labels_d = labels[:, mask]
    return labels_d


train_X_path = sys.argv[1]
train_lbl_path = sys.argv[2]
test_lbl_path = sys.argv[3]
save_trn_lbl = sys.argv[4]
save_tst_lbl = sys.argv[5]
save_cut_idx = sys.argv[6]
tail_bound = int(sys.argv[7])
cut_rate = float(sys.argv[8])

t1 = time.time()
print('Loading data')
train_X = data_utils.read_sparse_file(train_X_path)
train_lbl = data_utils.read_sparse_file(train_lbl_path).astype(int)
test_lbl = data_utils.read_sparse_file(test_lbl_path).astype(int)

print('Finding NN')
NN_idx = findTrainNN(train_X, train_X)

print('Computing NNPP')
nnpp, label_counts = computeNNPP(NN_idx, train_lbl, tail_bound)

print('Cutting Tail label')
cut_idx = get_cut_idx(nnpp, label_counts, cut_rate, tail_bound)
train_lbl_d = cut_lbl(train_lbl, cut_idx)
test_lbl_d = cut_lbl(test_lbl, cut_idx)

print('Saving data')
np.savetxt(save_cut_idx,cut_idx, fmt='%d')
data_utils.write_sparse_file(train_lbl_d,save_trn_lbl)
data_utils.write_sparse_file(test_lbl_d,save_tst_lbl)

t2 = time.time()
num_labels = train_lbl.shape[1]
num_labels_d = train_lbl_d.shape[1]
report="Labels Reduced from %d to %d\nCost time %ds\n"%(num_labels, num_labels_d,int(t2-t1))
print(report)






