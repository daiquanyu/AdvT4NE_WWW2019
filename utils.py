# -*- coding: utf-8 -*-
import os
import math
import numpy as np
import scipy.sparse as sp
import scipy.io as sio
from collections import defaultdict as dd
import random
import networkx as nx
import tqdm
from time import time

from multiprocessing import Pool
from multiprocessing import cpu_count

def load_data(file):
    # load data
    net = sio.loadmat(file)
    net = net['network']
    return net

def read_graph(adj, directed, weighted):
    '''
    sparse adj to networkx graph
    '''
    if weighted:
        # node id starts from 0
        G = nx.from_scipy_sparse_matrix(adj, create_using=nx.DiGraph())
    else:
        G = nx.from_scipy_sparse_matrix(adj, create_using=nx.DiGraph())
        for edge in G.edges():
            G[edge[0]][edge[1]]['weight'] = 1

    if not directed:
        G = G.to_undirected()

    return G

def walks_to_pairs_with_ns_pool(walks, window_size, negative, table):
    '''
    parallelize node pair generation process with given walks
    '''
    begin_time = time()
    global _window_size
    global _negative
    global _table
    global _walks
    global _walk_length
    _window_size = window_size
    _negative = negative
    _table = table
    _walks = walks
    _walk_length = walks.shape[1]

    pool = Pool(cpu_count())
    pairs = pool.map(_walk_to_pairs, range(walks.shape[0]))
    pool.close()
    pool.join()

    pos_pairs, neg_pairs = [], []
    for i in range(len(pairs)):
        pos_pairs.append(pairs[i][0])
        neg_pairs.append(pairs[i][1])

    print("walks_to_pairs_with_ns_pool [%.1f s]" % (time() - begin_time))
    return np.concatenate(pos_pairs, axis=0), np.concatenate(neg_pairs, axis=0)

def _walk_to_pairs(n):
    pos_pairs, neg_pairs = [], []

    walk = _walks[n, :]
    for l in range(_walk_length):
        for m in range(l-_window_size, l+_window_size+1):
            if m<0 or m>=_walk_length: continue
            pos_pairs.append([walk[l], walk[m]])
            for k in range(_negative):
                n_neg = _table.sample(1)[0]
                while n_neg==walk[l] or n_neg==walk[m]:
                    n_neg = _table.sample(1)[0]
                neg_pairs.append([walk[l], n_neg])

    return np.array(pos_pairs, dtype=np.int32), np.array(neg_pairs, dtype=np.int32)


##################
# PPMI Matrix
##################

def PPMI(A, k=4, flag=True): 
    N = A.shape[0]

    # symmetric normalization
    D = sp.diags(1./np.sqrt(np.array(np.sum(A, axis=1)).reshape((1, N))), [0])
    A = D * A * D

    As = []
    tmp = A
    for i in range(k-1):
        tmp = A * tmp
        As.append(tmp)
    for i in range(k-1):
        A = A + As[i]
    A = A/k
    
    D = sp.diags(1./np.sqrt(np.array(np.sum(A, axis=1)).reshape((1, N))), [0])
    A = D * A * D
    
    if flag:
        # delete zero elements from the sparse matrix
        A.data = np.log(A.data) - np.reshape(np.array([math.log(1.0/N)] * A.data.shape[0]), (A.data.shape[0], ))
        A.data = np.maximum(A.data, 0)
    
    # A = sp.csc.csc_matrix(A)
    return A.toarray()

