import re
import numpy as np
import scipy.io as sio
from .mlc_liblinear import *


def save_results(results, file):
    f = open(file, 'a')
    N, D = results.shape[0], results.shape[1]
    for i in range(N):
        for n in range(D):
            f.write('{:.4f}\t'.format(results[i,n]))
            if n==(D-1): f.write('\n')
    f.close()

def repTxtToMat(file, adjMat):
    repFile = open(file, 'r')
    line = repFile.readline()
    line = line.strip('\n')
    line = re.split('\t| ',line)
    node_num, dim = int(line[0]), int(line[1])

    rep = np.zeros((node_num+1, dim))

    line = repFile.readline()
    while line:
        line = line.strip('/n')
        line = re.split('\t| ',line)
        nodeID = int(line[0])
        for i in range(dim):
            rep[nodeID-1, i] = float(line[i+1])
        line = repFile.readline()

    repFile.close()
    sio.savemat(adjMat, {'rep':rep})
    # print 'repTxtToMat-Done!'

def formatLIBLINEAR(group, rep, trainOcc, trainFile, testFile):

    def formatOneNode(arr_label, arr_rep):
        pos = np.nonzero(arr_label)[0]
        label = ''
        for i in range(pos.shape[0]):
            if i==0:
                label = str(pos[i])
            else:
                label = label + ',' + str(pos[i])
        rep = ''
        for i in range(arr_rep.shape[0]):
            rep = rep + ' ' + str(i+1) + ':' + str(arr_rep[i])
        return label + ' ' + rep + '\n'

    train_id = open(trainFile, 'w')
    test_id = open(testFile, 'w')

    N = rep.shape[0]
    Ntr = int(N*trainOcc)
    IDX = np.random.permutation(N)

    for i in range(N):
        if i<=Ntr:
            node_i = formatOneNode(group[IDX[i]], rep[IDX[i]])
            train_id.write(node_i)
        else:
            node_i = formatOneNode(group[IDX[i]], rep[IDX[i]])
            test_id.write(node_i)
    train_id.close()
    test_id.close()
    # print 'formatLINLINEAR-Done!'

def labelSelection(group, rep, k_max):
    '''
    extracting a sub-network from network, with k_max groups info.
    '''
    g_sum = np.sum(group, axis=0)
    g_sum = np.array(g_sum)[0]
    g_pos = []
    for i in range(k_max):
        max_pos = g_sum.argmax()
        g_pos.append(max_pos)
        g_sum[max_pos] = -1
    g_pos = np.array(g_pos)

    sub_g = group[:, g_pos]
    sub_g_sum = np.sum(sub_g, axis=1)
    sub_g_pos = np.nonzero(sub_g_sum)[0]

    sub_g = sub_g[sub_g_pos,:].toarray()  # sparse matrix to array
    sub_rep = rep[sub_g_pos, :]
    return sub_rep, sub_g


def MLC(args):

    group = sio.loadmat(args.input_net)['group']

    # to one-hot vector
    if group.shape[1] == 1:
        group = np.reshape(group, (-1))
        n_group = np.max(group) + 1
        group = np.eye(n_group)[group]

    rep = sio.loadmat(args.rep)['rep']

    if args.k_max>1000:
        sub_rep = rep
        sub_g = group
    else:
        sub_rep, sub_g = labelSelection(group, rep, args.k_max)

    # Occ = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    Occ = [0.1, 0.3, 0.5, 0.7, 0.9]

    microF1 = np.zeros((10, len(Occ)))
    macroF1 = np.zeros((10, len(Occ)))

    for v in range(10):
        for i in range(len(Occ)):
            trainOcc = Occ[i]
            formatLIBLINEAR(sub_g, sub_rep, trainOcc, args.trainFile, args.testFile)
            result = mlc_liblinear(args.trainFile, args.testFile, args.path, args.dataset)
            microF1[v, i] = result[1]
            macroF1[v, i] = result[2]

    microF1 = np.sum(microF1, axis=0)/10*100
    macroF1 = np.sum(macroF1, axis=0)/10*100
    results = np.zeros((2, len(Occ)))
    results[0,:] = microF1
    results[1,:] = macroF1

    return results


# def MLC(args):

#     group = sio.loadmat(args.input_net)['group']
#     rep = sio.loadmat(args.rep)['rep']
#     sub_rep, sub_g = labelSelection(group, rep, args.k_max)

#     microF1 = np.zeros((10, 18))
#     macroF1 = np.zeros((10, 18))

#     for v in range(10):
#         Occ = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
#         for i in range(18):
#             trainOcc = Occ[i]
#             formatLIBLINEAR(sub_g, sub_rep, trainOcc, args.trainFile, args.testFile)
#             result = mlc_liblinear(args.trainFile, args.testFile, args.path, args.dataset)
#             microF1[v, i] = result[1]
#             macroF1[v, i] = result[2]

#     microF1 = np.sum(microF1, axis=0)/10*100
#     macroF1 = np.sum(macroF1, axis=0)/10*100
#     results = np.zeros((2, 18))
#     results[0,:] = microF1
#     results[1,:] = macroF1

#     return results


def parse_args():
    '''
    Parses the TripletNE arguments.
    '''
    parser = argparse.ArgumentParser(description="Run TripletNE.")
    parser.add_argument('--input_net', nargs='?', default='input/sDBLP-20.mat', help='Input adjMat')
    parser.add_argument('--rep', nargs='?', default='output/sDBLP-20-rep.mat', help='Embeddings path')
    parser.add_argument('--k_max', type=int, default=5, help='k_max')
    parser.add_argument('--path', nargs='?', default='model_saving/', help='path')
    parser.add_argument('--dataset', nargs='?', default='sDBLP-20', help='dataset')
    parser.add_argument('--trainFile', nargs='?', default='model_saving/sDBLP-20-train.txt', help='trainFile')
    parser.add_argument('--testFile', nargs='?', default='model_saving/sDBLP-20-test.txt', help='testFile')
    parser.add_argument('--resultTxt', nargs='?', default='data/RWTripletNE-sDBLP-20.txt')

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    MLC(args)