import scipy
import scipy.io as sio
import numpy as np
import argparse
import time
from .liblinearutil import *


def load_data(file):
    # load data
    net_dict = sio.loadmat(file)
    network = net_dict['network']
    network = network.toarray()
    label = net_dict['group']
    label = np.transpose(label)
    label = label[0]
    return network, label

def save_results(epoch, results, file):
	f = open(file, 'a')
	N, D = results.shape[0], results.shape[1]
	for i in range(N):
		f.write('{:3d}\t'.format(epoch))
		for n in range(D):
			f.write('{:.4f}\t'.format(results[i,n]))
			if n==(D-1): f.write('\n')
	f.close()


def mcc_liblinear_one_file(args):

	netFile = args.input_net
	resultTxt = args.resultTxt
	repFile = args.rep

	network, labels = load_data(netFile)
	N = network.shape[0]
	results = np.zeros((1, 18))

	rep = sio.loadmat(repFile)
	rep = rep['rep']
	tmp = np.zeros((10, 18))

	ratio = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 60, 70, 80, 90]

	np.random.seed(960126)
	for i in range(18):
		for j in range(10):
			Ntr = int((float(ratio[i]))/100.0*N)

			IDX = np.random.permutation(N)
			IDXtr = IDX[0:Ntr]
			IDXts = IDX[Ntr:]

			Xtr = scipy.sparse.csr_matrix(rep[IDXtr])
			Xts = scipy.sparse.csr_matrix(rep[IDXts])

			Ytr = scipy.asarray(labels[IDXtr])
			Yts = scipy.asarray(labels[IDXts])
			# cmd = '-s 2 -c {} -q'.format(0.1)
			# cmd = '-s 2 -c {} -q'.format((i+1.0)/200)
			cmd = '-s 2 -c 1 -q'
			m = train(Ytr, Xtr, cmd)
			p_label, p_acc, p_val = predict(Yts, Xts, m)   # accuracy in p_acc[0]
			tmp[j,i] = p_acc[0]

	results[0,:] = np.sum(tmp, axis=0)/10
	return results

def mcc_liblinear_for_vis(rep, labels):

	N = rep.shape[0]
	results = np.zeros((1, 9))
	tmp = np.zeros((10, 9))

	for i in range(9):
		for j in range(10):
			Ntr = int((i+1.0)/10*N)

			IDX = np.random.permutation(N)
			IDXtr = IDX[0:Ntr]
			IDXts = IDX[Ntr:]

			Xtr = scipy.sparse.csr_matrix(rep[IDXtr])
			Xts = scipy.sparse.csr_matrix(rep[IDXts])

			Ytr = scipy.asarray(labels[IDXtr])
			Yts = scipy.asarray(labels[IDXts])
			# cmd = '-s 2 -c {} -q'.format(0.1)
			# cmd = '-s 2 -c {} -q'.format((i+1.0)/200)
			cmd = '-s 2 -c 1 -q'
			m = train(Ytr, Xtr, cmd)
			p_label, p_acc, p_val = predict(Yts, Xts, m)   # accuracy in p_acc[0]
			tmp[j,i] = p_acc[0]

	results[0,:] = np.sum(tmp, axis=0)/10
	return results
	
def mcc_liblinear(args):

	netFile = args.input_Mat
	resultTxt = args.resultTxt
	repFile = args.rep

	network, labels = load_data(netFile)
	N = network.shape[0]
	results = np.zeros((20, 2))

	occ = [0, 8]
	for n in range(20):

		repFile = repFile.format(n+1)
		rep = sio.loadmat(repFile)
		rep = rep['rep']
		tmp = np.zeros((10, 2))

		count = 0
		for i in occ:
			for j in range(10):
				Ntr = int((i+1.0)/10*N)

				IDX = np.random.permutation(N)
				IDXtr = IDX[0:Ntr]
				IDXts = IDX[Ntr:]

				Xtr = scipy.sparse.csr_matrix(rep[IDXtr])
				Xts = scipy.sparse.csr_matrix(rep[IDXts])

				Ytr = scipy.asarray(labels[IDXtr])
				Yts = scipy.asarray(labels[IDXts])

				m = train(Ytr, Xtr, '-s 2 -c 1 -q')
				p_label, p_acc, p_val = predict(Yts, Xts, m)   # accuracy in p_acc[0]
				tmp[j,count] = p_acc[0]
			count = count + 1

		results[n,:] = np.sum(tmp, axis=0)/10
	save_results(results, resultTxt)

def parse_args():
	parser = argparse.ArgumentParser(description="Run mcc_liblinear.")
	parser.add_argument('--input_net', nargs='?', default='input/cora-undirected.mat', help='Input adjMat')
	parser.add_argument('--rep', nargs='?', default='input/cora-rep-{}.mat', help='Input embeddings')
	parser.add_argument('--resultTxt', nargs='?', default='data/DWGAN-citeseer.txt')

	return parser.parse_args()

def main(args):
	localtime = time.asctime(time.localtime(time.time()))
	print ('Begining Time :', localtime)

	mcc_liblinear(args)

	localtime = time.asctime(time.localtime(time.time()))
	print ('Endding Time :', localtime)

if __name__ == "__main__":
	args = parse_args()
	main(args)


