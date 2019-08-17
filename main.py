# -*- coding: utf-8 -*-

'''
	Created on Sept. 09, 2018
	Author: Quanyu Dai, dqyzm100@hotmail.com
    The Hong Kong Polytechnic University
'''

'''
adaptive l2-norm constraint for adversarial training
'''

import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1" 
import logging
import numpy as np
import scipy.io as sio
import argparse
import math
import tensorflow as tf
import tqdm
from time import time
from time import strftime
from time import localtime
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

import AdvT4NE
import graph
import utils
import LIBLINEAR.mlc_NE_LIBLINEAR as mlc_NE_LIBLINEAR
import LIBLINEAR.mcc_liblinear as mcc_liblinear
import evaluation.evaluation as evaluation

# global variables
_window_size = None
_walk_length = None
_table = None
_negative = None
_walks = None
_G = None

def parse_args():
	'''
	Parses the advesarial node2vec arguments.
	'''
	parser = argparse.ArgumentParser(description="Running Adversarial node2vec.")
	# for input
	parser.add_argument('--input_net', nargs='?', default='input/wiki.mat', help='Input adjMat')
	parser.add_argument('--dataset', nargs='?', default='wiki', help='Dataset name')
	parser.add_argument('--restore', type=str, default=None, help='The restore time_stamp for weights in Pretrain')
	parser.add_argument('--ckpt', type=int, default=0, help='Save the model per X epochs.')
	parser.add_argument('--base', type=str, default='deepwalk', help='Base model')
	# for random walk
	parser.add_argument('--walk_length', type=int, default=40,help='Length of walk per source.')
	parser.add_argument('--num_walks', type=int, default=1, help='Number of walks per source.')
	parser.add_argument('--window_size', type=int, default=5, help='Context size for optimization.')
	parser.add_argument('--negative', type=int, default=5, help='Number of negative pairs for each positive pair.')
	parser.add_argument('--p', type=float, default=1, help='Return hyperparameter.')
	parser.add_argument('--q', type=float, default=1, help='Inout hyperparameter.')
	parser.add_argument('--weighted', dest='weighted', action='store_true', help='Boolean specifying (un)weighted. Default is unweighted.')
	parser.add_argument('--unweighted', dest='unweighted', action='store_false')
	parser.set_defaults(weighted=False)
	parser.add_argument('--directed', dest='directed', action='store_true', help='Graph is (un)directed. Default is undirected.')
	parser.add_argument('--undirected', dest='undirected', action='store_false')
	parser.set_defaults(directed=False)
	# for LINE
	parser.add_argument('--order', type=int, default=1,help='For graph preprocessing.')
	# for adversarial
	parser.add_argument('--adver', type=int, default=0, help='Adversarial training or not.')
	parser.add_argument('--eps', type=float, default=0.5, help='Epsilon for adversarial weights.')
	parser.add_argument('--reg_adv', type=float, default=1.0, help='Epsilon for adversarial weights.')
	parser.add_argument('--adv', nargs='?', default='grad', help='Generate the adversarial sample by gradient method or random method')
	parser.add_argument('--adapt_l2', type=int, default=1, help='Whether to add the adaptive l2 norm constraint.')
	# for training
	parser.add_argument('--embed_size', type=int, default=128, help='Dimension of embedding vectors.')
	parser.add_argument('--lr', type=float, default=0.001, help='Learning rate.')
	parser.add_argument('--batch_size', type=int, default=100, help='Batch size.')
	parser.add_argument('--nepoch', type=int, default=5, help='Number of epoch.')
	parser.add_argument('--pretraining_nepoch', type=int, default=5, help='Number of epoch for pretraining.')
	parser.add_argument('--normalized', type=int, default=0, help='Normalize the embeddings before evaluation or not.')
	# for testing
	parser.add_argument('--original_graph', nargs='?', default='input/wiki.mat', help='Original network for link prediction.')
	parser.add_argument('--check_link_prediction', nargs='?', default='2,10,100,200,300,500,800,1000,2000,5000,10000')
	parser.add_argument('--check_reconstruction', nargs='?', default='2,10,100,200,300,500,800,1000,2000,5000,10000')

	# for multi-label classification
	parser.add_argument('--trainFile', nargs='?', default='./LIBLINEAR/model_saving/blog-train.txt', help='trainFile')
	parser.add_argument('--testFile', nargs='?', default='./LIBLINEAR/model_saving/blog-test.txt', help='testFile')
	parser.add_argument('--path', nargs='?', default='./LIBLINEAR/model_saving/', help='path')

	# for link prediction testing
	parser.add_argument('--rep_txt', nargs='?', default='../output/cora-rep.txt')
	parser.add_argument('--train_pos_file', nargs='?', default='../AUC/cora-50-train-pos.net')
	parser.add_argument('--train_neg_file', nargs='?', default='../AUC/cora-50-train-neg.net')
	parser.add_argument('--test_pos_file', nargs='?', default='../AUC/cora-50-test-pos.net')
	parser.add_argument('--test_neg_file', nargs='?', default='../AUC/cora-50-test-neg-')
	parser.add_argument('--lp_path', nargs='?', default='/home/wonniu/Embeddings/Link-Prediction')

	parser.add_argument('--resultTxt', nargs='?', default='result/NE-wiki.txt')
	parser.add_argument('--task', type=str, default='mcc', help='lp, mcc, mlc, re, or mcc+re, or mlc+re')
	parser.add_argument('--k_max', type=int, default=5, help='k_max')
	parser.add_argument('--rep', nargs='?', default='output/wiki-rep.mat')

	# for visualization
	parser.add_argument('--vis_label', type=int, default=5, help='Label number for visualization.')

	return parser.parse_args()

def sigmoid(x):
  return 1-1 / (1 + np.exp(-(x-4)))

def batch_feed_dict(model, pos_pairs, neg_pairs, index_pos, negative, ppmi):
	index_neg = []
	for i in list(index_pos):
		index_neg.extend(range(i*negative, (i+1)*negative))

	weights = 1 - ppmi[pos_pairs[index_pos, 0], pos_pairs[index_pos, 1]]/np.max(ppmi)
	# weights = sigmoid(ppmi[pos_pairs[index_pos, 0], pos_pairs[index_pos, 1]])

	# print(weights.shape)

	feed_dict = {model.target_P: np.reshape(pos_pairs[index_pos, 0], (-1)), 
				 model.positive: np.reshape(pos_pairs[index_pos, 1], (-1)),
				 model.target_N: np.reshape(neg_pairs[index_neg, 0], (-1)),
				 model.negative: np.reshape(neg_pairs[index_neg, 1], (-1)),
				 model.weights: np.reshape(weights, (-1)),
				 model.batch_size: index_pos.shape[0]}
	return feed_dict

def batch_feed_dict_LINE(model, pos_pairs, neg_pairs, ppmi):

	weights = 1 - ppmi[pos_pairs[:, 0], pos_pairs[:, 1]]/np.max(ppmi)

	feed_dict = {model.target_P: np.reshape(pos_pairs[:, 0], (-1)), 
				 model.positive: np.reshape(pos_pairs[:, 1], (-1)),
				 model.target_N: np.reshape(neg_pairs[:, 0], (-1)),
				 model.negative: np.reshape(neg_pairs[:, 1], (-1)),
				 model.weights: np.reshape(weights, (-1)),
				 model.batch_size: pos_pairs.shape[0]}
	return feed_dict

def print_info(results):
    for i in range(results.shape[0]):
        info = 'AFTER COMPRESSION |'
        for j in range(results.shape[1]):
            info = info + ' {:.4f} |'.format(results[i, j])
        print(info)

def training_deepwalk(args, G, ppmi, model, sess, epoch_start, epoch_end, saver_ckpt, ckpt_save_path):
	#############################
	# recording best results
	best_epoch = 0
	best_embedding = None
	score = 0
	#############################

	#######################################################
	# walking
	walks = G.simulate_walks(args.num_walks, args.walk_length)
	print('walks:', walks.shape)
	pos_pairs, neg_pairs = utils.walks_to_pairs_with_ns_pool(walks, args.window_size, args.negative, G.table)
	print('pos_pairs:', pos_pairs.shape)
	print('neg_pairs:', neg_pairs.shape)
	pair_num = pos_pairs.shape[0]
	iters = int(pair_num/args.batch_size)
	#######################################################

	print ('Begin training...')
	np.random.seed()
	for epoch in range(epoch_start+1, epoch_end+1):

		epoch_begin_time = time()

		if epoch>epoch_start:
			walks = G.simulate_walks(args.num_walks, args.walk_length)
			pos_pairs, neg_pairs = utils.walks_to_pairs_with_ns_pool(walks, args.window_size, args.negative, G.table)

		random_seq = np.random.permutation(pair_num)
		# for j in tqdm.tqdm(range(iters)):
		for j in range(iters):
			if j!=iters-1:
				index_pos = random_seq[j*args.batch_size:(j+1)*args.batch_size]
			else:
				index_pos = random_seq[j*args.batch_size:]

			feed_dict = batch_feed_dict(model, pos_pairs, neg_pairs, index_pos, args.negative, ppmi)
			if args.adver:
				sess.run([model.update_T, model.update_C], feed_dict)
			sess.run(model.optimizer, feed_dict)

		print("Epoch training time [%.1f s]" % (time() - epoch_begin_time))
		eval_begin_time = time()

		results = evaluation.evaluate(model, sess, args, epoch)

		###########################################################
		# recording best results
		if np.sum(results)>score:
			best_epoch = epoch
			score = np.sum(results)

			if args.normalized:
			    best_embedding = sess.run(model.get_normalized_embeddings())
			else:
			    best_embedding = sess.run(model.embedding_T)
		###########################################################

		print("Evaluation [%.1f s]" % (time() - eval_begin_time))

		if args.ckpt>0 and (epoch-epoch_start)%args.ckpt==0:
			saver_ckpt.save(sess, ckpt_save_path+'weights', global_step=epoch)

		if (not args.adver) and (epoch)==(args.pretraining_nepoch+epoch_start):
			saver_ckpt.save(sess, ckpt_save_path+'weights', global_step=epoch)

	return best_epoch, best_embedding


def training_LINE(args, G, ppmi, model, sess, epoch_start, epoch_end, saver_ckpt, ckpt_save_path):
	#############################
	# recording best results
	best_epoch = 0
	best_embedding = None
	score = 0
	#############################

	batch_generator = G.edge_batch
	iterations = int(G.edge_num/args.batch_size) + 1

	print ('Begin training...')
	for epoch in range(epoch_start+1, epoch_end+1):

		epoch_begin_time = time()

		for j in range(iterations):
			pos_pairs, neg_pairs = next(batch_generator)
			feed_dict = batch_feed_dict_LINE(model, pos_pairs, neg_pairs, ppmi)

			if args.adver:
				if args.base=='deepwalk' or args.base=='node2vec' or args.base=='LINE_2':
					sess.run([model.update_T, model.update_C], feed_dict)
				elif args.base=='LINE_1':
					sess.run([model.update_T], feed_dict)
			sess.run(model.optimizer, feed_dict)

		epoch_training_time = time() - epoch_begin_time

		if (epoch-epoch_start-1)%1==0:
			eval_begin_time = time()
			results = evaluation.evaluate(model, sess, args, epoch)
			print("Epoch [%d] - Training [%.1f s] Evaluation [%.1f s]" % (epoch, epoch_training_time, time() - eval_begin_time))

		###########################################################
		# recording best results
		if np.sum(results)>score:
			best_epoch = epoch
			score = np.sum(results)

			if args.normalized:
			    best_embedding = sess.run(model.get_normalized_embeddings())
			else:
			    best_embedding = sess.run(model.embedding_T)
		###########################################################

		if args.ckpt>0 and (epoch-epoch_start)%args.ckpt==0:
			saver_ckpt.save(sess, ckpt_save_path+'weights', global_step=epoch)

		if (not args.adver) and (epoch)==(args.pretraining_nepoch+epoch_start):
			saver_ckpt.save(sess, ckpt_save_path+'weights', global_step=epoch)

	return best_epoch, best_embedding

def training(model, G, args, epoch_start, epoch_end, time_stamp):

	# logging and loading
	if args.adver:
		ckpt_save_path = 'Pretrain/{}/{}_adv/embed_{}/{}/'.format(args.dataset, args.base, args.embed_size, time_stamp)
		ckpt_restore_path = 'Pretrain/{}/{}/embed_{}/{}/'.format(args.dataset, args.base, args.embed_size, time_stamp)
	else:
		ckpt_save_path = 'Pretrain/{}/{}/embed_{}/{}/'.format(args.dataset, args.base, args.embed_size, time_stamp)
		ckpt_restore_path = 0 if args.restore is None else 'Pretrain/{}/{}/embed_{}/{}/'.format(args.dataset, args.base, args.embed_size, args.restore)

	if not os.path.exists(ckpt_save_path):
		os.makedirs(ckpt_save_path)
	if ckpt_restore_path and not os.path.exists(ckpt_restore_path):
		os.makedirs(ckpt_restore_path)

	saver_ckpt = tf.train.Saver({'embedding_T': model.embedding_T,
								 'embedding_C': model.embedding_C,
								 'context_bias': model.context_bias})

	# initialization
	init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
	sess = tf.InteractiveSession()
	sess.run(init)

	# restore
	if args.restore is not None or epoch_start:
		ckpt = tf.train.get_checkpoint_state(os.path.dirname(ckpt_restore_path + 'checkpoint'))
		print(ckpt.model_checkpoint_path)
		if ckpt and ckpt.model_checkpoint_path:
			saver_ckpt.restore(sess, ckpt.model_checkpoint_path)
			print('================Done loading======================')			
	else:
		logging.info('Initialized from scratch')
		print('Initialized from scratch')

	evaluation.evaluate(model, sess, args, epoch_start)

	###############################################################
	# adaptive l2 norm based on node similarities from PPMI matrix
	A = sio.loadmat(args.input_net)['network']
	PPMI = utils.PPMI(A, k=2, flag=False)
	###############################################################

	#######################################################
	# deepwalk, node2vec, or LINE
	if args.base=='deepwalk' or args.base=='node2vec':
		best_epoch, best_embedding = training_deepwalk(args, G, PPMI, model, sess, epoch_start, epoch_end, saver_ckpt, ckpt_save_path)
	elif args.base=='LINE_1' or args.base=='LINE_2':
		best_epoch, best_embedding = training_LINE(args, G, PPMI, model, sess, epoch_start, epoch_end, saver_ckpt, ckpt_save_path)
	#######################################################

	# saver_ckpt.save(sess, ckpt_save_path+'weights', global_step=epoch)
	print('Finish training.')

	#######################################################
	if args.adver:
		sio.savemat('./output/{}-{}-adv-vis.mat'.format(args.dataset, args.base), {'rep': best_embedding})
	else:
		sio.savemat('./output/{}-{}-vis.mat'.format(args.dataset, args.base), {'rep': best_embedding})
	print('------------------------------------------')
	print('Best Epoch: {}'.format(best_epoch))
	print('------------------------------------------')
	evaluation.print_settings(args, flag='best_epoch', best_epoch=best_epoch)
	#######################################################

def init_logging(args, time_stamp):
    path = "Log/%s_%s/" % (strftime('%Y-%m-%d_%H', localtime()), args.task)
    if not os.path.exists(path):
        os.makedirs(path)
    logging.basicConfig(filename=path + "%s_log_embed_size%d_%s" % (args.dataset, args.embed_size, time_stamp),
                        level=logging.INFO)
    logging.info(args)
    print(args)

def main(args):
	time_stamp = strftime('%Y_%m_%d_%H_%M_%S', localtime())
	init_logging(args, time_stamp)

	if args.base=='deepwalk' or args.base=='node2vec':
		if args.task=='lp':
			G = graph.Graph(args.input_net, args.directed, args.weighted, args.p, args.q, flag=False)
		else:
			G = graph.Graph(args.input_net, args.directed, args.weighted, args.p, args.q, flag=True)
	elif args.base=='LINE_1' or args.base=='LINE_2':
		G = graph.Graph_LINE(args.input_net, args.directed, args.weighted, base='LINE_2', batch_size=1024, K=1, order=args.order)

	evaluation.print_settings(args, 'settings')

	args.adver = 0
	args.rep = 'output/{}-rep-{}.mat'.format(args.dataset, args.base)
	n2v = AdvT4NE.AdvT4NE(G.node_num, args.embed_size, args.negative, args.adv, args.adver, \
									  args.reg_adv, args.lr, args.eps, args.adapt_l2, args.base)
	print('Initialize {}'.format(args.base))
	# training(n2v, G, args, 0, args.nepoch, time_stamp)
	training(n2v, G, args, 0, args.pretraining_nepoch, time_stamp)

	evaluation.print_settings(args, 'breakline')

	tf.reset_default_graph()
	
	args.adver = 1
	args.rep = 'output/{}-rep-{}-adv.mat'.format(args.dataset, args.base)
	n2v_adv = AdvT4NE.AdvT4NE(G.node_num, args.embed_size, args.negative, args.adv, args.adver, \
										  args.reg_adv, args.lr, args.eps, args.adapt_l2, args.base)
	print('Initizlize adversarial {}'.format(args.base))
	training(n2v_adv, G, args, args.pretraining_nepoch, args.nepoch, time_stamp)

if __name__ == "__main__":
	args = parse_args()
	main(args)