# -*- coding: utf-8 -*-

'''
	Created on Sept. 09, 2018
	Author: Quanyu Dai, quanyu.dai@connect.polyu.hk
    The Hong Kong Polytechnic University
'''

'''
adaptive l2-norm constraint for adversarial training
'''

import os
import logging
import numpy as np
import scipy.io as sio
import argparse
import math
import tensorflow as tf


class AdvT4NE:
	def __init__(self, node_num, embed_size, K, adv, adver, reg_adv, learning_rate, eps, adapt_l2=0, base='deepwalk'):
		self.node_num = node_num
		self.embed_size = embed_size
		self.K = K
		self.adver = adver
		self.adv = adv
		self.reg_adv = reg_adv
		self.learning_rate = learning_rate
		self.eps = eps
		self.adapt_l2 = adapt_l2  # 0: no adaptive l2-norm constraint, 1: adaptive l2-norm, 2: weighted regularizer for different pairs
		self.base = base

		self._build_graph()

	def _build_graph(self):	
	    self._create_placeholders()
	    self._create_variables()
	    self._create_loss()
	    self._create_optimizer()
	    self._create_adversarial()
	    self._normalize_embedding()

	def _create_placeholders(self):
	    with tf.name_scope("input_data"):
	        self.target_P = tf.placeholder(tf.int32, shape=(None), name="target_P")
	        self.positive = tf.placeholder(tf.int32, shape=(None), name="positive")

	        self.weights = tf.placeholder(tf.float32, shape=(None), name="weights")

	        self.target_N = tf.placeholder(tf.int32, shape=(None), name="target_N")            
	        self.negative = tf.placeholder(tf.int32, shape=(None), name="negative")
	        self.batch_size = tf.placeholder(tf.int32, shape=(), name="batch_size")

	def _create_variables(self):
		with tf.name_scope("embedding"):
			# target embeddings
			with tf.device("/cpu:0"):
				self.embedding_T = tf.Variable(
				    tf.random_uniform([self.node_num, self.embed_size], -1/self.embed_size, 1/self.embed_size),
				    name="embedding_T", dtype=tf.float32)
				# context embeddings
				self.embedding_C = tf.Variable(
				    tf.truncated_normal([self.node_num, self.embed_size], stddev=1.0 / math.sqrt(self.embed_size)),
				    name="embedding_C", dtype=tf.float32)
				# self.context_bias = tf.Variable(
				# 	tf.zeros([self.node_num]),
				# 	name="context_bias", trainable=False)			    
				self.context_bias = tf.Variable(
				    tf.zeros([self.node_num]),
				    name="context_bias", trainable=True)

				self.delta_T = tf.Variable(tf.zeros(shape=[self.node_num, self.embed_size]),
				                           name="delta_T", dtype=tf.float32, trainable=False)
				self.delta_C = tf.Variable(tf.zeros(shape=[self.node_num, self.embed_size]),
				                           name="delta_C", dtype=tf.float32, trainable=False)
			    # self.delta_B = tf.Variable(tf.zeros(shape=[self.node_num]), 
			    # 						   name='delta_B', dtype=tf.float32, trainable=False)

	def _normalize_embedding(self):
		with tf.name_scope('normalization'):
			self.normalize_emb_T = self.embedding_T.assign(tf.nn.l2_normalize(self.embedding_T))
			self.normalize_emb_C = self.embedding_C.assign(tf.nn.l2_normalize(self.embedding_C))

	def _create_inference(self, target, node_ctx):
	    with tf.name_scope("inference"):
	        # embedding look up
	        self.embedding_t = tf.nn.embedding_lookup(self.embedding_T, target)

	        if self.base=='deepwalk' or self.base=='node2vec' or self.base=='LINE_2':
	        	self.embedding_c = tf.nn.embedding_lookup(self.embedding_C, node_ctx)  # (b, embed_size)
	        elif self.base=='LINE_1':
	        	self.embedding_c = tf.nn.embedding_lookup(self.embedding_T, node_ctx)  # (b, embed_size)

	        self.bias = tf.nn.embedding_lookup(self.context_bias, node_ctx)
	        return tf.reduce_sum(tf.multiply(self.embedding_t, self.embedding_c), 1) + self.bias  # (b, embed_size) * (embed_size, 1)

	def _create_inference_adv(self, target, node_ctx, flag=True):
		'''
		flag: true for target-positive, false for target-negative
		'''
		with tf.name_scope("inference_adv"):
			# embedding look up
			self.embedding_t = tf.nn.embedding_lookup(self.embedding_T, target)

			if self.base=='deepwalk' or self.base=='node2vec' or self.base=='LINE_2':
				self.embedding_c = tf.nn.embedding_lookup(self.embedding_C, node_ctx)  # (b, embed_size)
			elif self.base=='LINE_1':
				self.embedding_c = tf.nn.embedding_lookup(self.embedding_T, node_ctx)  # (b, embed_size)

			self.bias = tf.nn.embedding_lookup(self.context_bias, node_ctx)
			# add adversarial noise
			if not flag:
				self.T_plus_delta = self.embedding_t + tf.nn.embedding_lookup(self.delta_T, target)
				if self.base=='deepwalk' or self.base=='node2vec' or self.base=='LINE_2':
					self.C_plus_delta = self.embedding_c + tf.nn.embedding_lookup(self.delta_C, node_ctx)
				elif self.base=='LINE_1':
					self.C_plus_delta = self.embedding_c + tf.nn.embedding_lookup(self.delta_T, node_ctx)
			else:
				weights = tf.concat([tf.reshape(self.weights, (-1, 1))]*self.embed_size, axis=-1)
				self.T_plus_delta = self.embedding_t + weights * tf.nn.embedding_lookup(self.delta_T, target)
				if self.base=='deepwalk' or self.base=='node2vec' or self.base=='LINE_2':
					self.C_plus_delta = self.embedding_c + weights * tf.nn.embedding_lookup(self.delta_C, node_ctx)
				elif self.base=='LINE_1':
					self.C_plus_delta = self.embedding_c + weights * tf.nn.embedding_lookup(self.delta_T, node_ctx)

			# self.B_plus_delta = self.bias + tf.nn.embedding_lookup(self.delta_B, node_ctx)
			return tf.reduce_sum(tf.multiply(self.T_plus_delta, self.C_plus_delta), 1) + self.bias  # (b, embed_size) * (embed_size, 1)

	def _loss(self, score, score_neg, flag=True):
		'''
		flag: 
		- True: basic loss
		- False: regularizer
		'''
		with tf.name_scope("loss"):
			if flag:
				true_xent = tf.nn.sigmoid_cross_entropy_with_logits(
				    labels=tf.ones_like(score), logits=score)
				negative_xent = tf.nn.sigmoid_cross_entropy_with_logits(
				        labels=tf.zeros_like(score_neg), logits=score_neg)
				loss = tf.reduce_sum(true_xent) + tf.reduce_sum(negative_xent)
			else:
				if self.adapt_l2==0 or self.adapt_l2==1:
					true_xent = tf.nn.sigmoid_cross_entropy_with_logits(
					    labels=tf.ones_like(score), logits=score)
					negative_xent = tf.nn.sigmoid_cross_entropy_with_logits(
					        labels=tf.zeros_like(score_neg), logits=score_neg)
					loss = tf.reduce_sum(true_xent) + tf.reduce_sum(negative_xent)
				elif self.adapt_l2==2:
					true_xent = tf.nn.sigmoid_cross_entropy_with_logits(
					    labels=tf.ones_like(score), logits=score)
					negative_xent = tf.nn.sigmoid_cross_entropy_with_logits(
					        labels=tf.zeros_like(score_neg), logits=score_neg)
					weight_true = self.weights
					weight_neg = tf.reshape(tf.concat([tf.reshape(self.weights, (-1, 1))]*self.K, axis=1), (-1, 1))
					weight_neg = tf.reshape(weight_neg, [-1])
					loss = tf.reduce_sum(weight_true*true_xent) + tf.reduce_sum(weight_neg*negative_xent)

			return loss / tf.cast(self.batch_size, tf.float32)

	def _create_loss(self):
		with tf.name_scope("loss_overall"):
			# loss for L(Theta)
			self.score = self._create_inference(self.target_P, self.positive)
			self.score_neg = self._create_inference(self.target_N, self.negative)
			self.loss_basic = self._loss(self.score, self.score_neg, flag=True)
			self.loss_opt = self.loss_basic

			if self.adver:
				# loss for L(Theta + adv_Delta)
				if self.adapt_l2>0:
					if self.adapt_l2==1:
						self.score_adv = self._create_inference_adv(self.target_P, self.positive, flag=True)
						# self.score_neg_adv = self._create_inference_adv(self.target_N, self.negative, flag=False)  # WWW 2019
						self.score_neg_adv = self._create_inference_adv(self.target_N, self.negative, flag=False)
					elif self.adapt_l2==2:
						self.score_adv = self._create_inference_adv(self.target_P, self.positive, flag=False)
						self.score_neg_adv = self._create_inference_adv(self.target_N, self.negative, flag=False)
				else:
					self.score_adv = self._create_inference_adv(self.target_P, self.positive, flag=False)
					self.score_neg_adv = self._create_inference_adv(self.target_N, self.negative, flag=False)

				self.loss_adv = self._loss(self.score_adv, self.score_neg_adv, flag=False)
				self.loss_opt += self.reg_adv * self.loss_adv

	def _create_adversarial(self):
		with tf.name_scope("adversarial"):
			# generate the adversarial weights by random method
			if self.adv == "random":
				# generation
				self.adv_T = tf.truncated_normal(shape=[self.node_num, self.embed_size], mean=0.0, stddev=0.01)
				self.adv_C = tf.truncated_normal(shape=[self.node_num, self.embed_size], mean=0.0, stddev=0.01)
				# self.adv_B = tf.truncated_normal(shape=(self.node_num), mean=0.0, stddev=0.01)

				# normalization and multiply epsilon
				self.update_T = self.delta_T.assign(tf.nn.l2_normalize(self.adv_T, 1) * self.eps)
				self.update_C = self.delta_C.assign(tf.nn.l2_normalize(self.adv_C, 1) * self.eps)
				# self.update_B = self.delta_B.assign(tf.nn.l2_normalize(self.adv_B) * self.eps)

			# generate the adversarial weights by gradient-based method
			elif self.adv == "grad":
				# return the IndexedSlice Data: [(values, indices, dense_shape)]
				# grad_var_P: [grad, var], grad_var_Q: [grad, var]

				if self.base=='deepwalk' or self.base=='node2vec' or self.base=='LINE_2':
					self.grad_T, self.grad_C = tf.gradients(self.loss_basic, [self.embedding_T, self.embedding_C])
					# convert the IndexedSlice Data to Dense Tensor
					self.grad_T_dense = tf.stop_gradient(self.grad_T)
					self.grad_C_dense = tf.stop_gradient(self.grad_C)
					# normalization: new_grad = (grad / |grad|) * eps
					self.update_T = self.delta_T.assign(tf.nn.l2_normalize(self.grad_T_dense, 1) * self.eps)
					self.update_C = self.delta_C.assign(tf.nn.l2_normalize(self.grad_C_dense, 1) * self.eps)
				elif self.base=='LINE_1':
					self.grad_T = tf.gradients(self.loss_basic, [self.embedding_T])[0]
					# convert the IndexedSlice Data to Dense Tensor
					self.grad_T_dense = tf.stop_gradient(self.grad_T)
					# normalization: new_grad = (grad / |grad|) * eps
					self.update_T = self.delta_T.assign(tf.nn.l2_normalize(self.grad_T_dense, 1) * self.eps)

	def _create_optimizer(self):
		with tf.name_scope("optimizer"):
			# self.optimizer = tf.train.AdagradOptimizer(learning_rate=self.learning_rate).minimize(self.loss_opt)  # learn nothing using this optimizer
			self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss_opt)

	def get_normalized_embeddings(self):
		return tf.nn.l2_normalize(self.embedding_T, 1)
