#!/usr/bin/env python

import sys, os.path
from sys import argv
from os import system
from string import *
import scipy
import scipy.io as sio
import numpy as np
import argparse
import time
from .liblinearutil import *
from .subr import *

new_class = []

def build_new_file(file, file_name):
	out_file = open(file_name,"w")
	in_file = open(file,"r")
	for line in in_file:
		spline = line.split()

		labels = []
		if spline[0].find(':') == -1:
			labels = spline[0].split(',')
			labels.sort()

		if (labels not in new_class):
			new_class.append(labels)

		if len(labels) == 0:
			out_file.write("%s %s\n"%(new_class.index(labels), ' '.join(spline)))
		else:
			out_file.write("%s %s\n"%(new_class.index(labels), ' '.join(spline[1:])))
	out_file.close()
	in_file.close()

def trans_class(trainFile, testFile, path, dataset):
	build_new_file(trainFile, path+dataset+"-train")
	# print "Number of Training classes (sets of labels) is %s" % len(new_class)

	out_class = open(path+dataset+"-class","w")	
	for cl in new_class:
		out_class.write("%s\n" % ",".join(map(lambda num:("%s"%num),cl)))
	out_class.close()

	build_new_file(testFile, path+dataset+"-test")


def mlc_liblinear(trainFile, testFile, path, dataset):
	# transform to multi-class problem
	trans_class(trainFile, testFile, path, dataset)
	tmp_train = path + dataset + "-train"
	tmp_test = path + dataset + "-test"
	tmp_class = path + dataset + "-class"
	model = path + dataset + "-train.model"

	Ytr, Xtr = svm_read_problem(tmp_train)
	Yts, Xts = svm_read_problem(tmp_test)

	# train and predict
	m = train(Ytr, Xtr, '-s 2 -c 1 -q')
	save_model(model, m)
	p_predict, p_acc, p_val = predict(Yts, Xts, m)

	# measure
	original = read_first_column(testFile)
	train_new_class = read_first_column(tmp_class)

	prediction = []
	for i in range(len(p_predict)):
		idx = int(str(int(p_predict[i])))
		prediction.append(train_new_class[idx])

	if(len(prediction) != len(original)):
		print ("Error: lines of %s and %s are different." % (argv[1],argv[2]))
		sys.exit(1)

	labels = []
	for i in range(len(train_new_class)):
		for lab in train_new_class[i]:
			if (lab not in labels):
				labels.append(lab)

	# print "number of labels = %s" % len(labels)

	# result[0]: exact match ratio; result[1]: Micro-F1; result[2]: Macro-F1
	result = measure(original,prediction,labels)
	return result

def main():
	trainFile = 'test/blog-train-1-1.txt'
	testFile = 'test/blog-test-1-1.txt'
	path = 'deepwalk/'
	dataset = 'blog'
	result = mlc_liblinear(trainFile, testFile, path, dataset)
	print (result)

if __name__ == "__main__":
    main()








# def mlc_liblinear(trainFile, testFile, path):
# 	# transform to multi-class problem
# 	trans_class(trainFile, testFile, path)
# 	tmp_train = path + "tmp_train"
# 	tmp_test = path + "tmp_test"
# 	tmp_class = path + "tmp_class"
# 	tmp_predict = path + "tmp_predict"
# 	model = path + 'tmp_train.model'

# 	Ytr, Xtr = svm_read_problem(tmp_train)
# 	Yts, Xts = svm_read_problem(tmp_test)
# 	tmp_pre = open(tmp_predict, "w")

# 	# train and predict
# 	m = train(Ytr, Xtr, '-s 2 -c 1 -q')
# 	save_model(model, m)
# 	p_predict, p_acc, p_val = predict(Yts, Xts, m)

# 	for i in range(len(p_predict)):
# 		tmp_pre.write('{}\n'.format(str(int(p_predict[i]))))
# 	tmp_pre.close()

# 	# measure
# 	original = read_first_column(testFile)
# 	train_new_class = read_first_column(tmp_class)
# 	test_output = read_first_column(tmp_predict)

# 	prediction = []
# 	for i in range(len(test_output)):
# 		idx = atoi(test_output[i][0])
# 		prediction.append(train_new_class[idx])

# 	if(len(prediction) != len(original)):
# 		print "Error: lines of %s and %s are different." % (argv[1],argv[2])
# 		sys.exit(1)

# 	labels = []
# 	for i in range(len(train_new_class)):
# 		for lab in train_new_class[i]:
# 			if (lab not in labels):
# 				labels.append(lab)

# 	print "number of labels = %s" % len(labels)

# 	# result[0]: exact match ratio; result[1]: Micro-F1; result[2]: Macro-F1
# 	result = measure(original,prediction,labels)
# 	return result
	
