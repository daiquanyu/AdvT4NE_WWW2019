#!/usr/bin/env python

import sys, os.path
from string import *
from sys import argv
from subr import *

if len(argv) < 4:
	print "Usage: %s testing_file testing_output_file training_class" % (argv[0])
	sys.exit(1)

def main():
	original = read_first_column(argv[1])
	test_output = read_first_column(argv[2])
	train_new_class = read_first_column(argv[3])

	predict = []
	for i in range(len(test_output)):
		idx = atoi(test_output[i][0])
		predict.append(train_new_class[idx])

	if(len(predict) != len(original)):
		print "Error: lines of %s and %s are different." % (argv[1],argv[2])
		sys.exit(1)

	labels = []
	for i in range(len(train_new_class)):
		for lab in train_new_class[i]:
			if (lab not in labels):
				labels.append(lab)

	print "number of labels = %s" % len(labels)

	result = measure(original,predict,labels)

	print "Exact match ratio: %s" % result[0]
	print "Microaverage F-measure: %s" % result[1]
	print "Macroaverage F-measure: %s" % result[2]

main()
