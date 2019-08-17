import sys, os.path
from string import *

def read_first_column(file):
	obj = []
	assert os.path.exists(file), "%s not found." % (file)
	in_file = open(file, "r")
	for line in in_file:
		spline = line.split()
		if len(line) == 1 or spline[0].find(':') != -1:
			# instance with no label or no feature
			obj.append([])
		else:
			obj.append(spline[0].split(','))
	in_file.close()
	return obj

def measure(__original, __predict, __labels):
	"""
	Return exact match ratio, microaverage F-measure, and macroaverage F-measure.
	"""
	result = []
	
	# Exact Match Ratio
	ratio = 0
	for i in range(len(__predict)):
		__original[i].sort()
		__predict[i].sort()
		if(__original[i] == __predict[i]):
			ratio = ratio+1

	result.append(float(ratio)/len(__predict))
	
	# Microaverage and Macroaverage F-measure
	F = 0
	tp_sum = 0
	fp_sum = 0
	fn_sum = 0

	for j in __labels:
		tp = 0
		fp = 0
		fn = 0
		tn = 0

		for i in range(len(__predict)):
			if (j in __original[i] and j in __predict[i]):
				tp = tp + 1
			elif (j not in __original[i] and j in __predict[i]):
				fp = fp + 1
			elif (j in __original[i] and j not in __predict[i]):
				fn = fn + 1
			else:
				tn = tn + 1

		# 0/0 is treated as 0 and #labels does *not* reduced
		if (tp != 0 or fp != 0 or fn != 0):
			F = F+float(2*tp)/float(2*tp+fp+fn)

		tp_sum = tp_sum + tp
		fp_sum = fp_sum + fp
		fn_sum = fn_sum + fn

	P = float(tp_sum)/float(tp_sum+fp_sum)
	R = float(tp_sum)/float(tp_sum+fn_sum)

	result.append(2*P*R/(P+R))
	result.append(F/len(__labels))

	return result
