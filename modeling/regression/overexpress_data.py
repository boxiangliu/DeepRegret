import pandas as pd
import numpy as np
import random
import collections

# Class to hold train, val, and test DataSet: 
Datasets = collections.namedtuple('Datasets', ['overexpress_data'])

# DataSet class
class DataSet(object):
	def __init__(self,seqs,reg_exprs,labels,meta):
		"""Construct a DataSet."""
		self._seqs = seqs
		self._reg_exprs = reg_exprs
		self._labels = labels
		self._meta = meta
		self._epochs_completed = 0
		self._index_in_epoch = 0
		self._num_examples = labels.shape[0]

	@property
	def seqs(self):
		return self._seqs

	@property
	def reg_exprs(self):
		return self._reg_exprs

	@property
	def labels(self):
		return self._labels

	@property
	def meta(self):
		return self._meta

	@property
	def num_examples(self):
		return self._num_examples


	@property
	def epochs_completed(self):
		return self._epochs_completed

	def next_batch(self, batch_size):
		"""Return the next `batch_size` examples from this data set."""
		start = self._index_in_epoch
		self._index_in_epoch += batch_size
		if self._index_in_epoch > self._num_examples:
			# Finished epoch
			self._epochs_completed += 1
			# Shuffle the data
			perm = np.arange(self._num_examples)
			np.random.shuffle(perm)
			self._seqs = self._seqs[perm]
			self._reg_exprs = self._reg_exprs[perm]
			self._labels = self._labels[perm]
			# Start next epoch
			start = 0
			self._index_in_epoch = batch_size
			assert batch_size <= self._num_examples
		end = self._index_in_epoch
		return self._seqs[start:end], self._reg_exprs[start:end], self._labels[start:end], self._meta[start:end]


def one_hot(seq):
	encoded_seq = np.zeros((4, len(seq)))
	for pos in range(len(seq)):
		nt = seq[pos]
		if nt == "A":
			encoded_seq[0, pos] = 1
		if nt == "C":
			encoded_seq[1, pos] = 1
		if nt == "G":
			encoded_seq[2, pos] = 1
		if nt == "T":
			encoded_seq[3, pos] = 1
	return encoded_seq

# ratios is a 3-tuple or 3-list that gives training:val:test
def partition(data, ratios):
	ratio_tvt = np.array(ratios)
	pct_tvt = ratio_tvt/float(sum(ratio_tvt))

	partition_vec = np.zeros(len(data))
	partition_vec[int(len(data)*pct_tvt[:1]):] = 1
	partition_vec[int(len(data)*pct_tvt[:2].sum()):] = 2
	
	random.seed(42)
	random.shuffle(partition_vec)
	
	return [data[partition_vec==i] for i in range(3)]



seq_file='../data/yeast_promoters.txt'
expr_file='../data/complete_dataset.txt'
reg_names_file='../data/reg_names_R.txt'

def read_data_sets(seq_file='../data/yeast_promoters.txt',expr_file='../data/complete_dataset.txt',reg_names_file='../data/reg_names_R.txt',fold_change=0.0):
	'''Read data from text files into numpy arrays
		Args:
		seq_file: promoter sequence file.
		expr_file: expression values file. 
		reg_names_file: regulator gene names file. 

		Returns:
		seq: promoter sequences (one-hot encoding).
		reg_expr: regulator gene expression (1D array).
		label: gene expression (scalar). 
	'''
	# Read data: 
	reg_names = pd.read_table(reg_names_file, names = ["UID"])
	expr_data = pd.read_table(expr_file).fillna(0).drop("NAME", axis=1).drop("GWEIGHT", axis=1)
	promoters = pd.read_table(seq_file, names=["UID", "sequence"])

	# Some transformation: 
	target_expr_data = pd.melt(expr_data, id_vars="UID", var_name="experiment", value_name="expression")
	promoters.loc[:, "one_hot_sequence"] = [one_hot(seq) for seq in promoters.loc[:, "sequence"]]
	reg_data = pd.merge(reg_names, expr_data, on="UID", how="inner").drop("UID", axis=1)

	reg = pd.DataFrame()
	reg = reg.append(pd.DataFrame({"experiment": reg_data.columns[0], "reg_exp": [np.zeros(reg_data.shape[0])], 'overexpress': 'reference'}))

	for i in range(reg_data.shape[0]):
		overexpress=np.zeros(reg_data.shape[0])
		overexpress[i]=overexpress[i]+fold_change
		reg=reg.append(pd.DataFrame({"experiment": reg_data.columns[0], "reg_exp": [overexpress], 'overexpress': reg_names.iloc[i,0]}))

	data_complete = pd.merge(promoters, target_expr_data, on="UID", how="inner").merge(reg, on="experiment", how="inner")
	data_nrow=data_complete['one_hot_sequence'].shape[0]
	data_seq=np.swapaxes(np.reshape(np.vstack(data_complete['one_hot_sequence']),[data_nrow,4,1000]),1,2)
	data_reg_exp=np.vstack(data_complete['reg_exp'])
	data_label=np.array(data_complete['expression'])
	data_meta=np.array(data_complete.loc[:,('UID','experiment','overexpress')])

	data=DataSet(data_seq, data_reg_exp, data_label, data_meta)

	return Datasets(overexpress_data=data)