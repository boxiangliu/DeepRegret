import pandas as pd
import numpy as np
import random
import pickle




# One hot encoding:
def one_hot(seq):
	encoded_seq = np.zeros((4, len(seq)),dtype=np.int8)
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


def discretize(x):
	'''Discretize x. Values less than -1.3 will be 0. 
	Values greater than 1.3 will be 2. Values in between will 
	be 1.
	''' 
	x=np.array(x)
	y=np.zeros(x.shape[0],dtype=np.int8)
	for i in range(x.shape[0]):
		if x[i] <= -0.5:
			y[i]=0
		elif x[i] >= 0.5:
			y[i]=2
		else:
			y[i]=1
	return y

def reformat(data):
	nrow=data['one_hot_sequence'].shape[0]
	seq=np.reshape(np.vstack(data['one_hot_sequence']),[nrow,4,1000,1])
	reg_exp=np.vstack(data['reg_exp'])
	exp=np.array(data['expression'])
	label=discretize(exp)
	data2={'seq':seq, 'reg':reg_exp, 'expr':exp, 'class':label, 'uid': data['UID'], 'name': data['NAME'], 'experiment': data['experiment'], 'sequence': data['sequence']}
	return data2


# seq_file='../data/yeast_promoters.txt'
# expr_file='../data/complete_dataset.txt'
# reg_names_file='../data/reg_names_R.txt'

def read_data_sets(train_pct=80,val_pct=10,test_pct=10,seq_file='../data/yeast_promoters.txt',expr_file='../data/complete_dataset.txt',reg_names_file='../data/reg_names_R.txt'):
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
	expr_data = pd.read_table(expr_file).fillna(0).drop("GWEIGHT", axis=1)
	expr_data['NAME']=[x[10:21].strip() for x in expr_data['NAME']]
	expr_data[expr_data.columns[2:]]=expr_data[expr_data.columns[2:]].astype('float32')
	promoters = pd.read_table(seq_file, names=["UID", "sequence"])

	# Some transformation: 
	target_expr_data = pd.melt(expr_data, id_vars=["UID","NAME"], var_name="experiment", value_name="expression")
	promoters.loc[:, "one_hot_sequence"] = [one_hot(seq) for seq in promoters.loc[:, "sequence"]]
	reg_data = pd.merge(reg_names, expr_data, on="UID", how="inner").drop("UID", axis=1)


	reg = pd.DataFrame()
	for col in range(len(reg_data.columns)):
		data = np.array([exp_level for exp_level in reg_data.iloc[:, col]])
		reg = reg.append(pd.DataFrame({"experiment": reg_data.columns[col], "reg_exp": [data]}))

	data_complete = pd.merge(promoters, target_expr_data, on="UID", how="inner").merge(reg, on="experiment", how="inner")
	# train_pct=80
	# val_pct=10
	# test_pct=10

	train, val, test = partition(data_complete, (train_pct,val_pct,test_pct))

	train_data=reformat(train) if train_pct > 0 else []
	val_data=reformat(val) if val_pct > 0 else []
	test_data=reformat(test) if test_pct > 0 else []

	return [train_data,val_data,test_data]