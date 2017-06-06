import pandas as pd
import numpy as np
import random
import collections

# One hot encoding:
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


# seq_file='../data/yeast_promoters.txt'
# expr_file='../data/complete_dataset.txt'
# reg_names_file='../data/reg_names_R.txt'

def read_data_sets(seq_file='../data/yeast_promoters.txt',expr_file='../data/complete_dataset.txt',reg_names_file='../data/reg_names_R.txt'):
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
	for col in range(len(reg_data.columns)):
		data = np.array([exp_level for exp_level in reg_data.iloc[:, col]])
		reg = reg.append(pd.DataFrame({"experiment": reg_data.columns[col], "reg_exp": [data]}))

	data_complete = pd.merge(promoters, target_expr_data, on="UID", how="inner").merge(reg, on="experiment", how="inner")

	train, val, test = partition(data_complete, (80,10,10))

	train_nrow=train['one_hot_sequence'].shape[0]
	train_seq=np.reshape(np.vstack(train['one_hot_sequence']),[train_nrow,4,1000,1])
	train_reg_exp=np.vstack(train['reg_exp'])
	train_exp=np.array(train['expression'])

	val_nrow=val['one_hot_sequence'].shape[0]
	val_seq=np.reshape(np.vstack(val['one_hot_sequence']),[val_nrow,4,1000,1])
	val_reg_exp=np.vstack(val['reg_exp'])
	val_exp=np.array(val['expression'])

	test_nrow=test['one_hot_sequence'].shape[0]
	test_seq=np.reshape(np.vstack(test['one_hot_sequence']),[test_nrow,4,1000,1])
	test_reg_exp=np.vstack(test['reg_exp'])
	test_exp=np.array(test['expression'])

	train_data={'seq':train_seq, 'reg':train_reg_exp, 'expr':train_exp}
	val_data={'seq':val_seq, 'reg':val_reg_exp, 'expr':val_exp}
	test_data={'seq':test_seq, 'reg':test_reg_exp, 'expr':test_exp}

	return [train_data,val_data,test_data]