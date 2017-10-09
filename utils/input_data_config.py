import pandas as pd
import numpy as np
import random
import pickle
from deeplift import dinuc_shuffle



# One hot encoding:
def one_hot(seq):
	encoded_seq = np.zeros((4, len(seq)),dtype=np.float32)
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
		if nt == 'N':
			encoded_seq[:, pos] = [0.31,0.19,0.19,0.31]
	return encoded_seq

# ratios is a 3-tuple or 3-list that gives training:val:test
def partition(data, ratios, mode='random'):

	assert mode in ['random','whole_gene','whole_experiment'], \
		'mode must be one of "random", "whole_gene", and "whole experiment"'
	ratio_tvt = np.array(ratios)
	pct_tvt = ratio_tvt/float(sum(ratio_tvt))

	if mode == 'random':

		partition_vec = np.zeros(len(data))
		partition_vec[int(len(data)*pct_tvt[:1]):] = 1
		partition_vec[int(len(data)*pct_tvt[:2].sum()):] = 2
		
		random.seed(42)
		random.shuffle(partition_vec)
		
		split=[data[partition_vec==i] for i in range(3)]

	elif mode=='whole_gene':
		uid=data['UID'].unique()

		partition_vec = np.zeros(len(uid))
		partition_vec[int(len(uid)*pct_tvt[:1]):] = 1
		partition_vec[int(len(uid)*pct_tvt[:2].sum()):] = 2

		split=[]
		for i in range(3):
			idx=data['UID'].isin(uid[partition_vec==i])
			split.append(data[idx])

		assert all(~split[0]['UID'].isin(split[1]['UID'])) and \
			all(~split[0]['UID'].isin(split[2]['UID'])) and \
			all(~split[1]['UID'].isin(split[2]['UID'])), \
			'whole-gene hold out not correct!'
		

	else:
		experiment=data['experiment'].unique()

		partition_vec = np.zeros(len(experiment))
		partition_vec[int(len(experiment)*pct_tvt[:1]):] = 1
		partition_vec[int(len(experiment)*pct_tvt[:2].sum()):] = 2

		split=[]
		for i in range(3):
			idx=data['experiment'].isin(experiment[partition_vec==i])
			split.append(data[idx])

		assert all(~split[0]['experiment'].isin(split[1]['experiment'])) and \
			all(~split[0]['experiment'].isin(split[2]['experiment'])) and \
			all(~split[1]['experiment'].isin(split[2]['experiment'])), \
			'whole-experiment hold out not correct!'
		
	return(split)

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

def reformat(data,conv1d=False,seq_only=False,experiment_name=None):
	nrow=data['one_hot_sequence'].shape[0]
	if conv1d:
		seq=np.swapaxes(np.reshape(np.vstack(data['one_hot_sequence']),[nrow,4,1000]),1,2)
	else:
		seq=np.reshape(np.vstack(data['one_hot_sequence']),[nrow,4,1000,1])


	if seq_only==False:
		reg_exp=np.vstack(data['reg_exp'])
		exp=np.array(data['expression'])
		label=discretize(exp)
		data2={'seq':seq, 'reg':reg_exp, 'expr':exp, 'class':label, 'uid': data['UID'], 'name': data['NAME'], 'experiment': data['experiment'], 'sequence': data['sequence'], 'real': data['real']}
	else:
		exp=data[experiment_name]
		data2={'seq':seq, 'expr':exp, 'uid': data['UID'], 'name': data['NAME'], 'sequence': data['sequence'], 'real': data['real']}
	return data2


# seq_file='../data/yeast_promoters.txt'
# expr_file='../data/complete_dataset.txt'
# reg_names_file='../data/reg_names_R.txt'

def read_data_sets(train_pct=80,val_pct=10,test_pct=10,mode='random',conv1d=False,add_shuffle=False,seq_only=False,seq_file='../data/yeast_promoters.txt',expr_file='../data/complete_dataset.txt',reg_names_file='../data/reg_names_R.txt'):
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
	promoters['real'] = 1 # denotes real promoters
	promoters.loc[:, "one_hot_sequence"] = [one_hot(seq) for seq in promoters.loc[:, "sequence"]]


	# Create dinucleotide shuffled sequences:
	if add_shuffle:
		random.seed(a=42)
		shuffled_promoters=[]
		for i in xrange(promoters.shape[0]):
			shuffled_promoters.append(dinuc_shuffle.dinuc_shuffle(promoters.sequence[i]))
		promoters = pd.concat([promoters,pd.DataFrame({'UID':promoters.UID.tolist(),'sequence':shuffled_promoters, 'real':0})])


	if seq_only:
		data_complete=pd.merge(promoters,expr_data,on='UID',how='inner')

		# train_pct=90
		# val_pct=5
		# test_pct=5
		# mode = 'whole_gene'
		experiment_name=expr_data.drop(['UID','NAME'],axis=1).columns.tolist()

		train,val,test=partition(data_complete,(train_pct,val_pct,test_pct), mode=mode)

		train_data=reformat(train,conv1d,seq_only,experiment_name) if train_pct > 0 else []
		val_data=reformat(val,conv1d,seq_only,experiment_name) if val_pct > 0 else []
		test_data=reformat(test,conv1d,seq_only,experiment_name) if test_pct > 0 else []

	else:
		# Some transformation: 
		target_expr_data = pd.melt(expr_data, id_vars=["UID","NAME"], var_name="experiment", value_name="expression")
		reg_data = pd.merge(reg_names, expr_data, on="UID", how="inner").drop("UID", axis=1)


		reg = pd.DataFrame()
		for col in range(len(reg_data.columns)):
			data = np.array([exp_level for exp_level in reg_data.iloc[:, col]])
			reg = reg.append(pd.DataFrame({"experiment": reg_data.columns[col], "reg_exp": [data]}))

		data_complete = pd.merge(promoters, target_expr_data, on="UID", how="inner").merge(reg, on="experiment", how="inner")
		data_complete.expression=data_complete.expression*data_complete.real # force dinucleotide shuffled sequences to have 0 expression. 

		# train_pct=80
		# val_pct=10
		# test_pct=10
		# mode = 'random'
		train, val, test = partition(data_complete, (train_pct,val_pct,test_pct), mode=mode)

		train_data=reformat(train,conv1d) if train_pct > 0 else []
		val_data=reformat(val,conv1d) if val_pct > 0 else []
		test_data=reformat(test,conv1d) if test_pct > 0 else []

	return [train_data,val_data,test_data]