import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from keras.models import load_model
import sys
sys.path.append('utils/')
import input_data_config as input_data

from sklearn.metrics import confusion_matrix
from sklearn.metrics import log_loss
#Convert a keras sequential model
import deeplift
from deeplift.conversion import keras_conversion as kc
from deeplift.blobs import NonlinearMxtsMode
from deeplift.visualization import viz_sequence
import os

from sklearn.cluster.bicluster import SpectralBiclustering
from sklearn.metrics import consensus_score
from scipy.stats import ranksums
from scipy.stats import ttest_ind

from collections import OrderedDict

# Load model:
model_fn='../logs/concatenation/regression/model.41-0.2300.hdf5'
model=load_model(model_fn)
full_data,_,_=input_data.read_data_sets(train_pct=100,val_pct=0,test_pct=0)

#NonlinearMxtsMode defines the method for computing importance scores.
#NonlinearMxtsMode.DeepLIFT_GenomicsDefault uses the RevealCancel rule on Dense layers
#and the Rescale rule on conv layers (see paper for rationale)
#Other supported values are:
#NonlinearMxtsMode.RevealCancel - DeepLIFT-RevealCancel at all layers (used for the MNIST example)
#NonlinearMxtsMode.Rescale - DeepLIFT-rescale at all layers
#NonlinearMxtsMode.Gradient - the 'multipliers' will be the same as the gradients
#NonlinearMxtsMode.GuidedBackprop - the 'multipliers' will be what you get from guided backprop
#Use deeplift.util.get_integrated_gradients_function to compute integrated gradients
#Feel free to email avanti [dot] shrikumar@gmail.com if anything is unclear
deeplift_model = kc.convert_functional_model(model,
                    nonlinear_mxts_mode=deeplift.blobs.NonlinearMxtsMode.Gradient)

# deeplift_model.get_name_to_blob().keys() 
#Specify the index of the layer to compute the importance scores of.
#In the example below, we find scores for the input layer, which is idx 0 in deeplift_model.get_layers()
find_scores_layer_name = 'reg_input'

#Compile the function that computes the contribution scores
#For sigmoid or softmax outputs, target_layer_idx should be -2 (the default)
#(See "3.6 Choice of target layer" in https://arxiv.org/abs/1704.02685 for justification)
#For regression tasks with a linear output, target_layer_idx should be -1
#(which simply refers to the last layer)
#If you want the DeepLIFT multipliers instead of the contribution scores, you can use get_target_multipliers_func
deeplift_contribs_func = deeplift_model.get_target_contribs_func(find_scores_layer_name=find_scores_layer_name,
	pre_activation_target_layer_name='rgs_output')
#You can also provide an array of indices to find_scores_layer_idx to get scores for multiple layers at once

#compute scores on inputs
#input_data_list is a list containing the data for different input layers
#eg: for MNIST, there is one input layer with with dimensions 1 x 28 x 28
#In the example below, let X be an array with dimension n x 1 x 28 x 28 where n is the number of examples
#task_idx represents the index of the node in the output layer that we wish to compute scores.
#Eg: if the output is a 10-way softmax, and task_idx is 0, we will compute scores for the first softmax class
X=[full_data['reg'],full_data['seq']]
background = OrderedDict([('A', 0.31), ('C', 0.19), ('G', 0.19), ('T', 0.31)])
scores = np.array(deeplift_contribs_func(task_idx=0,
	input_data_list=X,
	batch_size=100,
	input_references_list=[np.array([0])[None,:],np.array([background['A'],background['C'],background['G'],background['T']])[None,:,None,None]],
	progress_update=1000))

out_dir='../processed_data/concatenation/concat.regres.deeplift/'
if not os.path.exists(out_dir): os.makedirs(out_dir)
np.save('%s/reg_deeplift.npy'%out_dir,scores)
# scores=np.genfromtxt('%s/reg_deeplift.txt'%out_dir,delimiter='\t')

scores_abs=abs(scores)
colnames=pd.read_csv('../data/reg_names_R.txt',header=None)
deeplift=pd.DataFrame(scores_abs,
	index=range(scores_abs.shape[0]),
	columns=colnames[0].tolist())
deeplift['uid']=full_data['uid']
deeplift['name']=full_data['name']
deeplift['experiment']=full_data['experiment']
deeplift['experiment']


experiment=['heat shock 17 to 37, 20 minutes',
			'heat shock 21 to 37, 20 minutes',
			'heat shock 25 to 37, 20 minutes',
			'heat shock 29 to 37, 20 minutes',
			'heat shock 33 to 37, 20 minutes',
			'steady state 17 dec C ct-2',
			'steady state 21 dec C ct-2',
			'steady state 25 dec C ct-2',
			'steady state 29 dec C ct-2',
			'steady state 33 dec C ct-2']

msn2_score=[]
msn4_score=[]
for e in experiment:
	deeplift_sub_experiment=deeplift[deeplift.experiment==e]
	deeplift_sub_experiment_sum=deeplift_sub_experiment.iloc[:,0:472].sum(axis=0)

	temp=deeplift_sub_experiment_sum[deeplift_sub_experiment_sum.index=='YMR037C']
	msn2_score.append(temp)

	temp=deeplift_sub_experiment_sum[deeplift_sub_experiment_sum.index=='YKL062W']
	msn4_score.append(temp)

deeplift_sum=deeplift.iloc[:,0:472].sum(axis=0)
deeplift_sum_rank=deeplift_sum.rank(ascending=False)
deeplift_sum_rank[deeplift_sum_rank<=10]
# YGL099W     5.0 LSG1 / YGL099W
# YGR123C     4.0 PPT1 / YGR123C
# YIL101C     3.0 XBP1 / YIL101C
# YIR026C     7.0 YVH1 / YIR026C
# YJL164C     8.0 TPK1 / YJL164C
# YKL062W    10.0 MSN4 / YKL062W
# YKR034W     2.0 DAL80 / YKR034W
# YOR028C     6.0 CIN5 / YOR028C
# YOR178C     9.0 GAC1 / YOR178C
# YPL230W     1.0 USV1 / YPL230W


deeplift_sum.plot.hist()
fig_dir='../figures/concatenation/concat.regress.deeplift/'
if not os.path.exists(fig_dir): os.makedirs(fig_dir)
plt.savefig('%s/deeplift_sum_hist.pdf'%fig_dir)


# msn2_downstream=deeplift[['YMR037C','uid','experiment']]
# msn2_downstream=msn2_downstream.pivot(index='experiment',columns='uid',values='YMR037C')
# msn2_downstream_colsum=msn2_downstream.sum(axis=0)
# msn2_downstream_colsum_rank=msn2_downstream_colsum.rank(ascending=False)

# msn2_targets=pd.read_csv('../data/sgd/MSN2_targets.txt',comment='!',delimiter='\t')
# y=[]
# for x in msn2_downstream_colsum_rank.index.tolist():
# 	temp=x in msn2_targets['Target Systematic Name'].tolist()
# 	y.append(temp)

# z=[not i for i in y]
# ttest_ind(msn2_downstream_colsum[y],msn2_downstream_colsum[z]).pvalue
# # 2.5398582296710098e-13


# msn2_targets_idx=[x in msn2_targets['Target Systematic Name'].tolist() for x in msn2_downstream.columns]
# not_msn2_targets_idx=[(not x) for x in msn2_targets_idx]

# experiment=['heat shock 17 to 37, 20 minutes',
# 			'heat shock 21 to 37, 20 minutes',
# 			'heat shock 25 to 37, 20 minutes',
# 			'heat shock 29 to 37, 20 minutes',
# 			'heat shock 33 to 37, 20 minutes',
# 			'steady state 17 dec C ct-2',
# 			'steady state 21 dec C ct-2',
# 			'steady state 25 dec C ct-2',
# 			'steady state 29 dec C ct-2',
# 			'steady state 33 dec C ct-2']

# msn2_target_genes=msn2_downstream.iloc[msn2_downstream.index.isin(experiment),msn2_targets_idx]
# not_msn2_target_genes=msn2_downstream.iloc[msn2_downstream.index.isin(experiment),not_msn2_targets_idx]
# msn2_target_genes.sum(axis=1)

# experiment
# heat shock 17 to 37, 20 minutes    1.444691
# heat shock 21 to 37, 20 minutes    0.926695
# heat shock 25 to 37, 20 minutes    1.255724
# heat shock 29 to 37, 20 minutes    2.217280
# heat shock 33 to 37, 20 minutes    6.043546
# steady state 17 dec C ct-2         0.610380
# steady state 21 dec C ct-2         0.575973
# steady state 25 dec C ct-2         1.223484
# steady state 29 dec C ct-2         0.487729
# steady state 33 dec C ct-2         1.398120

# not_msn2_target_genes.sum(axis=1)
# experiment
# heat shock 17 to 37, 20 minutes    17.215557
# heat shock 21 to 37, 20 minutes    12.213512
# heat shock 25 to 37, 20 minutes    18.266941
# heat shock 29 to 37, 20 minutes    33.889874
# heat shock 33 to 37, 20 minutes    85.273338
# steady state 17 dec C ct-2          8.701283
# steady state 21 dec C ct-2          8.778999
# steady state 25 dec C ct-2         17.701382
# steady state 29 dec C ct-2          7.966400
# steady state 33 dec C ct-2         22.219212


#---------- sequence layer -----------#
# deeplift_model.get_name_to_blob().keys() 
#Specify the index of the layer to compute the importance scores of.
find_scores_layer_name = 'seq_input'

#Compile the function that computes the contribution scores
deeplift_contribs_func = deeplift_model.get_target_contribs_func(find_scores_layer_name=find_scores_layer_name,
	pre_activation_target_layer_name='rgs_output')

#Compute scores on inputs
idx=232421
X=[full_data['reg'][:100],full_data['seq'][:100]]
background = OrderedDict([('A', 0.31), ('C', 0.19), ('G', 0.19), ('T', 0.31)])
scores = np.array(deeplift_contribs_func(task_idx=0,
	input_data_list=X,
	batch_size=100,
	input_references_list=[np.array([0])[None,:],np.array([background['A'],background['C'],background['G'],background['T']])[None,:,None,None]],
	progress_update=1000))

out_dir='../processed_data/concatenation/concat.regress.deeplift/'
if not os.path.exists(out_dir): os.makedirs(out_dir)
np.save('%s/seq_deeplift.npy'%out_dir,scores)
scores=np.load('%s/seq_deeplift.npy'%out_dir)
scores=np.squeeze(np.sum(scores, axis=1),axis=2)

idx=0
scores_for_idx=scores[idx]
original_onehot = full_data['seq'][idx]
scores_for_idx=original_onehot*scores_for_idx[None,:,None]
plt.ion()
viz_sequence.plot_weights(scores_for_idx, subticks_frequency=100)
plt.savefig('%s/deeplift_seq_0.pdf'%fig_dir)


# Gradient times input:
deeplift_model = kc.convert_functional_model(model,
                    nonlinear_mxts_mode=deeplift.blobs.NonlinearMxtsMode.Gradient)


# deeplift_model.get_name_to_blob().keys() 
#Specify the index of the layer to compute the importance scores of.
find_scores_layer_name = 'seq_input'

#Compile the function that computes the contribution scores
deeplift_contribs_func = deeplift_model.get_target_contribs_func(find_scores_layer_name=find_scores_layer_name,
	pre_activation_target_layer_name='rgs_output')

#Compute scores on inputs
X=[full_data['reg'],full_data['seq']]
background = OrderedDict([('A', 0.31), ('C', 0.19), ('G', 0.19), ('T', 0.31)])
scores = np.array(deeplift_contribs_func(task_idx=0,
	input_data_list=X,
	batch_size=100,
	input_references_list=[np.array([0])[None,:],np.array([background['A'],background['C'],background['G'],background['T']])[None,:,None,None]],
	progress_update=1000))
scores=np.squeeze(np.sum(scores, axis=1),axis=2)
out_dir='../processed_data/concatenation/concat.regres.deeplift/'
np.save('%s/seq_deeplift.npy'%out_dir,scores)
idx=np.unravel_index(np.argmax(scores),dims=scores.shape)[0]
scores_for_idx=scores[idx]
original_onehot = full_data['seq'][idx]
scores_for_idx=original_onehot*scores_for_idx[None,:,None]
plt.ion()
viz_sequence.plot_weights(scores_for_idx, subticks_frequency=100)
plt.savefig('%s/deeplift_seq_grad-times-inp.%s.pdf'%(fig_dir,idx))


scores2=np.reshape(scores,(6107,-1,1000))
scores3=np.sum(scores2,axis=1)
idx=np.unravel_index(np.argmax(scores3),dims=scores3.shape)[0]


scores_for_idx=scores3[idx]
original_onehot = full_data['seq'][idx]
scores_for_idx=original_onehot*scores_for_idx[None,:,None]
plt.ion()
viz_sequence.plot_weights(scores_for_idx, subticks_frequency=100)
plt.savefig('%s/deeplift_seq_sum-173-exp.%s.pdf'%(fig_dir,idx))



scores4=np.sum(scores2,axis=(0,1))

scores_for_idx=scores4
original_onehot = full_data['seq'][idx]
scores_for_idx=original_onehot*scores_for_idx[None,:,None]

plt.ion()
viz_sequence.plot_weights(scores_for_idx, subticks_frequency=100)
plt.savefig('%s/deeplift_seq_sum-173-exp-6107-genes.pdf'%(fig_dir))
