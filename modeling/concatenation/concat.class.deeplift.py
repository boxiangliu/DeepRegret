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
import os

from sklearn.cluster.bicluster import SpectralBiclustering
from sklearn.metrics import consensus_score
from scipy.stats import ranksums
from scipy.stats import ttest_ind

# Load model:
model_fn='../logs/concatenation/classification/model.32-0.5284.hdf5'
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
                    nonlinear_mxts_mode=deeplift.blobs.NonlinearMxtsMode.DeepLIFT_GenomicsDefault)

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
	pre_activation_target_layer_name='preact_cls_output')
#You can also provide an array of indices to find_scores_layer_idx to get scores for multiple layers at once

#compute scores on inputs
#input_data_list is a list containing the data for different input layers
#eg: for MNIST, there is one input layer with with dimensions 1 x 28 x 28
#In the example below, let X be an array with dimension n x 1 x 28 x 28 where n is the number of examples
#task_idx represents the index of the node in the output layer that we wish to compute scores.
#Eg: if the output is a 10-way softmax, and task_idx is 0, we will compute scores for the first softmax class
X=[full_data['reg'],full_data['seq']]
scores = np.array(deeplift_contribs_func(task_idx=0,
	input_data_list=X,
	batch_size=10,
	progress_update=1000))

out_dir='../processed_data/concatenation/concat.class.deeplift/'
if not os.path.exists(out_dir): os.makedirs(out_dir)
np.savetxt('%s/reg_deeplift.txt'%out_dir,scores,delimiter='\t',fmt='%s')
scores=np.genfromtxt('%s/reg_deeplift.txt'%out_dir,delimiter='\t')

scores2=abs(scores)
colnames=pd.read_csv('../data/reg_names_R.txt',header=None)
deeplift=pd.DataFrame(scores2,
	index=range(scores2.shape[0]),
	columns=colnames[0].tolist())
deeplift['uid']=full_data['uid']
deeplift['name']=full_data['name']
deeplift['experiment']=full_data['experiment']


deeplift_sum=deeplift.iloc[:,0:472].sum(axis=0)
deeplift_sum_rank=deeplift_sum.rank(ascending=False)
deeplift_sum_rank[deeplift_sum_rank<=10]
deeplift_sum_rank[deeplift_sum_rank.index=='YMR037C']
deeplift_sum_rank[deeplift_sum_rank.index=='YKL062W']
deeplift_sum.plot.hist()

fig_dir='../figures/concatenation/concat.class.deeplift/'
if not os.path.exists(fig_dir): os.makedirs(fig_dir)
plt.savefig('%s/deeplift_sum_hist.pdf'%fig_dir)


msn2_downstream=deeplift[['YMR037C','uid','experiment']]
msn2_downstream=msn2_downstream.pivot(index='experiment',columns='uid',values='YMR037C')
msn2_downstream_colsum=msn2_downstream.sum(axis=0)

msn2_downstream_colsum_rank=msn2_downstream_colsum.rank(ascending=False)

msn2_targets=pd.read_csv('../data/sgd/MSN2_targets.txt',comment='!',delimiter='\t')
y=[]
for x in msn2_downstream_colsum_rank.index.tolist():
	temp=x in msn2_targets['Target Systematic Name'].tolist()
	y.append(temp)

z=[not i for i in y]
ttest_ind(msn2_downstream_colsum[y],msn2_downstream_colsum[z])
# Ttest_indResult(statistic=3.1641265733701611, pvalue=0.0015631278619146669)

