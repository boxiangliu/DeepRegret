from keras.models import Model
from keras.models import load_model
from keras.layers import Input, Dense, Convolution2D, Flatten, Dropout, Merge, Activation, MaxPooling2D
import numpy as np
import pandas as pd
import sys,os
sys.path.append('utils/')
import input_data_config as input_data 
from input_data import one_hot
from subprocess import call
import matplotlib.pyplot as plt
import copy
from scipy.stats import mannwhitneyu

# Constants: 
num_reg=472
seq_length=1000
dir_suffix='/single_layer/ko_motif/'
out_dir='../processed_data/%s'%dir_suffix
fig_dir='../figures/%s'%dir_suffix
if not os.path.exists(out_dir): os.makedirs(out_dir)
if not os.path.exists(fig_dir): os.makedirs(fig_dir)
model_fn='../logs/single_layer/model.29-0.2093.hdf5'
tomtom_fn='../processed_data/single_layer/interpret/TomTom/tomtom.txt'
regulation_fn='../processed_data/single_layer/ko_motif/RegulationMatrix_Documented_2013927.csv'


# Read data: 
model=load_model(model_fn)
regulation=pd.read_csv(regulation_fn,sep=';')
tomtom=pd.read_csv(tomtom_fn,sep='\t')


# Get the top TF per filter from the TomTom output:
tomtom.rename(columns={'#Query ID':"Query ID"},inplace=True)
top=tomtom.groupby('Query ID').agg({'p-value':'min'})
top['Query ID']=top.index
top=pd.merge(top, tomtom, on=['Query ID','p-value'])
top['filter']=top['Query ID'].apply(lambda x: int(x[6:]))
top['TF']=top['Target ID'].apply(lambda x: x.split('&')[0])

# Prediction from original model:
train,val,test=input_data.read_data_sets(train_pct=80,val_pct=10,test_pct=10)
pred=model.predict({'seq_input':test['seq'],'reg_input':test['reg']},batch_size=100,verbose=1)
convolution2d_1=model.get_layer('convolution2d_1').get_weights()

# Setting convolutional filter to zero:
filter=list()
overall_mean=list()
target_mean=list()
overall_non_zero_prop=list()
target_non_zero_prop=list()
overall_non_zero_mean=list()
target_non_zero_mean=list()



for i in xrange(top.shape[0]):
	filter_id=top['filter'][i]
	print('filter%s'%filter_id)

	# Setting a filter to zero and predict:
	orig=copy.copy(convolution2d_1[0][:,:,:,filter_id])
	convolution2d_1[0][:,:,:,filter_id]=0.0

	model.get_layer('convolution2d_1').set_weights(convolution2d_1)
	pred_ko=model.predict({'seq_input':test['seq'],'reg_input':test['reg']},batch_size=100,verbose=1)

	convolution2d_1[0][:,:,:,filter_id]=orig

	diff=abs(pred-pred_ko)



	# Get row index of the TF:
	TF=top['TF'][i]
	row=int(np.where(regulation.TF==TF)[0])

	# Get index for target gene of the TF:
	target_genes=regulation.iloc[row,np.where(regulation.iloc[row,:])[0].tolist()].index.tolist()
	target_genes.remove('TF')
	target_genes=list(set(target_genes))
	target_gene_idx=np.where(test['name'].isin(target_genes))[0]

	filter.append(filter_id)

	overall_mean.append(np.mean(diff))
	target_mean.append(np.mean(diff[target_gene_idx]))

	overall_non_zero_prop.append(np.mean(diff>0))
	target_non_zero_prop.append(np.mean(diff[target_gene_idx]>0))

	overall_non_zero_mean.append(np.mean(diff[diff>0]))
	target_non_zero_mean.append(np.mean(diff[target_gene_idx][diff[target_gene_idx]>0]))

data=pd.DataFrame({'filter':filter,
	'overall_mean':overall_mean,
	'target_mean':target_mean,
	'overall_non_zero_prop':overall_non_zero_prop,
	'target_non_zero_prop':target_non_zero_prop,
	'overall_non_zero_mean':overall_non_zero_mean,
	'target_non_zero_mean':target_non_zero_mean})

mannwhitneyu(data['overall_mean'], data['target_mean'])


# Setting Azf1p motif to zero and predict:
filter_id=164
orig=copy.copy(convolution2d_1[0][:,:,:,filter_id])
convolution2d_1[0][:,:,:,filter_id]=0.0

model.get_layer('convolution2d_1').set_weights(convolution2d_1)
pred_ko=model.predict({'seq_input':test['seq'],'reg_input':test['reg']},batch_size=100,verbose=1)
diff=abs(pred-pred_ko)

convolution2d_1[0][:,:,:,filter_id]=orig

plt.ion()
plt.scatter(pred, pred_ko)
plt.scatter(pred[top_idx[[0,1,4]]], pred_ko[top_idx[[0,1,4]]])
plt.text(pred[top_idx[0]],pred_ko[top_idx[[0]]],'YGR137W')
plt.text(pred[top_idx[1]],pred_ko[top_idx[[1]]],'YFR053C/HXK1')
plt.text(pred[top_idx[4]],pred_ko[top_idx[[4]]]-0.5,'YML128C/MSC1')
plt.xlim(-7,11)
plt.savefig('%s/mig1p_filter164.png'%fig_dir)