from keras.models import Model
from keras.models import load_model
from keras.layers import Input, Dense, Convolution2D, Flatten, Dropout, Merge, Activation, MaxPooling2D
import numpy as np
import pandas as pd
import sys,os
sys.path.append('utils/')
from input_data import one_hot
from subprocess import call
import matplotlib.pyplot as plt


# Functions: 
def get_seq_data(seq_file='../data/yeast_promoters.txt'):
	sequence = pd.read_table(seq_file, names=["UID", "sequence"])
	sequence.loc[:, "one_hot_sequence"] = [one_hot(seq) for seq in sequence.loc[:, "sequence"]]
	seq_one_hot=np.reshape(np.vstack(sequence['one_hot_sequence']),[-1,4,1000,1])
	return sequence,seq_one_hot

def output_fasta_by_threshold(convolution2d_1_act,seq_data,sequence,threshold,filters_dir):
	if not os.path.exists(filters_dir): os.makedirs(filters_dir)
	n_filters=convolution2d_1_act.shape[3]
	for i in xrange(n_filters):
		# Get max activation:
		max_act=convolution2d_1_act[:,:,:,i].max()

		# Get seq. with activation greater than 50% of max: 
		seq_idx=np.where(convolution2d_1_act[:,:,:,i]>=threshold*max_act)
		print('INFO - filter %d'%(i))

		with open('%s/filter%.03d.fa'%(filters_dir,i),'w') as f:
			n_seq=len(seq_idx[0])
			for j in xrange(n_seq):
				promoter_idx=seq_idx[0][j]
				start=seq_idx[2][j]
				end=start+kernel_width[0]
				out=sequence['sequence'][promoter_idx][start:end]
				f.write('>\n'+out+'\n')

def output_fasta_by_topN(convolution2d_1_act,seq_data,sequence,topN,filters_dir):
	if not os.path.exists(filters_dir): os.makedirs(filters_dir)
	n_filters=convolution2d_1_act.shape[3]
	for i in xrange(n_filters):
		# Get max activation:
		max_act=convolution2d_1_act[:,:,:,i].max()

		# Get top 100 sequences with largest activations:
		filt_act=convolution2d_1_act[:,:,:,i]
		seq_idx=np.unravel_index(np.argsort(filt_act.ravel())[-topN:], filt_act.shape)
		print('INFO - filter %d'%(i))

		with open('%s/filter%.03d.fa'%(filters_dir,i),'w') as f:
			n_seq=len(seq_idx[0])
			for j in xrange(n_seq):
				promoter_idx=seq_idx[0][j]
				start=seq_idx[2][j]
				end=start+kernel_width[0]
				out=sequence['sequence'][promoter_idx][start:end]
				f.write('>\n'+out+'\n')


def output_pwm_by_threshold(convolution2d_1_act,seq_data,sequence,threshold,out_fn):
	n_filters=convolution2d_1_act.shape[3]
	with open(out_fn,'w') as f:
		for i in xrange(n_filters):
			# Get max activation:
			max_act=convolution2d_1_act[:,:,:,i].max()

			# Get seq. with activation greater than 50% of max: 
			seq_idx=np.where(convolution2d_1_act[:,:,:,i]>=threshold*max_act)
			print('INFO - filter %d'%(i))

			pwm=np.zeros(shape=(4,kernel_width[0]),dtype='int64')
			n_seq=len(seq_idx[0])
			for j in xrange(n_seq):
				promoter_idx=seq_idx[0][j]
				start=seq_idx[2][j]
				end=start+kernel_width[0]
				pwm+=seq_data[promoter_idx,:,start:end,0]
			f.write('>filter%.03d\n'%(i))
			np.savetxt(f,pwm,fmt='%d',delimiter='\t')

def output_pwm_by_topN(convolution2d_1_act,seq_data,sequence,topN,out_fn):
	n_filters=convolution2d_1_act.shape[3]
	with open(out_fn,'w') as f:
		for i in xrange(n_filters):
			# Get top 100 sequences with largest activations:
			filt_act=convolution2d_1_act[:,:,:,i]
			seq_idx=np.unravel_index(np.argsort(filt_act.ravel())[-topN:], filt_act.shape)
			
			print('INFO - filter %d'%(i))

			pwm=np.zeros(shape=(4,kernel_width[0]),dtype='int64')
			n_seq=len(seq_idx[0])
			for j in xrange(n_seq):
				promoter_idx=seq_idx[0][j]
				start=seq_idx[2][j]
				end=start+kernel_width[0]
				pwm+=seq_data[promoter_idx,:,start:end,0]
			f.write('>filter%.03d\n'%(i))
			np.savetxt(f,pwm,fmt='%d',delimiter='\t')

def information_content(pwm_fn):
	ic=np.zeros((256))
	with open(pwm_fn,'r') as f:
		i=0
		for line in f:
			if line.startswith('>'):
				row=0
				pwm=np.zeros((4,19))
			else:
				pwm[row,:]=line.strip().split('\t')
				if row==3:
					pwm=pwm/pwm.sum(axis=0)[None,:]
					pwm=pwm.ravel()
					pwm=pwm[pwm!=0]
					tmp=sum(pwm*np.log2(pwm))-sum([0.25]*4*19*np.log2([0.25]*4*19))
					ic[i]=tmp
					i+=1
				else:
					row+=1
	return pd.DataFrame(ic)

# Constants: 
num_reg=472
seq_length=1000
dir_suffix='single_layer/interpret.paper/'
filters_dir='../processed_data/%s/filters/'%(dir_suffix)
pwm_dir='../processed_data/%s/pwm/'%(dir_suffix)
fig_dir='../figures/%s/filters/'%(dir_suffix)

if not os.path.exists(filters_dir): os.makedirs(filters_dir)
if not os.path.exists(pwm_dir): os.makedirs(pwm_dir)
if not os.path.exists(fig_dir): os.makedirs(fig_dir)

# Get fitted model: 
fn='../logs/single_layer/model/model.44-0.2131.hdf5'
model=load_model(fn)



# Define model with only first conv layer:
filters=[256]
kernel_width=[19]
seq_input=Input(shape=(4,seq_length,1),dtype='float32',name='seq_input')
x=Convolution2D(nb_filter=filters[0],nb_row=4,nb_col=kernel_width[0],subsample=(1,1),border_mode='valid',weights=model.get_layer('convolution2d_1').get_weights())(seq_input)
inter_output=Activation('relu')(x)
model2=Model(input=[seq_input],output=[inter_output])


# Obtain activation of the first conv layer:
sequence,seq_one_hot=get_seq_data()
convolution2d_1_act=model2.predict({'seq_input':seq_one_hot},batch_size=100,verbose=1)


# Get the top 100 sequences with the largest activation:
# value of each filter:
pwm_fn='%s/filter_top_100.pwm'%(pwm_dir)
output_pwm_by_topN(convolution2d_1_act, seq_one_hot, sequence, 100, pwm_fn)
output_fasta_by_topN(convolution2d_1_act, seq_one_hot, sequence, 100, filters_dir)


# Caculate information content:
ic=information_content(pwm_fn)
ic.to_csv('%s/information_content.txt'%pwm_dir,index=True,header=False,sep='\t')