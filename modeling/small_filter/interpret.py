from keras.models import Model
from keras.models import load_model
from keras.layers import Input, Dense, Convolution2D, Flatten, Dropout, Merge, Activation, MaxPooling2D
import numpy as np
import pandas as pd
import sys,os
sys.path.append('/srv/scratch/bliu2/deepregret/scripts/modeling/keras1_small_filter/')
import input_data
from input_data import one_hot
from subprocess import call


# Functions: 
def get_seq_data(seq_file='../data/yeast_promoters.txt'):
	promoters = pd.read_table(seq_file, names=["UID", "sequence"])
	promoters.loc[:, "one_hot_sequence"] = [one_hot(seq) for seq in promoters.loc[:, "sequence"]]
	seq=np.reshape(np.vstack(promoters['one_hot_sequence']),[-1,4,1000,1])
	return promoters,seq

def output_fasta(convolution2d_1_act,seq_data,promoters,threshold,filters_dir):
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
				out=promoters['sequence'][promoter_idx][start:end]
				f.write('>\n'+out+'\n')


def output_pwm(convolution2d_1_act,seq_data,promoters,threshold,out_fn):
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


# Constants: 
num_reg=472
seq_length=1000
filters_dir='../processed_data/keras1_small_filter/interpret/filters/'
pwm_dir='../processed_data/keras1_small_filter/interpret/pwm/'
fig_dir='../figures/keras1_small_filter/interpret/filters/'
if not os.path.exists(filters_dir): os.makedirs(filters_dir)
if not os.path.exists(pwm_dir): os.makedirs(pwm_dir)
if not os.path.exists(fig_dir): os.makedirs(fig_dir)

# Get fitted model: 
fn='../logs/small_filter/model.29-0.1920.hdf5'
model=load_model(fn)



# Define model with only first conv layer:
filters=[256,128]
kernel_width=[19,11]
seq_input=Input(shape=(4,seq_length,1),dtype='float32',name='seq_input')
x=Convolution2D(nb_filter=filters[0],nb_row=4,nb_col=kernel_width[0],subsample=(1,1),border_mode='valid',weights=model.get_layer('convolution2d_1').get_weights())(seq_input)
inter_output=Activation('relu')(x)
model2=Model(input=[seq_input],output=[inter_output])


# Obtain activation of the first conv layer:
promoters,seq_data=get_seq_data()
convolution2d_1_act=model2.predict({'seq_input':seq_data},batch_size=100,verbose=1)


# Get the sequences that activates more than 50% of maximum
# value of each filter:
output_fasta(convolution2d_1_act, seq_data, promoters, 0.5, filters_dir='../processed_data/keras1_small_filter/interpret/filters/')
output_pwm(convolution2d_1_act, seq_data, promoters, 0.5, '%s/filter_0.5.pwm'%(pwm_dir))
output_pwm(convolution2d_1_act, seq_data, promoters, 0.7, '%s/filter_0.7.pwm'%(pwm_dir))
output_pwm(convolution2d_1_act, seq_data, promoters, 0.9, '%s/filter_0.9.pwm'%(pwm_dir))


# Make sequence logos: 
for i in xrange(n_filters):
	print('INFO - filter %d'%(i))
	cmd='weblogo -f %s/filter%.03d.fa -o %s/filter%.03d.eps -c classic'%(filters_dir,i,fig_dir,i)
	call(cmd,shell=True)