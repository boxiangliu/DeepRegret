import keras
from keras import backend as K
from keras.models import Model
from keras.layers import Input, Dense, Convolution2D, Flatten, Dropout, Merge, Activation, MaxPooling2D

import tensorflow as tf

from cleverhans.utils_tf import _FlagsWrapper
from cleverhans.utils import batch_indices
from cleverhans.attacks_tf import fgsm

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

import math
import six
import time
import pickle
import numpy as np
import sys
import os
sys.path.append('utils/')
import input_data_config as input_data
from callbacks import BatchHistory


dir_suffix='/adversarial/concatenation/regression/'
fig_dir='../figures/%s'%(dir_suffix)
log_dir='../logs/%s'%(dir_suffix)
out_dir='../processed_data/%s'%(dir_suffix)

if not os.path.exists(fig_dir): os.makedirs(fig_dir)
if not os.path.exists(log_dir): os.makedirs(log_dir)
if not os.path.exists(out_dir): os.makedirs(out_dir)


def concatenation_model(num_reg=472, seq_length=1000, units=512, filters=[256], 
	kernel_width=[19],pool_size=[16],pool_stride=[14],dropout_rate=0.5):

	print('INFO - %s'%('building regression model.'))
	reg_input=Input(shape=(num_reg,),dtype='float32',name='reg_input')
	reg_output=Dense(units,activation='relu')(reg_input)


	print('INFO - %s'%('building sequence model.'))
	seq_input=Input(shape=(4,seq_length,1),dtype='float32',name='seq_input')
	x=Convolution2D(nb_filter=filters[0],nb_row=4,nb_col=kernel_width[0],subsample=(1,1),border_mode='valid')(seq_input)
	x=Activation('relu')(x)
	x=MaxPooling2D(pool_size=(1,pool_size[0]),strides=(1,pool_stride[0]),border_mode='valid')(x)
	seq_output=Flatten()(x)

	print('INFO - %s'%('building concatenate model.'))
	x=Merge(mode='concat',concat_axis=1)([reg_output,seq_output])
	x=Dense(units,activation='relu')(x)
	x=Dropout(dropout_rate,seed=42)(x)
	x=Dense(units,activation='relu')(x)
	x=Dropout(dropout_rate,seed=42)(x)
	rgs_output=Dense(1,activation='linear',name='rgs_output')(x)


	model=Model(input=[reg_input,seq_input],output=[rgs_output])

	return(model)

def model_eval(sess, x_reg, x_seq, y, predictions, X_reg_val, X_seq_val, 
	Y_val, args, feed=None):
	"""
	Compute the accuracy of a TF model on some data
	:param sess: TF session to use when training the graph
	:param x: input placeholder
	:param y: output placeholder (for labels)
	:param predictions: model output predictions
	:param X_test: numpy array with training inputs
	:param Y_test: numpy array with training outputs
	:param feed: An optional dictionary that is appended to the feeding
			 dictionary before the session runs. Can be used to feed
			 the learning phase of a Keras model for instance.
	:param args: dict or argparse `Namespace` object.
				 Should contain `batch_size`
	:param model: (deprecated) if not None, holds model output predictions
	:return: a float with the accuracy value
	"""
	args = _FlagsWrapper(args or {})

	assert args.batch_size, "Batch size was not given in args dict"
	
	# Define accuracy symbolically:
	acc_value = tf.reduce_mean(tf.square(y-predictions))

	# Init result var
	accuracy = 0.0

	with sess.as_default():
		# Compute number of batches
		nb_batches = int(math.ceil(float(len(Y_val)) / args.batch_size))
		assert nb_batches * args.batch_size >= len(Y_val)

		for batch in range(nb_batches):
			if batch % 100 == 0 and batch > 0:
				print("Batch " + str(batch))

			# Must not use the `batch_indices` function here, because it
			# repeats some examples.
			# It's acceptable to repeat during training, but not eval.
			start = batch * args.batch_size
			end = min(len(Y_val), start + args.batch_size)
			cur_batch_size = end - start

			# The last batch may be smaller than all others, so we need to
			# account for variable batch size here
			feed_dict = {
				x_reg: X_reg_val[start:end], 
				x_seq: X_seq_val[start:end], 
				y: Y_val[start:end],
				K.learning_phase(): 0
			}

			if feed is not None:
				feed_dict.update(feed)
			cur_acc = acc_value.eval(feed_dict=feed_dict)

			accuracy += (cur_batch_size * cur_acc)

		assert end >= len(Y_val)

		# Divide by number of examples to get final value
		accuracy /= len(Y_val)

	return accuracy

def model_loss(y, model, mean=True, loss='mean_squared_error'):
	"""
	Define loss of TF graph
	:param y: correct labels
	:param model: output of the model
	:param mean: boolean indicating whether should return mean of loss
				 or vector of losses for each input of the batch
	:return: return mean of loss if True, otherwise return vector with per
			 sample loss
	"""
	if loss == 'mean_squared_error':
		out = tf.losses.mean_squared_error(labels=y,predictions=model)
	elif loss == 'categorical_crossentropy':
		pass 
	else: 
		raise ValueError(loss + ' loss not implemented!')
 
	if mean:
		out = tf.reduce_mean(out)
	return out

def model_train(sess, x_reg, x_seq, y, predictions, X_reg_train, X_seq_train, Y_train, 
	X_reg_val, X_seq_val, Y_val, predictions_adv=None, init_all=True, 
	verbose=True, feed=None, args=None):
	"""
	Train a TF graph
	:param sess: TF session to use when training the graph
	:param x: input placeholder, can be a dict for multiple inputs
	:param y: output placeholder (for labels)
	:param predictions: model output predictions
	:param X_train: numpy array with training inputs
	:param Y_train: numpy array with training outputs
	:param save: boolean controlling the save operation
	:param predictions_adv: if set with the adversarial example tensor,
							will run adversarial training
	:param init_all: (boolean) If set to true, all TF variables in the session
					 are (re)initialized, otherwise only previously
					 uninitialized variables are initialized before training.
	:param evaluate: function that is run after each training iteration
					 (typically to display the test/validation accuracy).
	:param verbose: (boolean) all print statements disabled when set to False.
	:param feed: An optional dictionary that is appended to the feeding
				 dictionary before the session runs. Can be used to feed
				 the learning phase of a Keras model for instance.
	:param args: dict or argparse `Namespace` object.
				 Should contain `nb_epochs`, `learning_rate`,
				 `batch_size`
				 If save is True, should also contain 'train_dir'
				 and 'filename'
	:return: True if model trained
	"""
	args = _FlagsWrapper(args or {})

	# Check that necessary arguments were given (see doc above)
	assert args.nb_epochs, "Number of epochs was not given in args dict"
	assert args.learning_rate, "Learning rate was not given in args dict"
	assert args.batch_size, "Batch size was not given in args dict"

	# Define loss
	loss = model_loss(y, predictions)
	if predictions_adv is not None:
		loss = (loss + model_loss(y, predictions_adv)) / 2

	train_step = tf.train.GradientDescentOptimizer(learning_rate=args.learning_rate)
	train_step = train_step.minimize(loss)

	with sess.as_default():
		if init_all:
			tf.global_variables_initializer().run()
		else:
			initialize_uninitialized_global_variables(sess)
		
		for epoch in six.moves.xrange(args.nb_epochs):
			if verbose:
				print("Epoch " + str(epoch))

			# Compute number of batches
			nb_batches = int(math.ceil(float(len(Y_train)) / args.batch_size))
			assert nb_batches * args.batch_size >= len(Y_train)

			prev = time.time()
			for batch in range(nb_batches):

				# Compute batch start and end indices
				start, end = batch_indices(batch, len(Y_train), args.batch_size)

				# Perform one training step
				feed_dict = {
					x_reg: X_reg_train[start:end], 
					x_seq: X_seq_train[start:end], 
					y: Y_train[start:end],
					K.learning_phase(): 1
				}

				if feed is not None:
					feed_dict.update(feed)

				train_step.run(feed_dict=feed_dict)
			assert end >= len(Y_train)  # Check that all examples were used
			cur = time.time()
			if verbose:
				print("\tEpoch took " + str(cur - prev) + " seconds")
			prev = cur

			eval_params = {'batch_size': 100}
			eval_accuracy=model_eval(sess, x_reg, x_seq, y, predictions, X_reg_val, X_seq_val, Y_val, args=eval_params)
			save_path = os.path.join(args.train_dir, args.filename)
			save_path = "%s.%s_%.04f.ckpt"%(save_path, epoch, eval_accuracy)
			saver = tf.train.Saver()
			saver.save(sess, save_path)
			print("Completed model training and saved at: " + str(save_path))


	return True



num_reg=472
seq_length=1000
# model=concatenation_model(num_reg,seq_length)

x_reg=tf.placeholder(tf.float32, shape=(None,num_reg))
x_seq=tf.placeholder(tf.float32, shape=(None,4,seq_length,1))
y=tf.placeholder(tf.float32, shape=(None,1))

predictions=model([x_reg,x_seq])

train,val,test=input_data.read_data_sets(train_pct=80,val_pct=10,test_pct=10)
for data in [train,val,test]:
	data['expr']=np.reshape(data['expr'], (data['expr'].shape[0],1))

sess_legit=tf.Session()
train_params = {
	'nb_epochs': 50,
	'batch_size': 100,
	'learning_rate': 0.01,
	'train_dir': log_dir,
	'filename': 'model_legit'
}

model_train(sess_legit, x_reg, x_seq, y, predictions, 
	train['reg'], train['seq'], train['expr'], 
	val['reg'], val['seq'], val['expr'], args=train_params)

pred=model.predict({'seq_input':test['seq'],'reg_input':test['reg']},batch_size=100,verbose=1)
plt.scatter(pred,test['expr'])
plt.savefig("%s/pred_vs_obs.legit.png"%(fig_dir))
output=np.column_stack((test['expr'], pred[:,0]))
np.savetxt("%s/prediction.legit.txt"%(out_dir), output,delimiter='\t')


# Adversarial training
sess_adv=tf.Session()
model_2=concatenation_model(num_reg,seq_length)
predictions_2 = model_2([x_reg,x_seq])
adv_x_seq_2 = fgsm(x_seq, predictions_2, eps=0.3)
predictions_2_adv = model_2([x_reg,adv_x_seq_2])
train_params = {
	'nb_epochs': 50,
	'batch_size': 100,
	'learning_rate': 0.01,
	'train_dir': log_dir,
	'filename': 'model_adv'
}

model_train(sess_adv, x_reg, x_seq, y, predictions_2, 
	train['reg'], train['seq'], train['expr'], 
	val['reg'], val['seq'], val['expr'], 
	predictions_adv=predictions_2_adv, args=train_params)


pred_2=model_2.predict({'seq_input':test['seq'],'reg_input':test['reg']},batch_size=100,verbose=1)
plt.scatter(pred_2,test['expr'])
plt.savefig("%s/pred_vs_obs.adv.png"%(fig_dir))
output=np.column_stack((test['expr'], pred_2[:,0]))
np.savetxt("%s/prediction.adv.txt"%(out_dir), output,delimiter='\t')
