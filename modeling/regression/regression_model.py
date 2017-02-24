import tensorflow as tf
import math

# Constants: 
NUM_REG = 472
SEQ_LENGTH = 1000


# helper function to build convolutional layer with relu activation: 
def conv_relu(input, kernal_shape, bias_shape,stride=4):
	weights=tf.get_variable("weights", kernal_shape,initializer=tf.contrib.layers.xavier_initializer())
	biases=tf.get_variable("biases", bias_shape,initializer=tf.constant_initializer(0.0))
	conv=tf.nn.conv1d(input,weights,stride=stride, padding='VALID')
	relu=tf.nn.relu(conv+biases)
	return(relu)


# build sequence model:
def build_sequence_model(seq,conv1_filter_depth=32,conv2_filter_depth=256,stride=4):
	'''sequence model consists of 2 convolutional layers 
	with ReLU activations. 
	the first convolutional layers uses 32 filters with size 4x100;
	the second convolutional layer uses 256 filters with size 32x15.
	'''
	# first conv layer: 
	with tf.variable_scope('conv1'):
		relu1=conv_relu(seq,[100,4,conv1_filter_depth],[conv1_filter_depth],stride)

	# second conv layer:
	with tf.variable_scope('conv2'):
		relu2=conv_relu(relu1,[15,32,conv2_filter_depth],[conv2_filter_depth],stride)

	relu1_length=math.ceil(float(SEQ_LENGTH)/float(stride))
	relu2_length=math.ceil(float(relu1_length)/float(stride))

	# return: 
	return relu2,relu2_length

# helper function to construct fully connected layers with ReLU activation:
def fully_connected_relu(input,weight_shape,bias_shape):
	weights=tf.get_variable('weights',weight_shape,initializer=tf.contrib.layers.xavier_initializer())
	biases=tf.get_variable('biases',bias_shape,initializer=tf.constant_initializer(0.0))
	return tf.nn.relu(tf.matmul(input,weights)+biases)


# helper function to construct fully connected layers with identity activation:
def fully_connected_identity(input,weight_shape,bias_shape):
	weights=tf.get_variable('weights',weight_shape,initializer=tf.contrib.layers.xavier_initializer())
	biases=tf.get_variable('biases',bias_shape,initializer=tf.constant_initializer(0.0))
	return tf.matmul(input,weights)+biases


# build regulator model: 
def build_regulator_model(regulator_expression,hidden_layer_size=512):
	'''regulator model consists of 3 fully connected layers
	the hidden layers have 512 units each.
	'''
	with tf.variable_scope('hidden1'):
		hidden1=fully_connected_relu(regulator_expression, [NUM_REG,hidden_layer_size], [hidden_layer_size])

	with tf.variable_scope('hidden2'):
		hidden2=fully_connected_relu(hidden1, [hidden_layer_size,hidden_layer_size], [hidden_layer_size])

	with tf.variable_scope('hidden3'):
		hidden3=fully_connected_relu(hidden2, [hidden_layer_size,hidden_layer_size], [hidden_layer_size])

	return hidden3

# concatenate sequence and regulator model: 
def concatenate(sequence_model,regulator_model,batch_size):
	return tf.concat([tf.reshape(sequence_model,[batch_size,-1]),regulator_model],1,name='concat')


# build regression prediction model:
def regression(concat_model,keep_prob,hidden_layer_size=512):
	with tf.variable_scope('reg_hidden1'):
		reg_hidden1=fully_connected_relu(concat_model, [concat_model.get_shape().as_list()[1],hidden_layer_size], [hidden_layer_size])

	with tf.variable_scope('reg_hidden2'):
		reg_hidden2=fully_connected_relu(reg_hidden1, [hidden_layer_size,hidden_layer_size], [hidden_layer_size])

	with tf.variable_scope('dropout'):
		dropout=tf.nn.dropout(reg_hidden2,keep_prob)

	with tf.variable_scope('reg_output'):
		reg_output=fully_connected_identity(dropout, [hidden_layer_size,1], [1])

	return(tf.transpose(reg_output))

# build entire inference graph: 
def inference(seq,regulator_expression,keep_prob,batch_size):
	conv1_filter_depth=32
	conv2_filter_depth=256
	stride=4
	reg_model_hidden_layer_size=512
	concat_model_hidden_layer_size=512
	seq_model,_=build_sequence_model(seq,conv1_filter_depth,conv2_filter_depth,stride)
	regulator_model=build_regulator_model(regulator_expression,reg_model_hidden_layer_size)
	concat_model=concatenate(seq_model,regulator_model,batch_size)
	pred=regression(concat_model,keep_prob,concat_model_hidden_layer_size)
	return(pred)


# define loss function:
# def loss(y,y_):
# 	huber=tf.cond(tf.abs(y-y_) < 0.5, lambda: tf.square(y-y_), lambda: tf.abs(y-y_))
# 	return tf.reduce_mean(huber,name='huber_mean')
def loss(y,y_):
	se=tf.square(y-y_, name='se')
	return tf.reduce_mean(se,name='mse')


# define training function: 
def training(loss,learning_rate):
	# Create the gradient descent optimizer with the given learning rate.
	optimizer = tf.train.GradientDescentOptimizer(learning_rate)

	# Create a variable to track the global step.
	global_step = tf.Variable(0, name='global_step', trainable=False)

	# Use the optimizer to apply the gradients that minimize the loss
	# (and also increment the global step counter) as a single training step.
	train_op = optimizer.minimize(loss, global_step=global_step)
	return train_op


# define evaluation function: 
def evaluation(y, y_):
	"""Evaluate the quality of the logits at predicting the label.

	Args:
	y: prediction tensor, float - [batch_size].
	y_: observation tensor, int32 - [batch_size].

	Returns:
	A scalar int32 tensor with the sum of squared errors.
	"""
	# Correct if label has highest probability:
	se = tf.square(y-y_)

	# Return the number of true entries:
	return tf.reduce_sum(se)




