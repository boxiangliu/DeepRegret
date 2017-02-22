import tensorflow as tf


# Constants: 
NUM_REG = 472
SEQ_LENGTH = 1000


# helper function to build convolutional layer with relu activation: 
def conv_relu(input, kernal_shape, bias_shape):
	weights=tf.get_variable("weights", kernal_shape,initializer=tf.contrib.layers.xavier_initializer())
	biases=tf.get_variable("biases", bias_shape,initializer=tf.constant_initializer(0.0))
	conv=tf.nn.conv1d(input,weights,stride=4, padding='VALID')
	relu=tf.nn.relu(conv+biases)
	return(relu)


# build sequence model:
def build_sequence_model(seq):
	'''sequence model consists of 2 convolutional layers 
	with ReLU activations. 
	the first convolutional layers uses 32 filters with size 4x100;
	the second convolutional layer uses 256 filters with size 32x15.
	'''
	# first conv layer: 
	with tf.variable_scope('conv1'):
		relu1=conv_relu(seq,[100,4,32],[32])

	# second conv layer:
	with tf.variable_scope('conv2'):
		relu2=conv_relu(relu1,[15,32,256],[256])

	# return: 
	return relu2

# helper function to construct fully connected layers with ReLU activation:
def fully_connected_relu(input,weight_shape,bias_shape):
	weights=tf.get_variable('weights',weight_shape,initializer=tf.contrib.layers.xavier_initializer())
	biases=tf.get_variable('biases',bias_shape,initializer=tf.constant_initializer(0.0))
	return tf.nn.relu(tf.matmul(input,weights)+biases)

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


# build softmax prediction model:
def classification(concat_model,hidden_layer_size=512):
	with tf.variable_scope('class_hidden1'):
		class_hidden1=fully_connected_relu(concat_model, [concat_model.get_shape().as_list()[1],hidden_layer_size], [hidden_layer_size])

	with tf.variable_scope('class_hidden2'):
		class_hidden2=fully_connected_relu(class_hidden1, [hidden_layer_size,hidden_layer_size], [hidden_layer_size])

	with tf.variable_scope('dropout'):
		# keep_prob = tf.placeholder(tf.float32)
		keep_prob = 0.5
		dropout=tf.nn.dropout(class_hidden2,keep_prob)

	with tf.variable_scope('class_output'):
		logits=fully_connected_relu(dropout, [hidden_layer_size,2], [2])

	return(logits)


# build entire inference graph: 
def inference(seq,regulator_expression,batch_size):
	seq_model=build_sequence_model(seq)
	regulator_model=build_regulator_model(regulator_expression)
	concat_model=concatenate(seq_model,regulator_model,batch_size)
	logits=classification(concat_model)
	return(logits)


# define loss function:
def loss(logits,labels):
	cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,labels=labels,name='xentropy')
	return tf.reduce_mean(cross_entropy,name='xentropy_mean')


# define training function: 
def training(loss,learning_rate):
	# Add a scalar summary for the snapshot loss.
	tf.summary.scalar('loss', loss)

	# Create the gradient descent optimizer with the given learning rate.
	optimizer = tf.train.GradientDescentOptimizer(learning_rate)

	# Create a variable to track the global step.
	global_step = tf.Variable(0, name='global_step', trainable=False)

	# Use the optimizer to apply the gradients that minimize the loss
	# (and also increment the global step counter) as a single training step.
	train_op = optimizer.minimize(loss, global_step=global_step)
	return train_op


# define evaluation function: 
def evaluation(logits, labels):
	"""Evaluate the quality of the logits at predicting the label.

	Args:
	logits: Logits tensor, float - [batch_size, NUM_CLASSES].
	labels: Labels tensor, int32 - [batch_size], with values in the
	  range [0, NUM_CLASSES).

	Returns:
	A scalar int32 tensor with the number of examples (out of batch_size)
	that were predicted correctly.
	"""
	# Correct if label has highest probability:
	correct = tf.nn.in_top_k(logits, labels, 1)

	# Return the number of true entries:
	return tf.reduce_sum(tf.cast(correct, tf.int32))


