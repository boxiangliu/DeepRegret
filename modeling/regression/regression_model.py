# build regression prediction model:
def regression(concat_model,hidden_layer_size=512):
	with tf.variable_scope('reg_hidden1'):
		reg_hidden1=fully_connected_relu(concate_model, [concat_model.get_shape().as_list()[1],hidden_layer_size], [hidden_layer_size])

	with tf.variable_scope('reg_hidden2'):
		reg_hidden2=fully_connected_relu(reg_hidden1, [hidden_layer_size,hidden_layer_size], [hidden_layer_size])

	with tf.variable_scope('dropout'):
		keep_prob = tf.placeholder(tf.float32)
		dropout=tf.nn.dropout(reg_hidden2)

	with tf.variable_scope('reg_output'):
		reg_output=fully_connected_relu(dropout, [hidden_layer_size,1], [1])

	return(reg_output)
