from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import sys
import tensorflow as tf

import regression_model
import overexpress_data

def placeholder_inputs(batch_size):
	"""
	Args:
	  batch_size: The batch size will be baked into both placeholders.

	Returns:
	  seq_placeholder: Sequence placeholder.
	  reg_expr_placeholder: Regulator expression placeholder.
	  labels_placeholder: Labels placeholder.
	"""
	# create placeholders: 
	seq_placeholder = tf.placeholder(tf.float32, shape=(batch_size,regression_model.SEQ_LENGTH,4))
	reg_expr_placeholder = tf.placeholder(tf.float32, shape=(batch_size,regression_model.NUM_REG))
	labels_placeholder = tf.placeholder(tf.float32, shape=(batch_size))
	keep_prob = tf.placeholder(tf.float32)
	meta_placeholder = tf.placeholder(tf.string, shape=(batch_size,3))
	return seq_placeholder, reg_expr_placeholder, labels_placeholder, keep_prob, meta_placeholder

def fill_feed_dict(data_set, seq_pl, reg_expr_pl, labels_pl, keep_prob_pl, keep_prob, meta_pl, batch_size):
	"""Construct feed dictionary.
	Args:
		data_set: The set of images and labels, from input_data.read_data_sets()
		seq_pl: The sequence placeholder, from placeholder_inputs().
		reg_expr_pl: the regulator expression placeholder, from placeholder_inputs().
		labels_pl: The labels placeholder, from placeholder_inputs().

	Returns:
		feed_dict: The feed dictionary mapping from placeholders to values.
	"""
	# Create the feed_dict for the placeholders filled with the next `batch size` examples.
	seq_feed, reg_expr_feed, labels_feed, meta_feed = data_set.next_batch(batch_size)
	feed_dict = {seq_pl: seq_feed,reg_expr_pl: reg_expr_feed,labels_pl: labels_feed, keep_prob_pl: keep_prob, meta_pl: meta_feed}
	return feed_dict


def prediction(sess,pred_tensor,seq_placeholder,reg_expr_placeholder,labels_placeholder,keep_prob_pl,meta_placeholder,data_set):
	assert data_set.num_examples % FLAGS.batch_size == 0 
	steps_per_epoch = data_set.num_examples // FLAGS.batch_size
	num_examples = steps_per_epoch * FLAGS.batch_size
	y=np.empty(0)
	y_=np.empty(0)
	meta=np.empty(0)
	for step in xrange(steps_per_epoch):
		feed_dict = fill_feed_dict(data_set,seq_placeholder,reg_expr_placeholder,labels_placeholder,keep_prob_pl,keep_prob=1.0,meta_pl=meta_placeholder,batch_size=FLAGS.batch_size)
		tmp1,tmp2,tmp3 = sess.run([pred_tensor,labels_placeholder,meta_placeholder], feed_dict=feed_dict)
		tmp1=np.transpose(tmp1)
		if y.shape[0]==0:
			y=tmp1
			y_=tmp2
			meta=tmp3
		else: 
			y=np.concatenate((y,tmp1),axis=0)
			y_=np.concatenate((y_,tmp2),axis=0)
			meta=np.concatenate((meta,tmp3),axis=0)
	return np.column_stack((y_,y,meta))


def run_training():
	"""Train for a number of steps."""
	data_sets = overexpress_data.read_data_sets(seq_file=FLAGS.seq_file,expr_file=FLAGS.expr_file,reg_names_file=FLAGS.reg_names_file,fold_change=FLAGS.fold_change)

	# Tell TensorFlow that the model will be built into the default Graph.
	with tf.Graph().as_default():
		# Generate placeholders for the images and labels.
		seq_placeholder, reg_expr_placeholder, labels_placeholder, keep_prob_placeholder, meta_placeholder = placeholder_inputs(FLAGS.batch_size)

		# Build a Graph that computes predictions from the inference model.
		y = regression_model.inference(seq_placeholder,reg_expr_placeholder,keep_prob_placeholder,FLAGS.batch_size)

		# Add the variable initializer Op.
		init = tf.global_variables_initializer()

		# Create a saver for writing training checkpoints.
		saver = tf.train.Saver()

		# Create a session for running Ops on the Graph.
		sess = tf.Session()

		# Add the variable initializer Op.
		init = tf.global_variables_initializer()

		# Create a saver for writing training checkpoints.
		saver = tf.train.Saver()

		# Create a session for running Ops on the Graph.
		sess = tf.Session()

		# And then after everything is built:

		# Run the Op to initialize the variables.
		sess.run(init)

		saver = tf.train.import_meta_graph(FLAGS.graph)
		saver.restore(sess,tf.train.latest_checkpoint(FLAGS.log_dir))

		# Make confusion matrix:
		# print(data_sets.overexpress_data.next_batch())
		pred=prediction(sess,y,seq_placeholder,reg_expr_placeholder,labels_placeholder,keep_prob_placeholder,meta_placeholder,data_sets.overexpress_data)
		np.savetxt(FLAGS.out, pred, fmt=['%.3f','%.3f','%s','%s','%s'],delimiter='\t')

def main(_):
	if tf.gfile.Exists(FLAGS.log_dir):
		pass
	tf.gfile.MakeDirs(FLAGS.log_dir)
	run_training()


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--learning_rate',type=float,default=0.01,help='Initial learning rate.')
	parser.add_argument('--max_steps',type=int,default=200000,help='Number of steps to run trainer.')
	parser.add_argument('--batch_size',type=int,default=473,help='Batch size.  Must divide evenly into the dataset sizes.')
	parser.add_argument('--seq_file',type=str,default='../data/yeast_promoters.txt',help='Path to promoter sequence data.')
	parser.add_argument('--expr_file',type=str,default='../data/complete_dataset.txt',help='Path to expression data.')
	parser.add_argument('--reg_names_file',type=str,default='../data/reg_names_R.txt',help='Path to regulator names.')
	parser.add_argument('--fold_change',type=float,default=0.0,help='Fold change in expression level.')
	parser.add_argument('--log_dir',type=str,default='../processed_data/regression/',help='Directory to put the log data.')
	parser.add_argument('--graph',type=str,default='../processed_data/regression/model.ckpt-199999.meta',help='Directory where the log data is stored.')
	parser.add_argument('--out',type=str,help='Path to write the predictions.')
	FLAGS, unparsed = parser.parse_known_args()
	tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)


