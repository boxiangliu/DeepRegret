from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import sys
import tensorflow as tf

import classification_model
import input_data

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
	seq_placeholder = tf.placeholder(tf.float32, shape=(batch_size,classification_model.SEQ_LENGTH,4))
	reg_expr_placeholder = tf.placeholder(tf.float32, shape=(batch_size,classification_model.NUM_REG))
	labels_placeholder = tf.placeholder(tf.int32, shape=(batch_size))
	keep_prob = tf.placeholder(tf.float32)
	return seq_placeholder, reg_expr_placeholder, labels_placeholder, keep_prob

def fill_feed_dict(data_set, seq_pl, reg_expr_pl, labels_pl, keep_prob_pl, keep_prob, batch_size):
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
	seq_feed, reg_expr_feed, labels_feed = data_set.next_batch(batch_size)
	feed_dict = {seq_pl: seq_feed,reg_expr_pl: reg_expr_feed,labels_pl: labels_feed, keep_prob_pl: keep_prob}
	return feed_dict


def confusion(sess,cm,seq_placeholder,reg_expr_placeholder,labels_placeholder,keep_prob_pl,data_set):
	steps_per_epoch = data_set.num_examples // FLAGS.batch_size
	num_examples = steps_per_epoch * FLAGS.batch_size
	confusion_mat= np.matrix([[0]*3]*3)
	for step in xrange(steps_per_epoch):
		feed_dict = fill_feed_dict(data_set,seq_placeholder,reg_expr_placeholder,labels_placeholder,keep_prob_pl,keep_prob=1.0,batch_size=FLAGS.batch_size)
		confusion_mat += sess.run(cm, feed_dict=feed_dict)
	return confusion_mat

# FLAGS=collections.namedtuple('FLAGS', ['batch_size'])
# FLAGS.batch_size=100
# FLAGS.seq_file='../../../data/yeast_promoters.txt'
# FLAGS.expr_file='../../../data/complete_dataset.txt'
# FLAGS.reg_names_file='../../../data/reg_names_R.txt'
# FLAGS.learning_rate=0.01

def run_training():
	"""Train for a number of steps."""
	data_sets = input_data.read_data_sets(seq_file=FLAGS.seq_file,expr_file=FLAGS.expr_file,reg_names_file=FLAGS.reg_names_file)

	# Tell TensorFlow that the model will be built into the default Graph.
	with tf.Graph().as_default():
		# Generate placeholders for the images and labels.
		seq_placeholder, reg_expr_placeholder, labels_placeholder, keep_prob_pl = placeholder_inputs(FLAGS.batch_size)

		# Build a Graph that computes predictions from the inference model.
		logits = classification_model.inference(seq_placeholder,reg_expr_placeholder,keep_prob_pl,FLAGS.batch_size)

		# Predictions based on the logits: 
		pred = tf.argmax(logits,1)

		# Add to the Graph the Ops for loss calculation.
		loss = classification_model.loss(logits, labels_placeholder)

		# Add to the Graph the Ops that calculate and apply gradients.
		train_op = classification_model.training(loss, FLAGS.learning_rate)

		# Add the Op to compare the logits to the labels during evaluation.
		cm = tf.confusion_matrix(labels=labels_placeholder, predictions=pred, num_classes=3)

		# Add the variable initializer Op.
		init = tf.global_variables_initializer()

		# Create a saver for writing training checkpoints.
		saver = tf.train.Saver()

		# Create a session for running Ops on the Graph.
		sess = tf.Session()

		# And then after everything is built:

		# Run the Op to initialize the variables.
		sess.run(init)

		# saver = tf.train.import_meta_graph('../../../processed_data/dropout/model.ckpt-381999.meta')
		saver = tf.train.import_meta_graph(FLAGS.graph)
		saver.restore(sess,tf.train.latest_checkpoint(FLAGS.log_dir))

		# Make confusion matrix:
		confusion_mat=confusion(sess,cm,seq_placeholder,reg_expr_placeholder,labels_placeholder,keep_prob_pl,data_sets.test)
		np.savetxt("%s/confusion_matrix.txt"%(FLAGS.log_dir), confusion_mat)


def main(_):
	if tf.gfile.Exists(FLAGS.log_dir):
		pass
	tf.gfile.MakeDirs(FLAGS.log_dir)
	run_training()


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--learning_rate',type=float,default=0.01,help='Initial learning rate.')
	parser.add_argument('--max_steps',type=int,default=200000,help='Number of steps to run trainer.')
	parser.add_argument('--batch_size',type=int,default=100,help='Batch size.  Must divide evenly into the dataset sizes.')
	parser.add_argument('--seq_file',type=str,default='../data/yeast_promoters.txt',help='Path to promoter sequence data.')
	parser.add_argument('--expr_file',type=str,default='../data/complete_dataset.txt',help='Path to expression data.')
	parser.add_argument('--reg_names_file',type=str,default='../data/reg_names_R.txt',help='Path to regulator names.')
	parser.add_argument('--log_dir',type=str,default='../processed_data/dropout',help='Directory to put the log data.')
	parser.add_argument('--graph',type=str,default='../processed_data/dropout/model.ckpt-499999.meta',help='Directory where the log data is stored.')
	FLAGS, unparsed = parser.parse_known_args()
	tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)


