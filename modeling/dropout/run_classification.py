from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os.path
import sys
import time
import numpy as np

from six.moves import xrange
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
	seq_placeholder = tf.placeholder(tf.float32, shape=(None,classification_model.SEQ_LENGTH,4))
	reg_expr_placeholder = tf.placeholder(tf.float32, shape=(None,classification_model.NUM_REG))
	labels_placeholder = tf.placeholder(tf.int32, shape=(None))
	keep_prob = tf.placeholder(tf.float32)
	return seq_placeholder, reg_expr_placeholder, labels_placeholder, keep_prob

def fill_feed_dict(data_set, seq_pl, reg_expr_pl, labels_pl, keep_prob_pl, keep_prob, all_data=False):
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
	if all_data:
		seq_feed, reg_expr_feed, labels_feed = data_set.all_data()
	else:
		seq_feed, reg_expr_feed, labels_feed = data_set.next_batch(FLAGS.batch_size)
	feed_dict = {seq_pl: seq_feed,reg_expr_pl: reg_expr_feed,labels_pl: labels_feed, keep_prob_pl: keep_prob}
	return feed_dict


# def do_eval(sess,eval_correct,seq_placeholder,reg_expr_placeholder,labels_placeholder,keep_prob_pl,data_set):
# 	"""Runs one evaluation against the full epoch of data.
# 	Args:
# 		sess: The session in which the model has been trained.
# 		eval_correct: The Tensor that returns the number of correct predictions.
# 		seq_placeholder: The sequence placeholder.
# 		reg_expr_feed: The regulator expression placeholder. 
# 		labels_placeholder: The labels placeholder.
# 		data_set: The set of images and labels to evaluate, from
# 		  input_data.read_data_sets().
# 	"""
# 	# And run one epoch of eval.
# 	true_count = 0  # Counts the number of correct predictions.
# 	steps_per_epoch = data_set.num_examples // FLAGS.batch_size
# 	num_examples = steps_per_epoch * FLAGS.batch_size
# 	for step in xrange(steps_per_epoch):
# 		feed_dict = fill_feed_dict(data_set,seq_placeholder,reg_expr_placeholder,labels_placeholder,keep_prob_pl,keep_prob=1.0)
# 		true_count += sess.run(eval_correct, feed_dict=feed_dict)
# 		precision = float(true_count) / num_examples
# 	print('  Num examples: %d  Num correct: %d  Precision @ 1: %0.04f' % (num_examples, true_count, precision))



def do_eval(sess,accuracy,seq_placeholder,reg_expr_placeholder,labels_placeholder,keep_prob_pl,data_set,summary,summary_writer):
	"""Runs one evaluation against the full epoch of data.
	Args:
		sess: The session in which the model has been trained.
		eval_correct: The Tensor that returns the number of correct predictions.
		seq_placeholder: The sequence placeholder.
		reg_expr_feed: The regulator expression placeholder. 
		labels_placeholder: The labels placeholder.
		data_set: The set of images and labels to evaluate, from
		  input_data.read_data_sets().
	"""
	# evaluation using all data:
	feed_dict = fill_feed_dict(data_set,seq_placeholder,reg_expr_placeholder,labels_placeholder,keep_prob_pl,keep_prob=1.0,all_data=True)
	accuracy_str, summary_str = sess.run([accuracy,summary],feed_dict=feed_dict)
	print('  Num examples: %d  Accuracy: %0.04f' % (data_set.num_examples, accuracy_str))
	summary_writer.add_summary(summary_str, step)
	summary_writer.flush()


def run_training():
	"""Train for a number of steps."""
	data_sets = input_data.read_data_sets(seq_file=FLAGS.seq_file,expr_file=FLAGS.expr_file,reg_names_file=FLAGS.reg_names_file)

	# Tell TensorFlow that the model will be built into the default Graph.
	with tf.Graph().as_default():
		# Generate placeholders for the images and labels.
		seq_placeholder, reg_expr_placeholder, labels_placeholder, keep_prob_pl = placeholder_inputs(FLAGS.batch_size)

		# Build a Graph that computes predictions from the inference model.
		logits = classification_model.inference(seq_placeholder,reg_expr_placeholder,keep_prob_pl,FLAGS.batch_size)

		# Add to the Graph the Ops for loss calculation.
		loss = classification_model.loss(logits, labels_placeholder)

		# Add to the Graph the Ops that calculate and apply gradients.
		train_op = classification_model.training(loss, FLAGS.learning_rate)

		# Add the Op to compare the logits to the labels during evaluation.
		eval_correct = classification_model.evaluation(logits, labels_placeholder)
		accuracy = classification_model.accuracy(logits, labels_placeholder)

		# Add summaries: 
		tf.summary.scalar('loss', loss)
		tf.summary.scalar('accuracy',accuracy)

		# Build the summary Tensor based on the TF collection of Summaries.
		summary = tf.summary.merge_all()

		# Add the variable initializer Op.
		init = tf.global_variables_initializer()

		# Create a saver for writing training checkpoints.
		saver = tf.train.Saver()

		# Create a session for running Ops on the Graph.
		sess = tf.Session()

		# Instantiate a SummaryWriter to output summaries and the Graph.
		train_summary_writer = tf.summary.FileWriter(FLAGS.log_dir+'/train',sess.graph)
		val_summary_writer = tf.summary.FileWriter(FLAGS.log_dir+'/val',sess.graph)
		test_summary_writer = tf.summary.FileWriter(FLAGS.log_dir+'/test',sess.graph)

		# And then after everything is built:

		# Run the Op to initialize the variables.
		sess.run(init)

		# Start the training loop.
		for step in xrange(FLAGS.max_steps):
			start_time = time.time()

			# Fill a feed dictionary with the actual set of images and labels
			# for this particular training step.
			feed_dict = fill_feed_dict(data_sets.train,seq_placeholder,reg_expr_placeholder,labels_placeholder,keep_prob_pl,keep_prob=0.5)

			# Run one step of the model.  The return values are the activations
			# from the `train_op` (which is discarded) and the `loss` Op.
			_, loss_value = sess.run([train_op, loss],feed_dict=feed_dict)

			duration = time.time() - start_time

			# Write the summaries and print an overview fairly often.
			if step % 100 == 0:
				# Print status to stdout.
				print('Step %d: loss = %.2f (%.3f sec)' % (step, loss_value, duration))
				# Update the events file.
				# summary_str = sess.run(summary, feed_dict=feed_dict)
				# summary_writer.add_summary(summary_str, step)
				# summary_writer.flush()

			# Save a checkpoint and evaluate the model periodically.
			if (step + 1) % 1000 == 0 or (step + 1) == FLAGS.max_steps:
				checkpoint_file = os.path.join(FLAGS.log_dir, 'model.ckpt')
				saver.save(sess, checkpoint_file, global_step=step)

				# Evaluate against the training set.
				# print('Training Data Eval:')
				# do_eval(sess,eval_correct,seq_placeholder,reg_expr_placeholder,labels_placeholder,keep_prob_pl,data_sets.train)
				# do_eval(sess,accuracy,seq_placeholder,reg_expr_placeholder,labels_placeholder,keep_prob_pl,data_sets.train,summary,train_summary_writer)
				
				# Evaluate against the validation set.
				print('Validation Data Eval:')
				# do_eval(sess,eval_correct,seq_placeholder,reg_expr_placeholder,labels_placeholder,keep_prob_pl,data_sets.validation)
				do_eval(sess,accuracy,seq_placeholder,reg_expr_placeholder,labels_placeholder,keep_prob_pl,data_sets.validation,summary,val_summary_writer)
				
				# Evaluate against the test set.
				print('Test Data Eval:')
				# do_eval(sess,eval_correct,seq_placeholder,reg_expr_placeholder,labels_placeholder,keep_prob_pl,data_sets.test)
				do_eval(sess,accuracy,seq_placeholder,reg_expr_placeholder,labels_placeholder,keep_prob_pl,data_sets.test,summary,test_summary_writer)

def main(_):
	if tf.gfile.Exists(FLAGS.log_dir):
		tf.gfile.DeleteRecursively(FLAGS.log_dir)
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
	parser.add_argument('--log_dir',type=str,default='../processed_data/',help='Directory to put the log data.')
	FLAGS, unparsed = parser.parse_known_args()
	tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

