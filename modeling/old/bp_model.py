import tensorflow as tf
import numpy as np
import os
import time
import itertools
import dataset

flags = tf.flags
flags.DEFINE_float('rs', 0.0001, 'rs')
flags.DEFINE_string("log_dir", None, "log_dir") 
flags.DEFINE_string("checkpoint", None, "checkpoint")
flags.DEFINE_integer("max_epoch", 20, "max_epoch")
flags.DEFINE_float('lr', 0.005, 'lr')

FLAGS =flags.FLAGS

NUM_REG = 472
SEQ_LENGTH = 1000

class Config(object):
	batch_size = 32
	lr = FLAGS.lr
	rs = FLAGS.rs


class TFKOModel(object):
	def __init__(self,is_training = True,config = Config()):
		self.add_config(config)
		self.is_train = is_training
		self.add_placeholders()

		self.seq_features = self.build_seq_layers(self.X_gene_seqs)
		self.expression_features = self.build_expression_layers(self.X_regulator_expression)
		self.joined_features = self.build_joining_layers(self.seq_features, self.expression_features)
		self.expression_prediction = self.build_expression_prediction_layers(self.joined_features)
		#fitness_prediction = self.build_fitness_prediction_layers(joined_features)
		self.expression_loss = self.loss(self.expression_prediction)

		if is_training:
			self.train_op = self.add_train(self.expression_loss)
	
		self.add_summaries()
	print("model built")

	def add_config(self,config):
		self.batch_size = config.batch_size
		self.lr = config.lr
		self.rs= config.rs
		

	def add_placeholders(self):
		# input placeholders
		self.X_gene_seqs = tf.placeholder(tf.float32, [self.batch_size, SEQ_LENGTH, 4])
		self.X_regulator_expression = tf.placeholder(tf.float32, [self.batch_size, NUM_REG])

		#output placeholders - will need to make sure right ones are fed
		self.Y_gene_expression = tf.placeholder(tf.float32, [self.batch_size])
		self.Y_double_ko_fitness = tf.placeholder(tf.float32, [self.batch_size])

	def build_seq_layers(self,gene_seqs):
		layer_kernel_widths = [100,15]
		layer_depths = [4,32,256]


		with tf.variable_scope('seq_conv_layers'):
			assert len(layer_kernel_widths) +1 == len(layer_depths)
			weights = []
			biases = []
			for i in range(len(layer_kernel_widths)):
				weights.append(tf.get_variable('weight_'+str(i), shape = [layer_kernel_widths[i], layer_depths[i], layer_depths[i+1]],initializer=tf.contrib.layers.xavier_initializer()))
				biases.append(tf.get_variable('bias_'+str(i), shape = [layer_depths[i+1]]))

		out = gene_seqs

		self.seq_conv_w = weights
		for i in range(len(weights)):
			out = tf.nn.relu(tf.nn.conv1d(out, weights[i], stride = 4, padding = 'VALID') + biases[i])

		return out

	def build_expression_layers(self,regulator_expression):
		output = 512

		W1 = tf.get_variable("ExprW1", [NUM_REG, output], initializer=tf.random_normal_initializer(0, 0.0))
		b1 = tf.get_variable("ExprB1", [output], initializer=tf.random_normal_initializer(0, 0.0))
		y1 = tf.nn.relu(tf.matmul(regulator_expression, W1) + b1)

		W2 = tf.get_variable("ExprW2", [output, output], initializer=tf.random_normal_initializer(0, 0.0))
		b2 = tf.get_variable("ExprB2", [output], initializer=tf.random_normal_initializer(0, 0.0))
		y2 =tf.nn.relu(tf.matmul(y1, W2) + b2)

		W3 = tf.get_variable("ExprW3", [output, output], initializer=tf.random_normal_initializer(0, 0.0))
		b3 = tf.get_variable("ExprB3", [output], initializer=tf.random_normal_initializer(0, 0.0))

		self.regulator_fc_w = [W1,W2,W3]

		expression_features = tf.nn.relu(tf.matmul(y2, W3) + b3)



		return expression_features

	def build_joining_layers(self,seq_features,expression_features):
		
		return tf.concat(1, [tf.reshape(seq_features, [self.batch_size, -1]), expression_features], name = 'concat')

	def build_expression_prediction_layers(self,joined_features):
		
		output = 512

		W1 = tf.get_variable("PredW1", [joined_features.get_shape().as_list()[1], output], initializer=tf.random_normal_initializer(0, 0.0))
		b1 = tf.get_variable("PredB1", [output], initializer=tf.random_normal_initializer(0, 0.0))
		y1 = tf.nn.relu(tf.matmul(joined_features, W1) + b1)

		W2 = tf.get_variable("PredW2", [output, output], initializer=tf.random_normal_initializer(0, 0.0))
		b2 = tf.get_variable("PredB2", [output], initializer=tf.random_normal_initializer(0, 0.0))
		y2 = tf.nn.relu(tf.matmul(y1, W2) + b2)

		W3 = tf.get_variable("PredW3", [output, 1], initializer=tf.random_normal_initializer(0, 0.0))
		b3 = tf.get_variable("PredB3", [1], initializer=tf.random_normal_initializer(0, 0.0))

		expression_features = tf.matmul(y2, W3) + b3

		self.prediction_fc_w = [W1,W2,W3]
		return expression_features


	def loss(self, expression_prediction):
		
		loss = tf.reduce_mean(tf.square(expression_prediction-self.Y_gene_expression))
		return loss
		#return fitness_loss, expression_loss, total_loss

	def regularize(self):
		reg = 0
		for w in self.seq_conv_w:
			reg += tf.reduce_sum(tf.square(w))
		for w in self.prediction_fc_w:
			reg += tf.reduce_sum(tf.square(w))
		for w in self.regulator_fc_w:
			reg += tf.reduce_sum(tf.square(w))
		
		return self.rs * reg

	def add_summaries(self):
		if self.is_train:
			tag = "train_expression_loss"
		if not self.is_train:
			tag = "val_expression_loss"
		self.summary = tf.scalar_summary(tag,self.expression_loss)
		

	def add_train(self,loss):
		optimizer = tf.train.AdamOptimizer(learning_rate= self.lr)
		train_op = optimizer.minimize(loss) # + self.regularize())
		return train_op


def validate(val_model, session):
	its = 0
	val_summaries = []
	val_loss = 0
	for X_gene_seqs, X_regulator_expression, Y_gene_expression in dataset.batch_generator(dataset.getVal(),val_model.batch_size):        
		feed_dict = {val_model.X_gene_seqs:X_gene_seqs, val_model.X_regulator_expression:X_regulator_expression, val_model.Y_gene_expression:Y_gene_expression}
		vl, val_summary = session.run([val_model.expression_loss,val_model.summary],feed_dict = feed_dict)
		val_loss += vl
		val_summaries.append(val_summary)
		its +=1

	return val_loss/its, val_summaries

def test(model, session):
	its = 0
	test_summaries = []
	test_loss = 0
	for X_gene_seqs, X_regulator_expression, Y_gene_expression in dataset.batch_generator(dataset.getTest(), model.batch_size):        
		feed_dict = {model.X_gene_seqs:X_gene_seqs, model.X_regulator_expression:X_regulator_expression, model.Y_gene_expression:Y_gene_expression}
		tl, test_summary = session.run([model.expression_loss,model.summary],feed_dict = feed_dict)
		test_loss += tl
		test_summaries.append(test_summary)
		its +=1

	return test_loss/its, test_summaries
	

def train_epoch(train_model,val_model, session, global_iters, epoch, val_every=4000):
	summary_writer = tf.train.SummaryWriter(FLAGS.log_dir)
	i=0
	print("training")
	for X_gene_seqs, X_regulator_expression, Y_gene_expression in dataset.batch_generator(dataset.getTrain(),train_model.batch_size):
		feed_dict = {train_model.X_gene_seqs:X_gene_seqs, train_model.X_regulator_expression:X_regulator_expression, train_model.Y_gene_expression:Y_gene_expression}
		expression_loss, train_summary, _, seq, pred= session.run([train_model.expression_loss,train_model.summary, train_model.train_op, train_model.seq_features, train_model.expression_prediction],feed_dict = feed_dict)
		summary_writer.add_summary(train_summary,global_iters)
		

		if i%50 == 49:
			print "expression loss: %.4f" % expression_loss
			print "approximately %.4f percent remaining in epoch %d" % ((100*(1-i*32/float(633906))), epoch)
			print pred
		if i%val_every == val_every-1:
			val_loss, val_summaries = validate(val_model,session)
			for summary in val_summaries:
				summary_writer.add_summary(summary,global_iters)
			summary_writer.flush()
			print "validation loss was: %.4f" % val_loss
		i+=1

		global_iters+=1

	return global_iters

def main(_):
	if not FLAGS.log_dir:
		raise ValueError("Must set --log_dir to logging directory")    
	
	config = Config()
	
	

	with tf.Graph().as_default(), tf.Session() as session:
		
		initializer = tf.truncated_normal_initializer(stddev = 0.01)

		with tf.variable_scope("model", reuse = None, initializer = initializer):
			train_model = TFKOModel(config = config,is_training = True)
			saver = tf.train.Saver()

		with tf.variable_scope("model", reuse = True, initializer = initializer):
			val_model= TFKOModel(is_training = False)

		if FLAGS.checkpoint:
			saver.restore(session, FLAGS.checkpoint)
		else:
			tf.initialize_all_variables().run()

		global_iters = 0
		for i in range(FLAGS.max_epoch):
			global_iters = train_epoch(train_model,val_model, session,global_iters, i)

			if i%5 ==4:
				saver.save(session, "model.checkpoint", global_iters)
	
		test_loss, train_summary = test(val_model, session)
	print("Test loss was %.4f" % test_loss)

if __name__ == "__main__":
	tf.app.run()



