import tensorflow as tf
import numpy as np
import os
import time
import itertools
import dataset
from sklearn.metrics import confusion_matrix

flags = tf.flags
flags.DEFINE_float('rs', 0.0000001, 'rs')
flags.DEFINE_string("log_dir", None, "log_dir")
flags.DEFINE_string("checkpoint", None, "checkpoint")
flags.DEFINE_integer("max_epoch", 20, "max_epoch")
flags.DEFINE_float('lr', 0.0003, 'lr')

flags.DEFINE_boolean('interpret',False,'interpret')

FLAGS =flags.FLAGS

NUM_REG = 472
SEQ_LENGTH = 1000
NUM_BINS = 5

class Config(object):
    batch_size = 32
    lr = FLAGS.lr
    rs = FLAGS.rs

# EXPRESSION_MAT = np.load()
# DOUBLE_KO_MAT = np.load()
# GENE_SEQS = np.load()

# REGS = [1,5,6,7]

# NUM_GENES = EXPRESSION_MAT.shape[0]
# NUM_KO = EXPRESSION_MAT.shape[1]
# assert NUM_GENES == GENE_SEQS.shape[0]
# assert NUM_GENES == DOUBLE_KO_MAT.shape[0]


# class KODataSet(object):
#     def __init__(self, val_genes, val_ko, test_genes, test_ko):
#         self.val_genes = val_genes
#         self.val_ko = val_ko
#         self.test_genes = test_genes
#         self.test_ko = test_ko

#         self.train_genes = np.setdiff1d(np.arange(NUM_GENES), np.concatenate([val_genes, test_genes]))
#         self.train_ko = np.setdiff1d(np.arange(NUM_KO), np.concatenate([val_ko, test_ko]))      

#     def generate_training_batch(batch_size):
#         selected_gene_idx = np.random.choice(self.train_genes, size = batch_size)
#         selected_ko_idx = np.random.randint(self.train_ko, size = batch_size)

#         #get slices for the inputs
#         X_gene_seqs = GENE_SEQS[selected_gene_idx,:,:]
#         X_regulator_expression = EXPRESSION_MAT[regulator_idx,selected_ko_idx]

#         #get slices for the outputs
#         Y_gene_expression = EXPRESSION_MAT[selected_gene_idx, selected_ko_idx]
#         Y_double_ko_fitness = DOUBLE_KO_MAT[selected_gene_idx, selected_ko_idx]

#         return X_gene_seqs, X_regulator_expression, Y_gene_expression, Y_double_ko_fitness

#     def generate_val_batch(batch_size):
#         pass

#     def generate_test_batch(batch_size):
#         pass





class TFKOModel(object):
    def __init__(self,is_training = True,config = Config()):
        self.add_config(config)
        self.is_train = is_training
        self.add_placeholders()

        self.seq_features =seq_features = self.build_seq_layers(self.X_gene_seqs)
        expression_features = self.build_expression_layers(self.X_regulator_expression)
        joined_features = self.build_joining_layers(seq_features, expression_features)
        self.expression_prediction = expression_prediction = self.build_expression_prediction_layers(joined_features)
        #fitness_prediction = self.build_fitness_prediction_layers(joined_features)
        self.expression_loss,self.predicted_labels = self.loss(expression_prediction)

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
        self.Y_gene_expression = tf.placeholder(tf.int64, [self.batch_size])
        self.Y_double_ko_fitness = tf.placeholder(tf.float32, [self.batch_size])

    def build_seq_layers(self,gene_seqs):
        layer_kernel_widths = [5,5,5,5]
        layer_depths = [4,32,64,64,32]


        with tf.variable_scope('seq_conv_layers'):
            assert len(layer_kernel_widths) +1 == len(layer_depths)
            weights  = []
            biases = []
            for i in range(len(layer_kernel_widths)):
                weights.append(tf.get_variable('weight_'+str(i), shape = [layer_kernel_widths[i], layer_depths[i], layer_depths[i+1]], initializer = tf.contrib.layers.xavier_initializer(uniform = False)))
                biases.append(tf.get_variable('bias_'+str(i), shape = [layer_depths[i+1]]))


        out = gene_seqs

        self.seq_conv_w = weights
        for i in range(len(weights)):
            if i%2 ==1:
                stride = 2
            else:
                stride = 1
            l = tf.nn.conv1d(out, weights[i], stride = stride, padding = 'VALID') + biases[i]
            
            out = tf.nn.relu(l)#tf.nn.relu(tf.nn.batch_normalization(l, mean,variance, 0, 1, 0.000001))            


        out = tf.reshape(out, [self.batch_size, -1])
        print out.get_shape().as_list()
        with tf.variable_scope('seq_conv_fc'):
            w1 = tf.get_variable('fc_1', shape = [out.get_shape().as_list()[1], 256])
            w2 = tf.get_variable('fc_2', shape = [256, 128])
            b1 = tf.get_variable('fc_b1', shape = [256])
            b2 = tf.get_variable('fc_b2', shape = [128])
            sig = tf.get_variable('sig', [1])
            bet = tf.get_variable('beta', [1])

        out = tf.matmul(out, w1) +b1
        mean, variance = tf.nn.moments(out,[0], keep_dims = True)

        out =tf.nn.relu(tf.nn.batch_normalization(out, mean,variance, bet, sig, 0.000001))

        return out

    def build_expression_layers(self,regulator_expression):
        output = 128
        with tf.variable_scope('expression'):
            W1 = tf.get_variable('w1',[NUM_REG, output])
            b1 = tf.get_variable('b1',[output])
            W2 = tf.get_variable('w2',[output, 128])
            b2 = tf.get_variable('b2',[128])
            W3 = tf.get_variable('w3',[128,128])
            b3 = tf.get_variable('b3',[128])
            sig = tf.get_variable('sig', [1])
            bet = tf.get_variable('beta', [1])
        if self.is_train:
            noise = 0.000001*tf.random_normal(regulator_expression.get_shape().as_list())
        else:
            noise = 0
        l = tf.matmul(regulator_expression + noise, W1) + b1    
        mean, variance = tf.nn.moments(l,[0], keep_dims = True)        
        y1 =tf.nn.relu(l)#tf.nn.batch_normalization(l, mean,variance, 0, 1, 0.000001))

        l = tf.matmul(y1, W2) + b2
        mean, variance = tf.nn.moments(l,[0], keep_dims = True)
        
        y2 =tf.nn.relu(l)#tf.nn.batch_normalization(l, mean,variance, 0, 1, 0.000001))

        self.regulator_fc_w = [W1,W2,W3]

        l = tf.matmul(y2, W3) + b3
        mean, variance = tf.nn.moments(l,[0], keep_dims = True)

        expression_features = tf.nn.relu(tf.nn.batch_normalization(l, mean,variance, bet, sig, 0.000001))#tf.nn.batch_normalization(l, mean,variance, 0, 1, 0.000001))



        return expression_features

    def build_joining_layers(self,seq_features,expression_features):
        return tf.concat(1, [seq_features, expression_features], name = 'concat')   
#        w = tf.reshape(expression_features, [self.batch_size,128,16])
#	return tf.reduce_sum(tf.reshape(seq_features, [self.batch_size, 128,1])*w, 1)

    def build_expression_prediction_layers(self,joined_features):
        
        output = 512
        with tf.variable_scope('expression_prediction'):
            W1 = tf.get_variable('w1',[joined_features.get_shape().as_list()[1], output])
            b1 = tf.get_variable('b1',[output])
            W2 = tf.get_variable('w2',[output, output])
            b2 = tf.get_variable('b2',[output])
            W3 = tf.get_variable('w3',[ output,NUM_BINS])
            b3 = tf.get_variable('b3',[NUM_BINS])
            sig = tf.get_variable('sig', [1])
            bet = tf.get_variable('beta', [1])
            
        y1 =tf.nn.relu(tf.matmul(joined_features, W1) + b1)
        l = tf.nn.relu(tf.matmul(y1, W2) + b2)
        
        mean, variance = tf.nn.moments(l,[0], keep_dims = True)
        y2 = tf.matmul(l, W3) + b3

        

        self.prediction_fc_w = [W1,W2,W3]
        return y2

    def build_fitness_prediction_layers(self,joined_features):
        pass
        return fitness_prediction

    def loss(self, expression_prediction):
        nonzero = 1#tf.to_float(tf.greater(tf.abs(self.Y_gene_expression),0.01))
        #probs = tf.nn.sigmoid(expression_prediction)
        #loss = -tf.reduce_mean(labels*tf.log(probs) + (1-labels)*tf.log(1-probs))
        
        
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(expression_prediction, self.Y_gene_expression)
        #loss = tf.square(tf.reduce_sum(expression_prediction*self.Y_gene_expression))/(tf.reduce_sum(tf.square(expression_prediction))*tf.reduce_sum(tf.square(self.Y_gene_expression)))
        return tf.reduce_mean(loss),tf.argmax(expression_prediction, 1)
        #return fitness_loss, expression_loss, total_loss

    def regularize(self):
        reg = 0
        for w in self.seq_conv_w:
            reg += tf.reduce_sum(tf.square(w))
        for w in self.prediction_fc_w:
            reg += tf.reduce_sum(tf.square(w))
        for w in self.regulator_fc_w:
            reg += tf.reduce_sum(tf.square(w))
        
        return 0

    def add_summaries(self):
        if self.is_train:
            tag = "train_expression_loss"

        if not self.is_train:
            tag = "val_expression_loss"
        self.summary = tf.scalar_summary(tag,self.expression_loss)
        

    def add_train(self,loss):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate= self.lr)
        train_op = optimizer.minimize(loss + self.regularize())
        return train_op

def get_seq_hotspots(model, session, official_name,reg_idx=None):

    seq_mask = np.ones((1000,4))
    seq_mask[:6,:] = 0.0
    reg_means = dataset.get_reg_means()
    reg_mask = np.zeros(NUM_REG)
    reg_mask[0] = 1.0
    reg_batches = np.array_split(np.stack([reg_means+ 1.0*np.roll(reg_mask,i) for i in range(15*32)],0), 1+NUM_REG/32)
    gene_seq = dataset.get_gene_seq(official_name)
    rep_gene_seq = np.repeat(gene_seq, 32).reshape([32, gene_seq.shape[0], gene_seq.shape[1]])
    print len(reg_batches)
    reg_response = np.zeros((len(reg_batches)*32, 5))
    i = 0
    print "getting top regulator for",official_name
    for reg_batch in reg_batches:
        
        reg_response[(32*i):(32*(i+1)),:] = session.run(model.expression_prediction, feed_dict = {model.X_gene_seqs:rep_gene_seq, model.X_regulator_expression:reg_batch})
        i+=1

    zeroed_out= np.stack([gene_seq*np.roll(seq_mask, i) for i in range(994)], 0)
    print zeroed_out.shape
    seq_batches  = np.split(zeroed_out[:992,:,:], 31, axis = 0)
    print(len(seq_batches))
    seq_response = np.zeros((len(seq_batches)*32, 5))
    if reg_idx is None:
        reg_idx = np.argmin(reg_response[:,2])
        print "top regulator was:", reg_idx
    i = 0
    print "doing rolling window sequence test for gene",official_name
    regg = reg_means
    regg[reg_idx] += 1
    rep_reg_means = np.repeat(regg, 32).reshape([32, reg_means.shape[0]])
    for seq_batch in seq_batches:
        print seq_batch.shape
        seq_response[(32*i):(32*(i+1)),:] = session.run(model.expression_prediction, feed_dict = {model.X_gene_seqs:seq_batch, model.X_regulator_expression:rep_reg_means})         
        i+=1

    print "seq responses by window start position (negative is more change)", seq_response[:,2]
    return seq_response, reg_response




def discretize(vector):
    return np.maximum(np.minimum(np.floor(NUM_BINS*(0.5 + vector/8)), NUM_BINS-1),0)

def validate(val_model, session):
    its = 0
    val_summaries = []
    val_loss = 0
    true_labels = []
    predicted_labels = []
    for X_gene_seqs, X_regulator_expression, Y_gene_expression in dataset.batch_generator(dataset.getVal(),val_model.batch_size):        
        
        feed_dict = {val_model.X_gene_seqs:X_gene_seqs, val_model.X_regulator_expression:X_regulator_expression/10, val_model.Y_gene_expression:discretize(Y_gene_expression)}
        vl, val_summary,pred_lab = session.run([val_model.expression_loss,val_model.summary,val_model.predicted_labels],feed_dict = feed_dict)
        val_loss += vl

        val_summaries.append(val_summary)
        disc = discretize(Y_gene_expression)
        for k in range(len(pred_lab)):
            predicted_labels.append(pred_lab[k])
            true_labels.append(disc[k])
        its +=1

    return val_loss/its, val_summaries, confusion_matrix(np.array(true_labels), np.array(predicted_labels))


def test(val_model, session):
    its = 0
    val_summaries = []
    val_loss = 0
    true_labels = []
    predicted_labels = []
    for X_gene_seqs, X_regulator_expression, Y_gene_expression in dataset.batch_generator(dataset.getTest(),val_model.batch_size):        
        
        feed_dict = {val_model.X_gene_seqs:X_gene_seqs, val_model.X_regulator_expression:X_regulator_expression/10, val_model.Y_gene_expression:discretize(Y_gene_expression)}
        vl, val_summary,pred_lab = session.run([val_model.expression_loss,val_model.summary,val_model.predicted_labels],feed_dict = feed_dict)
        val_loss += vl

        val_summaries.append(val_summary)
        disc = discretize(Y_gene_expression)
        for k in range(len(pred_lab)):
            predicted_labels.append(pred_lab[k])
            true_labels.append(disc[k])
        its +=1

    return val_loss/its, val_summaries, confusion_matrix(np.array(true_labels), np.array(predicted_labels))


def train_epoch(train_model,val_model, session, global_iters,val_every=4000):
    summary_writer = tf.train.SummaryWriter(FLAGS.log_dir)
    i=0
    print("training")       
    for X_gene_seqs, X_regulator_expression, Y_gene_expression in dataset.batch_generator(dataset.getTrain(),train_model.batch_size):
        
        feed_dict = {train_model.X_gene_seqs:X_gene_seqs, train_model.X_regulator_expression:X_regulator_expression/10, train_model.Y_gene_expression:discretize(Y_gene_expression)}
        expression_loss, train_summary, _ = session.run([train_model.expression_loss,train_model.summary, train_model.train_op],feed_dict = feed_dict)
        summary_writer.add_summary(train_summary,global_iters)

        if i%200 == 199:
            print discretize(Y_gene_expression)
            print "expression loss: %.4f" % expression_loss
            print "approximately %.4f percent remaining" % (100*(1-i*32/float(633906)))
                
        if i%val_every == val_every-1:
            val_loss, val_summaries, val_confusion = validate(val_model,session)
            for summary in val_summaries:
                summary_writer.add_summary(summary,global_iters)
            summary_writer.flush()
            print "====VALIDATING===="
            print "validation loss was: %.4f" % val_loss
            print "confusion matrix", val_confusion
            print "====VALIDATING===="
        i+=1

        global_iters+=1

    return global_iters
    
'''
def train_epoch_old(train_model,val_model, session, global_iters,val_every=4000):
    summary_writer = tf.train.SummaryWriter(FLAGS.log_dir)
    i=0
    print("training")	    
    for X_gene_seqs, X_regulator_expression, Y_gene_expression in dataset.batch_generator(dataset.getTrain(),train_model.batch_size):
        
        feed_dict = {train_model.X_gene_seqs:X_gene_seqs, train_model.X_regulator_expression:X_regulator_expression/10, train_model.Y_gene_expression:discretize(Y_gene_expression)}
        expression_loss, train_summary, _ = session.run([train_model.expression_loss,train_model.summary, train_model.train_op],feed_dict = feed_dict)
        summary_writer.add_summary(train_summary,global_iters)

        if i%200 == 199:
            print discretize(Y_gene_expression)
            print "expression loss: %.4f" % expression_loss
            print "approximately %.4f percent remaining" % (100*(1-i*32/float(633906)))
                
        if i%val_every == val_every-1:
            val_loss, val_summaries = validate(val_model,session)
            for summary in val_summaries:
                summary_writer.add_summary(summary,global_iters)
            summary_writer.flush()
            print "validation loss was: %.4f" % val_loss
        i+=1

        global_iters+=1

    return global_iters
'''
def main(_):
    if not FLAGS.log_dir:
        raise ValueError("Must set --log_dir to logging directory")    
    
    config = Config()
    
    

    with tf.Graph().as_default(), tf.Session() as session:
        
        initializer = tf.truncated_normal_initializer(stddev = 0.001, seed=None, dtype=tf.float32)

        with tf.variable_scope("model", reuse = None, initializer = initializer):
            train_model = TFKOModel(config = config,is_training = True)
            saver = tf.train.Saver()

        with tf.variable_scope("model", reuse = True, initializer = initializer):
            val_model= TFKOModel(is_training = False)

        if FLAGS.checkpoint:
            saver.restore(session, FLAGS.checkpoint)
            print "successfully restored checkpoint"
        else:
            tf.initialize_all_variables().run()


        if FLAGS.interpret:
            genes = ['R0010W','YAL016W','YFR034C']            
            for gene in genes:
                seq_response, reg_response = get_seq_hotspots(val_model, session, gene)
                np.save(gene+'seq_response',seq_response)
                np.save(gene+'reg_response',reg_response)
            return

        global_iters = 0
        for i in range(FLAGS.max_epoch):
            global_iters = train_epoch(train_model,val_model, session,global_iters)

            if i%5 ==4:
                saver.save(session, os.path.join(FLAGS.log_dir,"model.checkpoint"), global_iters)

        print "testing"
        test_loss, test_summaries, test_confusion = test(val_model, session)

        print("Test loss was %.4f" % test_loss)
        print "Test confusion:", test_confusion

if __name__ == "__main__":
    tf.app.run()



