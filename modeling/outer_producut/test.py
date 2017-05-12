import tensorflow as tf
import numpy as np
sess = tf.InteractiveSession()
conv1_filter_depth=128
seq=tf.constant(np.repeat(1.0, 100*1000*4),shape=[100, 1000, 4])
weights=tf.constant(np.repeat(1.0,5*4*conv1_filter_depth),shape=[5,4,conv1_filter_depth])
conv=tf.nn.conv1d(seq,weights,stride=1, padding='SAME')
conv_rs=tf.reshape(conv, [100,1000*conv1_filter_depth,1])
reg_expr=tf.constant(np.repeat(1.0,100*472),shape=[100,1,472])
outer=tf.matmul(conv_rs, reg_expr)
outer_rs=tf.reshape(outer,[100,1000,conv1_filter_depth,472])
outer_rs