import tensorflow as tf
import numpy as np
sess = tf.InteractiveSession()
conv1_filter_depth=128
seq=tf.constant(np.repeat(1.0, 100*1000*4),shape=[100, 1000, 4])
weights=tf.constant(np.repeat(1.0,5*4*conv1_filter_depth),shape=[5,4,conv1_filter_depth])
# conv=tf.nn.conv1d(seq,weights,stride=1, padding='SAME')
conv=tf.constant(np.arange(1,100*2*2,dtype=np.float32),shape=[100,2,2])

# conv_rs=tf.reshape(conv, [100,1000*conv1_filter_depth,1])
conv_rs=tf.reshape(conv, shape=[100,-1,1])
conv_rs.eval()[0,]
# reg_expr=tf.constant(np.repeat(1.0,100*2),shape=[100,2])
reg_expr=tf.constant(np.arange(1,100*2,dtype=np.float32),shape=[100,2])
reg_expr_rs=tf.reshape(reg_expr,shape=[100,1,2])
reg_expr_rs.eval()[0,]
outer=tf.matmul(conv_rs, reg_expr_rs)
outer.eval()
outer_rs=tf.reshape(outer,[100,-1])
outer_rs.eval()[1,]
outer_rs