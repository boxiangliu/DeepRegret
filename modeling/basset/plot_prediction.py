from keras.models import load_model
import os,sys
import matplotlib.pyplot as plt
sys.path.append('/srv/scratch/bliu2/deepregret/scripts/modeling/basset/')
sys.path.append('/srv/scratch/bliu2/deepregret/scripts/utils/')
import input_data
from size_converter import bytes2human

model_fn='../logs/basset/model.27-0.64.hdf5'
model=load_model(model_fn)

train,val,test=input_data.read_data_sets()
train['seq']

pred=model.predict({'seq_input':test['seq'],'reg_input':test['reg']},batch_size=100,verbose=1)


plt.scatter(test['class'],pred)
plt.show()

conv2d_1=model.get_layer('convolution2d_1')
conv2d_1.get_weights()[0].shape
conv2d_1.get_weights()[0][:,:,0,0]

conv2d_1.get_weights()[0].shape
conv2d_1.get_weights()[0][:,:,0,0]

batchnorm_1=model.get_layer('batchnormalization_1')

maxpooling2d_1=model.get_layer('maxpooling2d_1')
convolution2d_2=model.get_layer('convolution2d_2')
convolution2d_2.get_weights()[0][:,:,0,0]
convolution2d_2.get_weights()[0].shape

convolution2d_3=model.get_layer('convolution2d_3')
convolution2d_3.get_weights()[0][:,:,0,0]

dense_1=model.get_layer('dense_1')
dense_1.get_weights()[0][:,0]

dense_2=model.get_layer('dense_2')
dense_2.get_weights()[0][:,0]

dense_3=model.get_layer('dense_3')
dense_3.get_weights()[0][:,0]


plt.hist(dense_3.get_weights()[0].reshape([1,-1])[0], bins=50, normed=1, histtype='bar')
plt.show()


plt.hist(convolution2d_3.get_weights()[0].reshape([1,-1])[0], bins=50, normed=1, histtype='bar')
plt.show()


dense_5=model.get_layer('dense_5')
dense_5.get_weights()[0][:,0]
plt.hist(dense_5.get_weights()[0].reshape([1,-1])[0], bins=50, normed=1, histtype='bar')
plt.show()

rgs_output=model.get_layer('rgs_output')
rgs_output.get_weights()[0][:,0]
plt.hist(rgs_output.get_weights()[0][:,0], bins=50, normed=1, histtype='bar')
plt.show()
