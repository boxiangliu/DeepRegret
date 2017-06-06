from keras.models import Model
from keras.layers import Input, Dense, Conv2D, concatenate, Flatten, Dropout, Concatenate
from keras.regularizers import l1
from keras.optimizers import Adam, SGD
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
from keras.utils import plot_model, to_categorical
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from keras import backend as K
import numpy as np
import sys
sys.path.append('/srv/scratch/bliu2/deepregret/scripts/modeling/keras/')
import input_data

class Prediction(Callback):
	def __init__(self,log_dir):
		self.log_dir = log_dir

	def on_train_begin(self, logs={}):
		self.pred = []

	def on_epoch_end(self, epoch, logs={}):
		y_pred = self.model.predict({'seq_input':self.validation_data[1],'reg_input':self.validation_data[0]})
		np.savetxt('%s/prediction.epoch%s.txt'%(self.log_dir,epoch), y_pred)
		self.pred.append(y_pred)
 		return


num_reg=472
seq_length=1000
L1=0.01

print('INFO - %s'%('building regression model.'))
units=512
reg_input=Input(shape=(num_reg,),dtype='float32',name='reg_input')
x=Dense(units,activation='relu')(reg_input)
x=Dense(units,activation='relu')(x)
reg_output=Dense(units,activation='relu')(x)

print('INFO - %s'%('building sequence model.'))
filters=[32,256]
kernel_width=[100,15]
seq_input=Input(shape=(4,seq_length,1),dtype='float32',name='seq_input')
x=Conv2D(filters=filters[0],kernel_size=(4,kernel_width[0]),strides=4,padding='valid',activation='relu',kernel_initializer='glorot_uniform',bias_initializer='zeros')(seq_input)
x=Conv2D(filters=filters[1],kernel_size=(1,kernel_width[1]),strides=4,padding='valid',activation='relu',kernel_initializer='glorot_uniform',bias_initializer='zeros')(x)
seq_output=Flatten()(x)

print('INFO - %s'%('building concatenate model.'))
units=512
x=concatenate([reg_output,seq_output])
x=Dense(units,activation='relu',kernel_initializer='glorot_uniform',bias_initializer='zeros')(x)
x=Dense(units,activation='relu',kernel_initializer='glorot_uniform',bias_initializer='zeros')(x)
x=Dropout(0.5,seed=42)(x)
rgs_output=Dense(1,activation='linear',name='rgs_output')(x)
cls_output=Dense(3,activation='sigmoid',name='cls_output')(x)


model=Model(inputs=[reg_input,seq_input],outputs=[rgs_output,cls_output])
# sgd = SGD(lr=0.01)
model.compile(loss={'rgs_output':'mean_squared_error','cls_output':'categorical_crossentropy'},optimizer='adam',metric='accuracy')
plot_model(model, show_shapes=True,to_file='../figures/keras/model.multi.eps')

print('INFO - %s'%('loading data.'))
train,val,test=input_data.read_data_sets()


print('INFO - %s'%('training model.'))
early_stopping=EarlyStopping(monitor='val_loss',patience=2)
checkpoint=ModelCheckpoint(filepath="../logs/keras/model.multi/model.{epoch:02d}-{val_loss:.2f}.hdf5", monitor='val_loss')
model.fit({'seq_input':train['seq'],'reg_input':train['reg']},{'rgs_output':train['expr'],'cls_output':to_categorical(train['class'])},validation_data=({'seq_input':val['seq'],'reg_input':val['reg']},{'rgs_output':val['expr'],'cls_output':to_categorical(val['class'])}),epochs=30,batch_size=100,callbacks=[early_stopping,checkpoint],verbose=1)


# pred=model.predict({'seq_input':test['seq'],'reg_input':test['reg']},batch_size=1000,verbose=1)
# plt.scatter(pred,test['expr'])
# plt.savefig("%s/pred_vs_obs.png"%('../figures/keras/model.multi/'))
# output=np.column_stack((test['expr'], pred[:,0]))
# np.savetxt("%s/prediction.txt"%('../processed_data/keras/model.multi/'), output,delimiter='\t')


#Convert a keras sequential model
import deeplift
from deeplift.conversion import keras_conversion as kc
from deeplift.blobs import NonlinearMxtsMode
#NonlinearMxtsMode defines the method for computing importance scores.
#NonlinearMxtsMode.DeepLIFT_GenomicsDefault uses the RevealCancel rule on Dense layers
#and the Rescale rule on conv layers (see paper for rationale)
#Other supported values are:
#NonlinearMxtsMode.RevealCancel - DeepLIFT-RevealCancel at all layers (used for the MNIST example)
#NonlinearMxtsMode.Rescale - DeepLIFT-rescale at all layers
#NonlinearMxtsMode.Gradient - the 'multipliers' will be the same as the gradients
#NonlinearMxtsMode.GuidedBackprop - the 'multipliers' will be what you get from guided backprop
#Use deeplift.util.get_integrated_gradients_function to compute integrated gradients
#Feel free to email avanti [dot] shrikumar@gmail.com if anything is unclear
deeplift_model = kc.convert_functional_model(model,
                    nonlinear_mxts_mode=deeplift.blobs.NonlinearMxtsMode.DeepLIFT_GenomicsDefault)

#Specify the index of the layer to compute the importance scores of.
#In the example below, we find scores for the input layer, which is idx 0 in deeplift_model.get_layers()
find_scores_layer_idx = 0

#Compile the function that computes the contribution scores
#For sigmoid or softmax outputs, target_layer_idx should be -2 (the default)
#(See "3.6 Choice of target layer" in https://arxiv.org/abs/1704.02685 for justification)
#For regression tasks with a linear output, target_layer_idx should be -1
#(which simply refers to the last layer)
#If you want the DeepLIFT multipliers instead of the contribution scores, you can use get_target_multipliers_func
deeplift_contribs_func = deeplift_model.get_target_contribs_func(
                            find_scores_layer_idx=find_scores_layer_idx,
                            target_layer_idx=-1)
#You can also provide an array of indices to find_scores_layer_idx to get scores for multiple layers at once

#compute scores on inputs
#input_data_list is a list containing the data for different input layers
#eg: for MNIST, there is one input layer with with dimensions 1 x 28 x 28
#In the example below, let X be an array with dimension n x 1 x 28 x 28 where n is the number of examples
#task_idx represents the index of the node in the output layer that we wish to compute scores.
#Eg: if the output is a 10-way softmax, and task_idx is 0, we will compute scores for the first softmax class
scores = np.array(deeplift_contribs_func(task_idx=0,
                                         input_data_list=[X],
                                         batch_size=10,
                                         progress_update=1000))