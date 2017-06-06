from keras.models import Model
from keras.layers import Input, Dense, Convolution2D, Flatten, Dropout, Merge
from keras.regularizers import l1
from keras.optimizers import Adam, SGD
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
from keras.utils.np_utils import to_categorical
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
x=Convolution2D(nb_filter=filters[0],nb_row=4,nb_col=kernel_width[0],subsample=(4,4),border_mode='valid',activation='relu')(seq_input)
x=Convolution2D(nb_filter=filters[1],nb_row=1,nb_col=kernel_width[1],subsample=(1,4),border_mode='valid',activation='relu')(x)
seq_output=Flatten()(x)

print('INFO - %s'%('building concatenate model.'))
units=512
x=Merge(mode='concat',concat_axis=1)([reg_output,seq_output])
x=Dense(units,activation='relu')(x)
x=Dense(units,activation='relu')(x)
x=Dropout(0.5,seed=42)(x)
rgs_output=Dense(1,activation='linear',name='rgs_output')(x)
cls_output=Dense(3,activation='sigmoid',name='cls_output')(x)


model=Model(input=[reg_input,seq_input],output=[rgs_output,cls_output])
# sgd = SGD(lr=0.01)
model.compile(loss={'rgs_output':'mean_squared_error','cls_output':'categorical_crossentropy'},optimizer='adam',metric='accuracy')
# plot_model(model, show_shapes=True,to_file='../figures/keras/model.multi.eps')

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
