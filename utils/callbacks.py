from keras.callbacks import Callback
import numpy as np

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


class BatchHistory(Callback):
	def __init__(self,val_data,loss_function,every_n_batch):
		num_sample=val_data['seq'].shape[0]
		subsample=min(10000,num_sample)
		small_val_data={'seq':val_data['seq'][:subsample,:,:,:],'reg':val_data['reg'][:subsample,:],'expr':val_data['expr'][:subsample],'class':val_data['class'][:subsample]}
		self.val=small_val_data
		self.loss_function=loss_function
		self.every_n_batch=every_n_batch

	def on_train_begin(self,logs={}):
		self.val_loss=[]
		self.num_batch=0

	def on_batch_end(self,batch,logs={}):
		self.num_batch+=1
		if self.num_batch % self.every_n_batch == 0:
			y_pred=self.model.predict({'seq_input':self.val['seq'],'reg_input':self.val['reg']},batch_size=100,verbose=0)
			if self.loss_function=='mse':
				loss=np.mean(np.square(self.val['expr']-y_pred[:,0]))
			self.val_loss.append(loss)

