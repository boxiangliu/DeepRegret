# Plot the learning curve for tensor product and concatenation
# networks.
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
import pickle
import os

# Variable:
fig_dir='../figures/evaluation/learning_curve/'
if (not os.path.exists(fig_dir)): os.makedirs(fig_dir)


# Classification:
with open('../logs/small_filter/simple.classification/history.pkl','rb') as f:
	concat_history=pickle.load(f)

with open('../logs/single_layer/model.classification/history.pkl','rb') as f:
	tensor_history=pickle.load(f)


plt.plot(concat_history['val_loss'],color='red',linestyle='solid')
plt.plot(concat_history['loss'],color='red',linestyle='dashed')
plt.plot(tensor_history['val_loss'],color='blue',linestyle='solid')
plt.plot(tensor_history['loss'],color='blue',linestyle='dashed')
plt.legend(['Concat val loss','Concat train loss','Tensor val loss','Tensor train loss'])
plt.xlabel('Epoch')
plt.ylabel('Cross Entropy')
plt.savefig('%s/classification_learning_curve.pdf'%fig_dir)



# Regression:
with open('../logs/small_filter/simple/history.pkl','rb') as f:
	concat_history,concat_batch_history=pickle.load(f)

with open('../logs/single_layer/model/history.pkl','rb') as f:
	tensor_history,tensor_batch_history=pickle.load(f)


plt.plot(concat_history['val_loss'],color='red',linestyle='solid')
plt.plot(concat_history['loss'],color='red',linestyle='dashed')
plt.plot(tensor_history['val_loss'],color='blue',linestyle='solid')
plt.plot(tensor_history['loss'],color='blue',linestyle='dashed')
plt.legend(['Concat val loss','Concat train loss','Tensor val loss','Tensor train loss'])
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.savefig('%s/regression_learning_curve.pdf'%fig_dir)

plt.plot(concat_batch_history,color='red',linestyle='solid')
plt.plot(tensor_batch_history,color='blue',linestyle='solid')
plt.legend(['Concatenation','Tensor Product'])
plt.xlabel('Batch')
plt.ylabel('MSE')
plt.savefig('%s/regression_learning_curve_batch.pdf'%fig_dir)

plt.xlim(400,500)
plt.ylim(0.21,0.24)
plt.savefig('%s/regression_learning_curve_batch_crop.pdf'%fig_dir)
