from keras.models import Model
from keras.layers import Input, Dense,     RevCompConv1D, Conv2D, Flatten, Dropout, Merge,     Activation, MaxPooling1D, RevCompConv1DBatchNorm,     BatchNormalization, DenseAfterRevcompConv1D
from keras.optimizers import Adam, SGD, RMSprop
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import keras

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

import pickle
import numpy as np
import sys
import os
sys.path.append('../../utils/')
import input_data_config as input_data
from callbacks import BatchHistory

import yaml

from ggplot import *
import tensorflow as tf
import argparse

fig_dir = 'figures/'
log_dir = 'logs/'
out_dir = 'processed_data/'

if not os.path.exists(fig_dir): os.mkdir(fig_dir)
if not os.path.exists(log_dir): os.mkdir(log_dir)
if not os.path.exists(out_dir): os.mkdir(out_dir)
if not os.path.exists('model_configs/'): os.mkdir('model_configs/')

def test_models(gpu,
                model_num,
                units=[32, 64, 128, 256],
                filters=[128, 256, 512, 1024],
                kernel_widths=[3, 5, 7, 9],
                reg_layers=[1, 2, 3, 4],
                seq_layers=[1, 2, 3, 4],
                concat_layers=[1, 2, 3, 4]):

    num_reg = 472
    seq_length = 1000

    reg_layers = int(np.random.choice(reg_layers, 1)[0])
    reg_units = np.random.choice(units, reg_layers).tolist()
    
    seq_layers = int(np.random.choice(seq_layers, 1)[0])
    filters = sorted(np.random.choice(filters, seq_layers).tolist(),
                   reverse=True)
    kernels = sorted(np.random.choice(kernel_widths, seq_layers).tolist())
    
    concat_layers = int(np.random.choice(concat_layers, 1)[0])
    concat_units = np.random.choice(units, concat_layers).tolist()
    
    config = {
        'model_num': model_num,
        'num_reg': num_reg,
        'seq_length': seq_length,
        'reg_layers': reg_layers,
        'reg_units': reg_units,
        'seq_layers': seq_layers,
        'filters': filters,
        'kernels': kernels,
        'concat_layers': concat_layers,
        'concat_units': concat_units
    }
    
    yaml.safe_dump(config,
                   open('model_configs/model' + str(model_num) + '.yaml', 'w'))

    with tf.device(gpu):
    
        # Build regression model
        print('INFO - %s'%('building regression model.'))
        reg_input = Input(shape=(num_reg,),dtype='float32',name='reg_input')
        x = reg_input
        n = 0

        while reg_layers > 1:
            x = Dense(reg_units[n], activation='relu')(x)

            n += 1
            reg_layers -= 1

        reg_output = Dense(reg_units[n], activation='relu')(x)

        # Build sequence model
        units = 512
        seq_input=Input(shape=(seq_length,4),dtype='float32',name='seq_input')
        x=seq_input
        
        n = 0
        
        while seq_layers > 0:
            x=RevCompConv1D(nb_filter=filters[n],filter_length=kernels[n],border_mode='valid')(x)
            x=RevCompConv1DBatchNorm()(x)
            x=Activation('relu')(x)
            x=MaxPooling1D(pool_length=4,border_mode='valid')(x)
            
            n += 1
            seq_layers -= 1
            
        seq_output=DenseAfterRevcompConv1D(output_dim=units)(x)

        # Build concatenate model
        print('INFO - %s'%('building concatenate model.'))
        
        x = Merge(mode='concat',concat_axis=1)([reg_output,seq_output])
        n = 0

        while concat_layers > 0:
            x = Dense(concat_units[n], activation='relu')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            x = Dropout(0.5, seed=42)(x)

            n += 1
            concat_layers -= 1
            
        rgs_output = Dense(1, activation='linear',name='rgs_output')(x)

        # Compile and train the model
        model = Model(output=[rgs_output], input=[reg_input, seq_input])
        model.compile(loss={'rgs_output':'mean_squared_error'},optimizer='sgd')
        
        print('INFO - %s'%('loading data.'))
        train,val,test=input_data.read_data_sets(train_pct=80,
                                                 val_pct=10,
                                                 test_pct=10,
                                                 mode='random',
                                                 seq_file='../../data/yeast_promoters.txt',
                                                 expr_file='../../data/complete_dataset.txt',
                                                 reg_names_file='../../data/reg_names_R.txt',
                                                 conv1d=True)
        
        print('INFO - %s'%('training model.'))
        reduce_lr=ReduceLROnPlateau(verbose=1,factor=0.5, patience=5)
        early_stopping=EarlyStopping(monitor='val_loss',patience=10)
        checkpoint=ModelCheckpoint(filepath="%s/model%03d.{epoch:02d}-{val_loss:.4f}.hdf5"%(log_dir, model_num),
            monitor='val_loss')
        history=model.fit({'seq_input':train['seq'],'reg_input':train['reg']},{'rgs_output':train['expr']},
            validation_data=({'seq_input':val['seq'],'reg_input':val['reg']},{'rgs_output':val['expr']}),
            nb_epoch=50,
            batch_size=100,
            callbacks=[early_stopping,checkpoint,reduce_lr],
            verbose=1)

        with open('%s/model%03d_history.pkl'%(log_dir),'wb') as f:
            pickle.dump([history.history],f)
        with open('%s/model%03d_history.pkl'%(log_dir),'rb') as f:
            x=pickle.load(f)


        # Plot the learning curve:
        history=pd.DataFrame(x[0])
        history['epoch']=(range(1,history.shape[0]+1))
        history_melt=pd.melt(history,id_vars=['epoch'],value_vars=['loss','val_loss'],var_name='type',value_name='loss')

        p1=ggplot(history_melt,aes('epoch','loss',color='type'))+geom_line()+theme_bw()
        p1.save(filename='%s/model%03d_learning_curve.png'%(fig_dir, model_num))


        # Plot prediction vs ground truth: 
        pred=model.predict({'seq_input':test['seq'],'reg_input':test['reg']},batch_size=100,verbose=1)
        plt.figure()
        plt.scatter(pred,test['expr'])
        plt.savefig("%s/model%03d_pred_vs_obs.png"%(fig_dir, model_num))
        output=np.column_stack((test['expr'], pred[:,0]))
        np.savetxt("%s/model%03d_prediction.txt"%(out_dir, model_num), output,delimiter='\t')


parser = argparse.ArgumentParser(description='Train neural network models with random search.')
parser.add_argument('num_models', type=int,
                    help='number of models to train')
parser.add_argument('--start', metavar='N', type=int,
                    help='starting model number', default=1)
parser.add_argument('--epochs', metavar='N', type=int,
                    help='number of epochs to train', default=100)
parser.add_argument('--gpu', metavar='N', type=int,
                    help='GPU to use', default=1)


if __name__ == '__main__':

    args = parser.parse_args()

    if args.gpu == 1:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        gpu = '/gpu:0'
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"
        gpu = '/gpu:1'

    model_num = args.start

    for i in range(args.num_models):

        test_models(gpu,
                    model_num,
                    units=[128, 256, 512, 1024],
                    filters=[32, 64, 128, 256, 512],
                    kernel_widths=[3, 5, 7, 9],
                    reg_layers=[1, 2, 3, 4],
                    seq_layers=[1, 2, 3, 4],
                    concat_layers=[1, 2, 3, 4])

        model_num += 1
    
