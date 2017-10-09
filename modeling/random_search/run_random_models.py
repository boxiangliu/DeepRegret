from keras.models import Model
from keras.layers import Input, Dense, Conv2D, Flatten, Dropout, Concatenate, Activation, MaxPooling2D
from keras.optimizers import Adam, SGD, RMSprop
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.utils import plot_model

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

import tensorflow as tf

from joblib import Parallel, delayed
import yaml
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
                units=[128, 256, 512, 1024],
                filters=[32, 64, 128, 256, 512],
                kernel_widths=[3, 5, 7, 9, 15, 19],
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
    kernels = sorted(np.random.choice(kernel_widths, seq_layers).tolist(),
                   reverse=True)
    
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
        print('INFO - %s'%('building sequence model.'))
        seq_input = Input(shape=(4,seq_length,1),dtype='float32',name='seq_input')
        n = 0
        x = Conv2D(kernel_size=(4, kernels[n]),
                       filters=filters[n],
                       strides=(1,1))(seq_input)

        while seq_layers > 0:
            if n > 0:
                x = Conv2D(kernel_size=(1, kernels[n]),
                           filters=filters[n],
                           strides=(1,1))(x)
            x = Activation('relu')(x)
            x = MaxPooling2D(pool_size=(1, 4),
                             strides=(1, 4),
                             padding="valid")(x)

            n += 1
            seq_layers -= 1

        seq_output = Flatten()(x)

        # Build concatenate model
        print('INFO - %s'%('building concatenate model.'))
        units=512
        x = Concatenate(axis=-1)([reg_output, seq_output])
        
        n = 0

        while concat_layers > 0:
            x = Dense(concat_units[n], activation='relu')(x)
            x = Dropout(0.5, seed=42)(x)

            n += 1
            concat_layers -= 1

        rgs_output=Dense(1, activation='linear',name='rgs_output')(x)

        # Compile and train the model
        model=Model(outputs=[rgs_output], inputs=[reg_input,seq_input])
        model.compile(loss={'rgs_output':'mean_squared_error'},optimizer='sgd')
#         plot_model(model, show_shapes=True, to_file='%s/model.png'%(fig_dir))

        print('INFO - %s'%('loading data.'))
        train,val,test=input_data.read_data_sets(train_pct=80,
                                                 val_pct=10,
                                                 test_pct=10,
                                                 mode='whole_gene',
                                                 seq_file='../../data/yeast_promoters.txt',
                                                 expr_file='../../data/complete_dataset.txt',
                                                 reg_names_file='../../data/reg_names_R.txt')

        print('INFO - %s'%('training model.'))
        reduce_lr=ReduceLROnPlateau(verbose=1,factor=0.5, patience=5)
        early_stopping=EarlyStopping(monitor='val_loss',patience=10)
        checkpoint=ModelCheckpoint(filepath="%s/model%d.{epoch:02d}-{val_loss:.4f}.hdf5"%(log_dir, model_num), monitor='val_loss')
        batchhistory=BatchHistory(val_data=val,loss_function='mse',every_n_batch=1000)
        history=model.fit({'seq_input':train['seq'],'reg_input':train['reg']},{'rgs_output':train['expr']},
            validation_data=({'seq_input':val['seq'],'reg_input':val['reg']},{'rgs_output':val['expr']}),
            epochs=100,
            batch_size=100,
            callbacks=[early_stopping,checkpoint,reduce_lr,batchhistory],
            verbose=1)
        with open('{}/model{}_history.pkl'.format(log_dir, model_num),'wb') as f:
            pickle.dump([history.history,batchhistory.val_loss],f)


        pred=model.predict({'seq_input':test['seq'],'reg_input':test['reg']},batch_size=100,verbose=1)
        plt.scatter(pred,test['expr'])
        plt.savefig("{}/model{}_pred_vs_obs.png".format(fig_dir, model_num))
        output=np.column_stack((test['expr'], pred[:,0]))
        np.savetxt('{}/model{}_prediction.txt'.format(out_dir, model_num), output, delimiter='\t')


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
                    filters=[128, 256, 512, 1024],
                    kernel_widths=[3, 5, 7, 9, 15, 19],
                    reg_layers=[1, 2, 3, 4],
                    seq_layers=[1, 2, 3, 4],
                    concat_layers=[1, 2, 3, 4])

        model_num += 1
    
