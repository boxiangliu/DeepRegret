import argparse
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

# Functions: 
def tmp_processing(df):
	df=df.assign(accuracy=df.num_correct/df.num_example)
	df=df.assign(step=range(999,df.shape[0]*1000,1000))
	return df



parser=argparse.ArgumentParser(description='Plot prediction accuracy over time.')
parser.add_argument('--train',help='training accuracy file',default='../processed_data/dropout/train_accuracy.log')
parser.add_argument('--val',help='validation accuracy file',default='../processed_data/dropout/val_accuracy.log')
parser.add_argument('--test',help='test accuracy file',default='../processed_data/dropout/test_accuracy.log')
parser.add_argument('--fig',help='output figure file',default='../figures/dropout/accuracy.pdf')
args=parser.parse_args()

train=pd.read_table(args.train,header=None,names=['step','num_example','num_correct','accuracy'])
val=pd.read_table(args.val,header=None,names=['step','num_example','num_correct','accuracy'])
test=pd.read_table(args.test,header=None,names=['step','num_example','num_correct','accuracy'])

plt.plot(train.step,train.accuracy)
plt.plot(val.step,val.accuracy)
plt.plot(test.step,test.accuracy)
plt.legend(['train', 'validation', 'test'], loc='upper left')
plt.savefig(args.fig)

