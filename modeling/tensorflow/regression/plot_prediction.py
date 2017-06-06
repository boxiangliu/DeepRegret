import pandas as pd
import matplotlib.pyplot as plt
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--pred',type=str,default='../processed_data/regression/prediction.txt',help='Prediction file.')
parser.add_argument('--fig',type=str,default='../figures/regression/pred_vs_obs.pdf',help='Output figure.')
args=parser.parse_args()
data=pd.read_table(args.pred,header=None,names=['Obs','Pred'],sep=' ')
plt.scatter(data.Obs,data.Pred)
plt.savefig(args.fig)
