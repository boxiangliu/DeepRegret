import os
import sys
sys.path.append('utils/')
import input_data_config as input_data
from heatmap import heatmap
from keras.models import load_model
import numpy as np
import pandas as pd
from scipy.stats import rankdata

import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
import statsmodels.stats.api as sms

reg_names_file='../data/reg_names_R.txt'
seq_dir='../processed_data/concatenation/msn24_ko/'
fig_dir='../figures/concatenation/msn24_ko/'
out_dir='../processed_data/concatenation/msn24_ko/'

if not os.path.exists(fig_dir): os.makedirs(fig_dir)
if not os.path.exists(out_dir): os.makedirs(out_dir)

def difference_to_matrix(x,y,full_data):
	diff=pd.DataFrame(y-x,columns=['diff'])
	diff['uid']=full_data['uid']
	diff['experiment']=full_data['experiment']
	diff=diff.pivot(index='uid',columns='experiment',values='diff')
	diff_t=diff.transpose()
	return diff,diff_t

def hclust(x,y,full_data):
	_,diff_t=difference_to_matrix(x, y, full_data)

	z=linkage(diff_t,'ward')
	plt.figure(figsize=(25, 10))
	plt.title('Hierarchical Clustering Dendrogram')
	plt.xlabel('sample index')
	plt.ylabel('distance')
	dendrogram(z,leaf_rotation=90,labels=diff_t.index)
	plt.tight_layout()


def plot_orig_vs_perturbed(x,y,ylab):
	plt.figure()
	plt.scatter(x,y)
	plt.xlabel('Original')
	plt.ylabel(ylab)


# wildtype model:
model_fn='../logs/concatenation/regression/model.41-0.2300.hdf5'
model=load_model(model_fn)
full_data,_,_=input_data.read_data_sets(train_pct=100,val_pct=0,test_pct=0)
pred=model.predict({'seq_input':full_data['seq'],'reg_input':full_data['reg']},batch_size=100,verbose=1)



# MSN2/4 motif KO:
full_data,_,_=input_data.read_data_sets(train_pct=100,val_pct=0,test_pct=0,seq_file='%s/yeast_promoters.msn24_ko.fa'%(seq_dir))
pred_2=model.predict({'seq_input':full_data['seq'],'reg_input':full_data['reg']},batch_size=100,verbose=1)

plot_orig_vs_perturbed(pred, pred_2, 'MSN2/4 motif KO')
plt.savefig('%s/orig_vs_msn24_motif_ko.png'%fig_dir)

hclust(pred, pred_2, full_data)
plt.savefig('%s/hclust_msn24_motif_ko.pdf'%fig_dir)


# MSN2/4 motif KO + expr baseline: 
full_data,_,_=input_data.read_data_sets(train_pct=100,val_pct=0,test_pct=0,seq_file='%s/yeast_promoters.msn24_ko.fa'%(seq_dir))
reg_names = pd.read_table(reg_names_file, names = ["UID"])
# MSN2 / YMR037C
# MSN4 / YKL062W
for gene in ['YMR037C','YKL062W']:
	idx=reg_names[reg_names.UID == gene].index[0]
	full_data['reg'][:,idx]=0
pred_3=model.predict({'seq_input':full_data['seq'],'reg_input':full_data['reg']},batch_size=100,verbose=1)


plot_orig_vs_perturbed(pred, pred_3, ylab='MSN2/4 motif KO + expr baseline')
plt.savefig('%s/orig_vs_msn24_motif_ko_expr_baseline.png'%fig_dir)

hclust(pred, pred_3, full_data)
plt.savefig('%s/hclust_msn24_motif_ko_expr_baseline.pdf'%fig_dir)


# MSN2/4 motif KO + expr KO: 
full_data,_,_=input_data.read_data_sets(train_pct=100,val_pct=0,test_pct=0,seq_file='%s/yeast_promoters.msn24_ko.fa'%(seq_dir))
reg_names = pd.read_table(reg_names_file, names = ["UID"])
# MSN2 / YMR037C
# MSN4 / YKL062W

for gene in ['YMR037C','YKL062W']:
	idx=reg_names[reg_names.UID == gene].index[0]
	full_data['reg'][:,idx]-=5.0
pred_4=model.predict({'seq_input':full_data['seq'],'reg_input':full_data['reg']},batch_size=100,verbose=1)


plot_orig_vs_perturbed(pred, pred_4, ylab='MSN2/4 motif KO + expr KO')
plt.savefig('%s/orig_vs_msn24_motif_ko_expr_ko.png'%fig_dir)


hclust(pred, pred_4, full_data)
plt.savefig('%s/hclust_msn24_motif_ko_expr_ko.pdf'%fig_dir)


# MSN2/4 knockout in normal vs. heat shock conditions:
diff,_=difference_to_matrix(pred, pred_4, full_data)

msn2_targets=pd.read_csv('../data/sgd/MSN2_targets.txt',comment='!',delimiter='\t')
msn4_targets=pd.read_csv('../data/sgd/MSN4_targets.txt',comment='!',delimiter='\t')
msn24_targets=pd.concat([msn2_targets,msn4_targets])
msn24_targets_idx=[x in msn24_targets['Target Systematic Name'].tolist() for x in diff.index]
not_msn24_targets_idx=[(not x) for x in msn24_targets_idx]

experiment=['heat shock 17 to 37, 20 minutes',
			'heat shock 21 to 37, 20 minutes',
			'heat shock 25 to 37, 20 minutes',
			'heat shock 29 to 37, 20 minutes',
			'heat shock 33 to 37, 20 minutes',
			'steady state 15 dec C ct-2',
			'steady state 17 dec C ct-2',
			'steady state 21 dec C ct-2',
			'steady state 25 dec C ct-2',
			'steady state 29 dec C ct-2',
			'steady state 33 dec C ct-2',
			'steady state 36 dec C ct-2',
			'steady state 36 dec C ct-2 (repeat hyb)',
			'17 deg growth ct-1',
			'21 deg growth ct-1',
			'25 deg growth ct-1',
			'29 deg growth ct-1',
			'37 deg growth ct-1',
			'YPD 2 h ypd-2',
			'YPD 4 h ypd-2',
			'YPD 6 h ypd-2',
			'YPD 8 h ypd-2',
			'YPD 10 h  ypd-2',
			'YPD 12 h ypd-2',
			'YPD 1 d ypd-2',
			'YPD 2 d ypd-2',
			'YPD 3 d ypd-2',
			'YPD 5 d ypd-2',
			'YPD stationary phase 2 h ypd-1',
			'YPD stationary phase 4 h ypd-1',
			'YPD stationary phase 8 h ypd-1',
			'YPD stationary phase 12 h ypd-1',
			'YPD stationary phase 1 d ypd-1',
			'YPD stationary phase 2 d ypd-1',
			'YPD stationary phase 3 d ypd-1',
			'YPD stationary phase 5 d ypd-1',
			'YPD stationary phase 7 d ypd-1',
			'YPD stationary phase 13 d ypd-1',
			'YPD stationary phase 22 d ypd-1',
			'YPD stationary phase 28 d ypd-1']

diff_msn24_targets=[]
diff_not_msn24_targets=[]

for e in experiment:
	temp=diff[e][not_msn24_targets_idx]
	diff_not_msn24_targets.append(temp)
	temp=diff[e][msn24_targets_idx]
	diff_msn24_targets.append(temp)

import matplotlib.patches as mpatches
patch_msn24_targets = mpatches.Patch(color='C1', label='MSN2/4 targets')
patch_not_msn24_targets = mpatches.Patch(color='C0', label='Other genes')
plt.figure(figsize=(20,10))
plt.violinplot(diff_not_msn24_targets,[x - 0.25 for x in range(1,len(experiment)*2,2)],showmeans=True)
plt.violinplot(diff_msn24_targets,[x + 0.25 for x in range(1,len(experiment)*2,2)],showmeans=True)
plt.xlabel('')
plt.ylabel('Log2 Fold change',fontsize=25)
plt.xticks(range(1,len(experiment)*2,2),experiment,rotation=90,fontsize=20)
plt.yticks(fontsize=20)
plt.legend(handles=[patch_msn24_targets,patch_not_msn24_targets])
plt.tight_layout()
plt.savefig('%s/msn24_ko_heat_shock_vs_steady_state.pdf'%fig_dir)


effect=[]
pval=[]
se=[]
for i in range(len(diff_msn24_targets)):
	temp=diff_not_msn24_targets[i].mean()-diff_msn24_targets[i].mean()
	effect.append(temp)
	ci=sms.CompareMeans(
		sms.DescrStatsW(diff_not_msn24_targets[i].tolist()),
		sms.DescrStatsW(diff_msn24_targets[i].tolist())).tconfint_diff(usevar='unequal')
	temp=(ci[1]-ci[0])/float(2)
	se.append(temp)

plt.figure(figsize=(20,10))
plt.axhline(y=0.0,color='r',linestyle='--')
plt.errorbar(range(len(experiment)), effect,yerr=se,fmt='o')
plt.xlabel('')
plt.ylabel('Mean log2 Fold change',fontsize=25)
plt.xticks(range(len(experiment)),experiment,rotation='vertical',fontsize=20)
plt.yticks(fontsize=20)
plt.tight_layout()
plt.savefig('%s/msn24_ko_heat_shock_vs_steady_state.diff.pdf'%fig_dir)


# Predicted vs microarray:
wildtype_37C_idx=np.where(full_data['experiment']=='DBY7286 37degree heat - 20 min')[0]
msn24_ko_37C_idx=np.where(full_data['experiment']=='DBYmsn2-4- 37degree heat - 20 min')[0]

assert wildtype_37C_idx.shape[0]==6107
assert msn24_ko_37C_idx.shape[0]==6107

wildtype_37C=full_data['expr'][wildtype_37C_idx]
msn24_ko_37C=full_data['expr'][msn24_ko_37C_idx]
msn24_ko_37C_pred=pred_4[wildtype_37C_idx]

_,(ax1,ax2)=plt.subplots(1,2,sharex=True,sharey=True,figsize=c())
ax1.plot([-4,6],[-4,6], ls="--", c=".3")
ax1.scatter(wildtype_37C,msn24_ko_37C)
ax1.set_xlim([-4,6])
ax1.set_ylim([-4,6])
ax1.set_xlabel('Wild-type at 37C')
ax1.set_ylabel('MSN2/4 KO at 37C')
ax1.set_title('Microarray')

ax2.plot([-4,6],[-4,6], ls="--", c=".3")
ax2.scatter(wildtype_37C,msn24_ko_37C_pred)
ax2.set_xlabel('Wild-type at 37C')
ax2.set_ylabel('MSN2/4 KO at 37C (predicted)')
ax2.set_title('Prediction')
plt.savefig('%s/wildtype_vs_msn24_ko_at_37C.pdf'%fig_dir)


_,ax=plt.subplots()
ax.scatter(msn24_ko_37C,msn24_ko_37C_pred)
ax.set_xlabel('Microarray')
ax.set_ylabel('Prediction')
plt.savefig('%s/microarry_vs_prediction_msn24_ko_at_37C.pdf'%fig_dir)


# Rank genes based on changes in gene expression:
msn24_ko_diff_expr_rank=rankdata(msn24_ko_37C-wildtype_37C)
msn24_ko_pred_diff_expr_rank=rankdata(msn24_ko_37C_pred.transpose()-wildtype_37C)
out=pd.DataFrame({
	'msn24_ko_diff_expr_rank':msn24_ko_diff_expr_rank,
	'msn24_ko_pred_diff_expr_rank':msn24_ko_pred_diff_expr_rank
	})

out.to_csv('%s/msn24_ko_37C_diff_expr_rank.tsv'%(out_dir),sep='\t',index=False,index_label=False)

# _,ax=plt.subplots()
# ax.scatter(msn24_ko_diff_expr_rank,msn24_ko_pred_diff_expr_rank,alpha=0.05)