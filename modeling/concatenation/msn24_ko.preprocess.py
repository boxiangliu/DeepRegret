import pandas as pd
from subprocess import call
in_fn='../data/yeast_promoters.txt'
out_fn='../data/yeast_promoters.fa'
out_dir='../processed_data/concatenation/msn24_ko/'

promoters=pd.read_table(in_fn,header=None,names=['gene','seq'])
with open(out_fn,'w') as f:
	for i in range(promoters.shape[0]):
		gene_name=promoters.iloc[i,0]
		seq=promoters.iloc[i,1]
		f.write(">%s\n%s\n"%(gene_name,seq))

cmd='fimo --oc %s/fimo/ --thresh %.02f %s %s'% \
	(out_dir,1.0,'../data/jaspar/msn24.meme',out_fn)
call(cmd,shell=True)

fimo=pd.read_table('%s/fimo/fimo.txt'%out_dir)

for i in range(fimo.shape[0]):
	gene=fimo.iloc[i,1]
	start=fimo.iloc[i,2]
	end=fimo.iloc[i,3]
	seq=promoters[promoters.gene==gene]['seq'].tolist()[0]
	orig=seq[(start-1):end]
	new='N'*(end-start+1)
	seq=seq.replace(orig,new)
	promoters.seq[promoters.gene==gene]=seq

promoters.to_csv('%s/yeast_promoters.msn24_ko.fa'%out_dir,sep='\t',header=None,index=None) 

count=0
for i in range(promoters.shape[0]):
	count+='AGGGG' in promoters.seq[i]
	count+='CCCCT' in promoters.seq[i]
assert count == 0

