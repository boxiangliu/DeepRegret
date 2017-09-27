in_fn='../processed_data/concatenation/msn24_ko/msn24_ko_37C_diff_expr_rank.tsv'
fig_dir='../figures/concatenation/msn24_ko/'

data=read.table(in_fn,header=TRUE)
pdf(sprintf('%s/msn24_ko_37C_diff_expr_rank.pdf',fig_dir))
cor=cor(data[,1],data[,2],method='spearman')
smoothScatter(x=data[,1],y=data[,2],xlab='Observed difference (MSN2/4 - WT @ 37C)',ylab='Predicted difference (MSN2/4 - WT @ 37C)')
text(x=0,y=6000,pos=4,labels=sprintf('spearman cor=%0.3f',cor))
dev.off()