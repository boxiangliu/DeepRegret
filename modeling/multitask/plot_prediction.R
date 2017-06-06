library(data.table)
library(ggplot2)
library(gridExtra)
in_fn='../processed_data/multitask/regression.txt'
fig_dir='../figures/multitask/'
if (!dir.exists(fig_dir)) dir.create(fig_dir,recursive=TRUE)
data=fread(in_fn,col.names=c('Observed','Predicted'))
fit=lm(Predicted~Observed,data=data)
pdf(sprintf('%s/pred_vs_obs.pdf',fig_dir),height=4,width=4)
ggplot(data,aes(Observed,Predicted))+geom_point()+stat_smooth(method='lm',formula=y~x-1)+theme_classic()+annotate(geom='text',x=-3,y=7,label=sprintf('y = %.4f + %.4f x\nr2 = %.4f',coefficients(fit)[1],coefficients(fit)[2],cor(data$Predicted,data$Observed)))
dev.off()

in_fn='../processed_data/multitask/classification.txt'
data=fread(in_fn,col.names=c('Observed','Predicted'))
t=formatC(100*table(data)/nrow(data),digits=4)
rownames(t)=colnames(t)=c('Down','Baseline','Up')
pdf(sprintf('%s/confusion.pdf',fig_dir),height=4,width=4)
grid.table(t)
dev.off()
