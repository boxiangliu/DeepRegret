library(data.table)
library(ggplot2)
in_fn='../processed_data/keras/prediction.txt'
fig_dir='../figures/keras/'
if (!dir.exists(fig_dir)) dir.create(fig_dir,recursive=TRUE)
data=fread(in_fn,col.names=c('Observed','Predicted'))
fit=lm(Predicted~Observed,data=data)
pdf(sprintf('%s/pred_vs_obs.pdf',fig_dir),height=4,width=4)
ggplot(data,aes(Observed,Predicted))+geom_point()+stat_smooth(method='lm',formula=y~x-1)+theme_classic()+annotate(geom='text',x=-3,y=7,label=sprintf('y = %.4f + %.4f x\nr2 = %.4f',coefficients(fit)[1],coefficients(fit)[2],summary(fit)$adj.r.squared))
dev.off()