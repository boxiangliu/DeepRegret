library(data.table)
library(cowplot)
library(limma)
library(dplyr)

# Functions:
scatterplot=function(x,y,xlab,ylab){
	fit=lm(formula=y~x)
	df=data.frame(x=x,y=y)
	correlation=with(df,cor(x,y))
	p=ggplot(df,aes(x,y))+geom_point()+stat_smooth(method='lm',formula=y~x)+annotate(geom='text',x=-2,y=4,label=sprintf('y = %.3f + %.3f * x',coefficients(fit)[1],coefficients(fit)[2]))+annotate(geom='text',x=-2,y=3.5,label=sprintf('R2: %.3f',correlation))+xlab(xlab)+ylab(ylab)
	return(p)
}


# Read and transform data:
kd=fread('../processed_data/regression/knockdown_prediction.kd_10.txt',col.names=c('y_','y','UID','experiment','knockdown'))
kd_wide=dcast(kd,UID~knockdown,value.var='y')
kd_concat=melt(kd_wide,id.vars=c('UID','WT'),variable.name='knockdown',value.name='mutant')
kd[UID=='YIR019C'&knockdown=='YMR070W',]
kd[UID=='YIR019C'&knockdown=='YOL116W',]
kd[UID=='YIR019C'&knockdown=='YDR043C',]


# Make scatterplot of observed vs predicted fold change, 
# and of knockdown vs wildtype fold change. 
p1=scatterplot(kd[knockdown=='WT',y_],kd[knockdown=='WT',y],xlab='Observed',ylab='Predicted')
p2=scatterplot(kd_wide[,WT],kd_wide[,YAL013W],'wild-type','YAL013W knockdown')
pdf('../figures/interpretation/obs_vs_pred_vs_knockdown.pdf')
print(p1)
print(p2)
dev.off()
