library(data.table)
library(gplots)
library(stringr)

msn2_fn='../processed_data/concatenation/concat.class.deeplift/msn2.tsv'
msn2=fread(msn2_fn)
setDF(msn2)
rownames(msn2)=msn2$experiment
msn2$experiment=NULL
pdf('1.pdf',height=20)
heatmap.2(as.matrix(msn2),trace='none')
dev.off()
msn2_2=msn2[rownames(msn2)!='Msn2 overexpression (repeat)',]

pdf('2.pdf',height=20)
heatmap.2(as.matrix(msn2_2),trace='none')
dev.off()


msn2_3=msn2[str_detect(rownames(msn2),'diauxic'),]


pdf('3.pdf',height=20)
heatmap.2(as.matrix(msn2_3),trace='none',margin=c(10,20))
dev.off()
