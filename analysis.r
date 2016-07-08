#***************************************************************************
# Author: Andrew Skeen
# Title: graphs for activity transpose data
# Date: 05/04/16
#***************************************************************************

# Libs
require(ggplot2)
require(data.table)
require(sqldf)
require(plyr)
require(gridExtra)
#require(RPostgreSQL)
require(bit64)
require(gridExtra)
require(scales)
library(RPostgreSQL)
options(sqldf.driver = "SQLite")


setwd("/home/andrew/bimbo/")

# clustering

# clients
prod=fread('producto_tabla.csv', data.table=T)
cliente=fread('cliente_tabla.csv', data.table=T)

#setkey(cliente, cols=c(names(cliente)[2]))
cliente=cliente[order(NombreCliente)]

cl_cnts=sqldf("select NombreCliente, sum(1) as total from cliente group by 1 order by 2 desc")

cl_cnts$grp<-cl_cnts$NombreCliente
cl_cnts$grp[11:nrow(cl_cnts)]<-"Rump"

write.csv(cl_cnts, 'client_grp.csv', row.names=F)

# products
prod<-as.data.frame(prod)

splits=t(sapply(prod$NombreProducto, function(y) {
  ind=regexpr("\\d",y)
  c(substr(y,1,ind-1), substr(y,ind,nchar(y)))
}))

prod$first<-splits[,1]
prod$second<-splits[,2]
prod$size=gsub("(?:^|^.*\\s)(\\d+[kgKGml]+)\\s.*$","\\1", prod$second)

prod$found<-(!prod$size==prod$second)

prod$first_low<-tolower(prod$first)

library(tm)
library(SnowballC)
library(skmeans)

CorpusShort <-VectorSource(prod$first_low)
CorpusShort<-Corpus(CorpusShort)
CorpusShort <- tm_map(CorpusShort, PlainTextDocument)

# Remove Punctuation
CorpusShort <- tm_map(CorpusShort, removePunctuation)

# Remove Stopwords
CorpusShort <- tm_map(CorpusShort, removeWords, stopwords("es"))

# Stemming
CorpusShort <- tm_map(CorpusShort, stemDocument, language="es")

# Create DTM
CorpusShort <- Corpus(VectorSource(CorpusShort))
dtmShort <- DocumentTermMatrix(CorpusShort)

# Delete Sparse Terms (all the words now)
sparseShort <- removeSparseTerms(dtmShort, 0.9999)
ShortWords <- as.data.frame(as.matrix(sparseShort))

# Create valid names
colnames(ShortWords) <- make.names(colnames(ShortWords))

non_zero=as.numeric(apply(ShortWords, 1, function(x){length(x)!=length(x[x==0])}))
non_zero=which(non_zero==1)

library(skmeans)
mat<-as.matrix(ShortWords[non_zero,])
for (k in 4:25){
  print(k)
  mod<-skmeans(mat, k, method="genetic")
  prod[,paste0("cluster_",k)] <-0
  prod[non_zero,paste0("cluster_",k)]<-mod$cluster
}

rm(mat)

# global mean
sum=0
num=0
for (k in 0:9){
  data=fread(paste0('train_',k,'.csv'), data.table=T)
  data[,`:=`(log_dem=log(1+Demanda_uni_equil))]
  global_mean=data[,.(sum=sum(log_dem), num=.N)]
  sum=sum+global_mean$sum
  num=num+global_mean$num
  rm(data)
  print(k)
}

global_mean=sum/num


# tss
sum=0
tss<-function(x){
  return((x-global_mean)**2)
}

for (k in 0:9){
  data=fread(paste0('train_',k,'.csv'), data.table=T)
  data[,`:=`(log_dem=log(1+Demanda_uni_equil))]
  tss_total=data[,lapply(.SD, tss), .SDcols=c("log_dem")]
  tss_total=tss_total[,.(total=sum(log_dem), num=.N)]
  sum=sum+tss_total$total
  rm(data, tss_total)
  print(k)
}

tss_total=sum

total_list=c()
for (j in 4:25){
  lst=list()
  for (k in 0:9){
    data=fread(paste0('train_',k,'.csv'), data.table=T)
    # merge cluster
    prod_dt<-as.data.table(prod)[,.SD,.SDcols=c("Producto_ID", paste0("cluster_",j))]
    data=merge(data, prod_dt, by.x='Producto_ID', by.y='Producto_ID', all.x=T)
    data[,`:=`(log_dem=log(1+Demanda_uni_equil))]
    wss_total=data[, .(clus_sum=mean(log_dem), clus_num=.N), by=c(paste0("cluster_",j))]
    lst[[length(lst)+1]]=wss_total
    rm(data)
  }
  
  wss_all=do.call(rbind, lst)
  
  wss_all[,sum(clus_sum*clus_num)/sum(clus_num), by=c(paste0("cluster_",j))]
  sum=0
  for (k in 0:9){
    data=fread(paste0('train_',k,'.csv'), data.table=T)
    prod_dt<-as.data.table(prod)[,.SD,.SDcols=c("Producto_ID", paste0("cluster_",j))]
    data=merge(data, prod_dt, by.x='Producto_ID', by.y='Producto_ID', all.x=T)
    
    data=merge(data, wss_total, by.x=paste0("cluster_",j), by.y=paste0("cluster_",j), all.x=T)
    data[,`:=`(log_dem=log(1+Demanda_uni_equil))]
    data[,`:=`(ss=(log_dem-clus_sum/clus_num)**2)]
    total=data[,sum(ss)]
    sum=sum+total
    rm(data)
  }
  total_list=c(total_list,sum)
  print(j)
}


ss=data.frame(num=4:25,total=total_list/tss_total)

write.csv(ss, file='ss.csv', row.names=F)

wss<-function(x){
  browser()
}

data[,.(ss=wss(log_dem)),by=paste0("cluster_",j)]

library(stringdist)

distmat<-stringdistmatrix(prod$first_low[prod$first_low!=''], prod$first_low[prod$first_low!=''], method="hamming")
mod=kmeans(distmat, 30)



out<-do.call("rbind",
             strsplit(gsub("^([a-zA-Z\\s]*)\\d?.*?(\\d+[kgKGml]{1,2}).*$", "\\1 \\2",prod$NombreProducto ), " "))



test=fread('test.csv', data.table=T)
train=fread('train_0.csv', data.table=T)
#train1=fread('train_1.csv', data.table=F)


# multiples on client, product and week

multiples=train[,list("total"=.N), by=c("Cliente_ID","Producto_ID", "Semana")]
multiples[order(Cliente_ID,Producto_ID),]

train<-merge(train, multiples, by.x=c("Cliente_ID","Producto_ID", "Semana"), by.y=c("Cliente_ID","Producto_ID", "Semana"), all.x=F, all.y=F)
train<-train[total==1,,]

train<-train[order(Cliente_ID,Producto_ID, Semana)]
train[,`:=`(demand_m1=shift(Demanda_uni_equil, type="lag"), 
            sales_m1=shift(Venta_uni_hoy, type="lag"),
            ret_m1=shift(Dev_uni_proxima, type="lag")),by=c("Cliente_ID","Producto_ID")]

train[,`:=`(price=Venta_hoy/Venta_uni_hoy)]

a<-ggplot(data=train[runif(nrow(train),0,1)<0.1], aes(x=demand_m1, y=Demanda_uni_equil))+geom_point()
a


multiples=sqldf("select Cliente_ID, Producto_ID, Semana, sum(1) as total from train group by 1,2,3")

train<-sqldf("select a.* from train as a join multiples as b  on a.Cliente_ID=b.Cliente_ID and
             a.Producto_ID=b.Producto_ID and a.Semana=b.Semana where b.total=1")

train<-sqldf(c("create index idx on train(Cliente_ID, Producto_ID)", "select 
        a.*, b.Demanda_uni_equil as demand_m1, b.Venta_uni_hoy as sales_m1
from train as a join train as b on
a.Cliente_ID=b.Cliente_ID and 
a.Producto_ID=b.Producto_ID  where a.Semana=b.Semana+1"))

train$change<-train$Demanda_uni_equil/train$demand_m1
train$change_s<-train$sales_m1/train$demand_m1

train$change_ind<-factor(ifelse(train$change_s>=1,0,1))

a<-ggplot(data=train, aes(x=change, y=change_ind))+geom_density(alpha=0.3)
a

#agg=sqldf("select Semana, Producto_ID, avg(Demanda_uni_equil) as demand from train group by 1,2")
agg<-ddply(train, .(Semana, Producto_ID), function(x){
  ret=log(quantile(x$change, seq(0,1,by=0.2), na.rm=T))
  names(ret)<-paste0("q_", 0:5)
  ret  
})

agg$Producto_ID<-as.factor(agg$Producto_ID)

agg<-melt(agg, id.vars=c("Semana", "Producto_ID"), measure.vars=paste0("q_", 0:5), variable.name="var", value.name="val")

a<-ggplot(data=agg, aes(x=Semana,y=val, col=var))+geom_line()+
  facet_wrap(~Producto_ID)
a

train$Semana<-as.factor(train$Semana)
train$Canal_ID<-as.factor(train$Canal_ID)

a<-ggplot(data=train[train$change<=quantile(train$change,0.99, na.rm=T),], aes(x=change, fill=Semana))+geom_density(alpha=0.3)+
  facet_wrap(~Canal_ID)
a


train$outlier<-0
train$outlier[train$change>quantile(train$change, 0.99, na.rm=T)]<-1

train$outlier<-as.factor(train$outlier)

train$log_sales<-log(1+train$Venta_uni_hoy)
train$log_returns<-log(1+train$Dev_uni_proxima)

a<-ggplot(data=train, aes(x=log_sales, fill=outlier))+geom_density(alpha=0.3)
a


a<-ggplot(data=train, aes(x=log_returns, fill=outlier))+geom_density(alpha=0.3)
a

nulls<-train[is.na(train$change),]





na.rm = 



a<-ggplot(data=agg, aes(x=Semana,y=demand, col=Producto_ID))+geom_line()
a

agg=prop.table(table(train$Semana, train$Demanda_uni_equil), margin=1)


Semana
train$Demanda_uni_equil



tt=test[test$Producto_ID==1284,]
