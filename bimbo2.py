# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 11:39:24 2016

@author: andrew
"""

import re
import gzip as gz
import numpy as np
import pdb
import pymc3 as pm
import seaborn as sns

#import datetime
import pandas as pd

path="/home/andrew/bimbo/"

data=pd.read_csv(path+'train_8.csv')

#sample
data=data[np.random.rand(data.shape[0])<0.1]

"""
cnts=data.groupby(['Cliente_ID', 'Producto_ID', 'Semana', 'Ruta_SAK', 'Agencia_ID', 'Canal_ID'])['Dev_uni_proxima'].count()
cnts.sort_values(ascending=False, inplace=True)
cnts=cnts.reset_index()
cnts.rename(columns={'Dev_uni_proxima': 'cnt'}, inplace=True)

# first look at those with multiples

mults=cnts[cnts.cnt>1]

join_cols=mults.columns[:-1]

mult_dat=pd.merge(data, mults,  left_on=list(join_cols), right_on=list(join_cols))

mult_dat.sort_values(by=list(join_cols), ascending=[True]*len(join_cols), inplace=True)

# per client, product and day are there missing values

# test for nulls
cols=['Cliente_ID', 'Producto_ID', 'Semana', 'Ruta_SAK', 'Agencia_ID', 'Canal_ID']
nulls=data[cols].isnull()
[(c,data[c].ix[nulls[c]].shape[0]) for i, c in enumerate(cols)]



mn=data.groupby(['Cliente_ID', 'Producto_ID', 'Semana', 'Ruta_SAK', 'Agencia_ID', 'Canal_ID'])['Dev_uni_proxima'].\
agg({'avg': lambda x: x.mean()})

#mn.reset_index(inplace=True)

mn=mn.unstack(level=2)
mn=mn.reset_index(inplace=True)

"""
mn=mn.pivot_table(index=['Cliente_ID', 'Producto_ID', 'Ruta_SAK', 'Agencia_ID', 'Canal_ID'], \
columns='Semana', values='avg')
"""

qnt=mn.avg.quantile(0.5, axis=1)
summary={col: [len(data[col].unique()), data[col].isnull().sum()] for col in data}

"""

# get demand lagged by one week
cols=['Cliente_ID', 'Producto_ID', 'Semana', 'Ruta_SAK', 'Agencia_ID', 'Canal_ID']
#data.sort_values(cols, inplace=True)

def shift_it(dd):
    dd.sort_values('Semana')
    dd['Demanda_uni_equil'].shift(1)
    
#[['Semana', 'Demanda_uni_equil']]
data['demand_m1']=data.groupby([c for c in cols if c!='Semana']).shift(1)
data['demand_ratio']=data['Demanda_uni_equil']/(data['demand_m1'])
data['null_flag']=((data.demand_ratio.isnull()) & (data.demand_ratio.isnull())).astype('int')

data[(data.null_flag==0) ].ix[:, 'demand_ratio'].quantile(q=np.linspace(0,0.999,1000))

from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import auc
from sklearn.metrics import roc_curve

clf = RandomForestRegressor(n_estimators=100, n_jobs=-1, verbose=1)
features=[c for c in cols if c!='Semana']
target='demand_ratio'

# use a full grid over all parameters
param_dist = {"max_depth": [3, None],
              "max_features": sp_randint(floor(0.5*sqrt(len(features))), floor(2*sqrt(len(features)))),
              "min_samples_split": sp_randint(2, 20),
              "min_samples_leaf": sp_randint(1, 20)}

# run randomized search
n_iter_search = 5
random_search = RandomizedSearchCV(clf, param_distributions=param_dist, n_iter=n_iter_search)
random_search.fit(data[features].ix[data.null_flag==0,:], data[target].ix[(data.null_flag==0) & \
(data[target]<=data[target].quantile(0.992))])



# Utility function to report best scores
def report(grid_scores, n_top=3):
    top_scores = sorted(grid_scores, key=itemgetter(1), reverse=True)[:n_top]
    for i, score in enumerate(top_scores):
        print("Model with rank: {0}".format(i + 1))
        print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
              score.mean_validation_score,
              np.std(score.cv_validation_scores)))
        print("Parameters: {0}".format(score.parameters))
        print("")
    return(top_scores)

scores=report(random_search.grid_scores_)

kw=scores[0].parameters
kw2=kw.copy()
kw2.update({'n_estimators':500, 'n_jobs':7, 'verbose':1})


d=data[((data.Producto_ID==1284) &(data.null_flag==0))]

thresh=0.995
qnt=d.demand_ratio.quantile(thresh)

#g=sns.FacetGrid(d[d.demand_ratio<=qnt], row='Semana', col='Canal_ID')
#g.map(sns.distplot, 'demand_ratio', hist=False)


# log demand ratio
d['log_d_ratio']=np.log(1+d.demand_ratio)
d=d[d['log_d_ratio']<d['log_d_ratio'].quantile(0.997)]

del data

with pm.Model() as hm:
    # priors for reg coefficients
    a=pm.Normal('a_int', mu=0, sd=100)
    b=pm.Normal('b_slope', mu=0, sd=100)
    # estimate of log demand ratio
    log_d_est=a+b*d['Semana']
    
    # student-t parameters
    sigma=pm.Flat('sigma')
    
    # define prior for Student T degrees of freedom
    #nu = pm.DiscreteUniform('nu', lower=1, upper=100)

    lh=pm.StudentT('likelihood', mu=log_d_est, sd=sigma, nu=30, observed=d['log_d_ratio'])

from scipy import optimize

with hm:
    # MAP
    start_MAP = pm.find_MAP(fmin=optimize.fmin_powell, disp=True)
    
    # two-step sampling to allow Metropolis for nu (which is discrete)
    step1 = pm.NUTS([a,b])
    #step2 = pm.Metropolis([nu])

    # take samples
    traces_studentt = pm.sample(2000, start=start_MAP, step=step1, progressbar=True)




def read_file(f_obj):
    for line in f_obj:
        yield line
   
"""     
def read_file(file_object):
    while True:
        data = file_object.readline()
        if not data:
            break
        yield data
"""
path='/home/andrew/bimbo/'

outfile='sample.txt'

#******************************************************************************
# header translation
#******************************************************************************

uni2ascii = {
            ord('\xe2\x80\x99'.decode('utf-8')): ord("'"),
            ord('\xe2\x80\x9c'.decode('utf-8')): ord('"'),
            ord('\xe2\x80\x9d'.decode('utf-8')): ord('"'),
            ord('\xe2\x80\x9e'.decode('utf-8')): ord('"'),
            ord('\xe2\x80\x9f'.decode('utf-8')): ord('"'),
            ord('\xc3\xa9'.decode('utf-8')): ord('e'),
            ord('\xe2\x80\x9c'.decode('utf-8')): ord('"'),
            ord('\xe2\x80\x93'.decode('utf-8')): ord('-'),
            ord('\xe2\x80\x92'.decode('utf-8')): ord('-'),
            ord('\xe2\x80\x94'.decode('utf-8')): ord('-'),
            ord('\xe2\x80\x94'.decode('utf-8')): ord('-'),
            ord('\xe2\x80\x98'.decode('utf-8')): ord("'"),
            ord('\xe2\x80\x9b'.decode('utf-8')): ord("'"),

            ord('\xe2\x80\x90'.decode('utf-8')): ord('-'),
            ord('\xe2\x80\x91'.decode('utf-8')): ord('-'),

            ord('\xe2\x80\xb2'.decode('utf-8')): ord("'"),
            ord('\xe2\x80\xb3'.decode('utf-8')): ord("'"),
            ord('\xe2\x80\xb4'.decode('utf-8')): ord("'"),
            ord('\xe2\x80\xb5'.decode('utf-8')): ord("'"),
            ord('\xe2\x80\xb6'.decode('utf-8')): ord("'"),
            ord('\xe2\x80\xb7'.decode('utf-8')): ord("'"),

            ord('\xe2\x81\xba'.decode('utf-8')): ord("+"),
            ord('\xe2\x81\xbb'.decode('utf-8')): ord("-"),
            ord('\xe2\x81\xbc'.decode('utf-8')): ord("="),
            ord('\xe2\x81\xbd'.decode('utf-8')): ord("("),
            ord('\xe2\x81\xbe'.decode('utf-8')): ord(")")}

def unicodetoascii(text):
    return text.decode('utf-8').translate(uni2ascii).encode('ascii')

txt='''Semana — Week number (From Thursday to Wednesday)
    Agencia_ID — Sales Depot ID
    Canal_ID — Sales Channel ID
    Ruta_SAK — Route ID (Several routes = Sales Depot)
    Cliente_ID — Client ID
    NombreCliente — Client name
    Producto_ID — Product ID
    NombreProducto — Product Name
    Venta_uni_hoy — Sales unit this week (integer)
    Venta_hoy — Sales this week (unit: pesos)
    Dev_uni_proxima — Returns unit next week (integer)
    Dev_proxima — Returns next week (unit: pesos)
    Demanda_uni_equil — Adjusted Demand (integer) (This is the target you will predict)'''

txt=unicodetoascii(txt)

txt=txt.split('\n')
    
capture=[re.findall(r'([a-zA-Z_]+)\s*\-\s*([a-zA-Z\s]+)\s*[^a-zA-Z\s]*.*$', t)[0] for t in txt]
capture={re.sub('\s','_', elt[1]): elt[0] for elt in capture}

#******************************************************************************
# read file seuqntially
#******************************************************************************

keys=['Client_ID', 'Product_ID', 'Adjusted_Demand_']

summary={}
with open(path+'train.csv') as f:
    rows=0
    for line in read_file(f):
        line=line.strip().split(',')
        if rows==0:
            ids=[ [index for index, val in enumerate(line) if capture[k]==val][0] for k in keys]
        else:
            client, product, demand=[line[i] for i in ids]
            summary[('a',product)]=summary.get(('a',product), 0)+float(demand)
        #pdb.set_trace()
        rows+=1
        if rows%500000==0:
            print('Done %s' % (str(rows)))
            #print(len(summary.keys()))

import pandas as pd
index, values=zip(*summary.items())

summary_s=pd.Series(values, index=index)

summary_s.sort_values(ascending=False, inplace=True)

_,products=zip(*summary_s.index)

summary_s=pd.DataFrame({'prod': products, 'cnt': summary_s.values})


summary_s['cumsum']=summary_s.cnt.cumsum()

total=summary_s.cumsum.max()

num_files=10

from math import floor, ceil
overall={}
prod_list=[]
file_num=0
for row in xrange(summary_s.shape[0]):
    if (file_num==9):
        #pdb.set_trace()
        pass
    if (summary_s['cumsum'].ix[row]<= (file_num+1)*ceil(total/num_files)) :
        prod_list.append(summary_s['prod'].ix[row])
    else:
        overall[file_num]=prod_list
        file_num+=1
        prod_list=[summary_s['prod'].ix[row]]
overall[file_num]=prod_list 

overall=[([key]*len(val), val) for key, val in overall.items()]

overall=[pd.DataFrame({'grp':key, 'prod': val}) for key, val in overall]

overall_d=reduce(lambda x,y: pd.concat((x,y), axis=0), overall)
overall_d['prod']=overall_d['prod'].astype('int')

overall_d={int(x[1]): x[0] for x in overall_d.values }


file_obj=[open(path+'train_%s.csv'%(str(k)), 'w') for k in xrange(10)]


with open(path+'train.csv') as f:
    rows=0
    for line in read_file(f):
        if rows==0:
            header=line
            h_split=header.strip().split(',')
            #pdb.set_trace()
            index=[i for i,h in enumerate(h_split) if h=='Producto_ID'][0]
            [f.write(header) for f in file_obj]
        else:
            #i=overall_d.grp[overall_d['prod']==int(line.strip().split(',')[index])].values
            i=overall_d[int(line.strip().split(',')[index])]
            #pdb.set_trace()
            file_obj[i].write(line)
        rows+=1
        if rows%500000==0:
            print('Done %s' % (str(rows)))

for f in file_obj:
    f.close()

    
    
    


summary=summary