
# coding: utf-8

# In[212]:

import hashlib, csv, math, os, subprocess                             
import pandas as pd
import numpy as np
NR_BINS = 1000000                                      


def hashstr(input):
    return str(int(hashlib.md5(input.encode('utf8')).hexdigest(), 16)%(NR_BINS-1)+1)


train_dat = pd.read_csv('sample_dataset/sample_train.csv',usecols=[6,9,10,12,13,14,15])
val_dat = pd.read_csv('sample_dataset/sample_validation.csv',usecols=[6,9,10,12,13,14,15])

merge_dat = train_dat.append(val_dat)
targets = merge_dat.clicked.values

train_len = train_dat.shape[0]

# merge_dat.head()

map_col = lambda dat,col: col+"-"+dat.map(str)

merge_dat_val = merge_dat.drop(['clicked'],axis=1)

features = []
for col in merge_dat_val.columns:
    features.append(map_col(merge_dat_val[col],col))
features = np.array(features).T

gen_hash_item = lambda field, feat: '{0}:{1}:1'.format(field,hashstr(feat))

cols = merge_dat.columns

def gen_hash_row(feats,label):
    result = []
    for idx, item in enumerate(feats):
        result.append(gen_hash_item(cols[idx],item))
    return str(label) + ' ' + ' '.join(result)+'\n'

with open('train.ffm','w') as f_tr, open('valid.ffm','w') as f_vl:
    i = 0;
    for item,label in zip(features,targets):
        if(i%10000==0):
            print i
        row = gen_hash_row(item,label)
        if (i < train_len):
            f_tr.write(row)
        else:
            f_vl.write(row)
        i+=1


# In[79]:

valid_target = targets[train_len:]


# In[81]:

import json


# In[200]:

from numpy import loadtxt
lines = loadtxt("ffm_test/result_2.csv", comments="#", delimiter="\n", unpack=False)


# In[201]:

sigmoid = lambda d: 1.0 / (1+ np.exp(-d))


# In[202]:

logloss = sigmoid(lines)


# In[203]:

val_logloss = pd.DataFrame(val_dat[['uuid','clicked']],columns=['uuid','clicked'])


# In[204]:

val_logloss['log_loss'] = logloss


# In[205]:

val_logloss.sort_values(by=['uuid','log_loss'],ascending=[True,True],inplace=True)


# In[206]:

from sklearn.linear_model import LogisticRegression


# In[207]:

linear_model = LogisticRegression(C=0.001,max_iter=20)


# In[208]:

linear_model.fit(logloss.reshape(-1,1),valid_target)


# In[209]:


a = linear_model.predict(logloss.reshape(-1,1))


# In[210]:

from sklearn.metrics import accuracy_score


# In[211]:

print accuracy_score(a,valid_target)


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:



