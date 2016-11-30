
# coding: utf-8

# In[102]:

import hashlib, csv, math, os                             
import pandas as pd
import numpy as np
import sys
import sys
NR_BINS = 1000000                                      
def hashstr(input):
    return str(int(hashlib.md5(input.encode('utf8')).hexdigest(), 16)%(NR_BINS-1)+1)

# file_name = 'sample_dataset/sample_train.csv'
file_name = sys.argv[1]
# output_name = 'csv2ffm_2.ffm'
output_name = sys.argv[2]
train_dat = pd.read_csv(file_name)
merge_dat = train_dat
targets = merge_dat.clicked.values
train_len = train_dat.shape[0]


# In[96]:

from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()


# In[97]:

col_to_encoder = ['uuid','publisher_id']
encoder_train = train_dat.copy(deep=True)
for col in col_to_encoder:
    encoder_train[col] = encoder.fit_transform(train_dat[col_one])


# In[5]:

# # merge_dat.head()
# # Mapping column: Feature - Value
# map_col = lambda dat,col: col+"-"+dat.map(str)
# gen_hash_item = lambda field, feat: '{0}:{1}:1'.format(field,hashstr(feat))
# def gen_hash_row(feats,label):
#     result = []
#     for idx, item in enumerate(feats):
#         val = item.split('-')[-1]
#         if val != 'nan':
#             print item
#             result.append(gen_hash_item(idx,item))
#     lbl = 1
#     if label == 0:
#         lbl = -1
#     return str(lbl) + ' ' + ' '.join(result)+'\n'
# merge_dat_val = merge_dat.drop(['display_id','clicked'],axis=1)
# cols = merge_dat_val.columns
# features = []
# for col in merge_dat_val.columns:
#     features.append(map_col(merge_dat_val[col],col))
# features = np.array(features).T
# rows = []
# with open(output_name,'w') as f_tr:
#     i = 0;
#     for item,label in zip(features,targets):
#         if(i%200000==0):
#             print i
#         row = gen_hash_row(item,label)
# #         rows.append(row)
#         f_tr.write(row)
#         i+=1


# # Create Conjunction between two features and calculate Mutual information

# In[47]:

cols_not_conjunction = ['display_id','campaign_id','advertiser_id','clicked']
def str_to_int(a):
    result = [ord(charc) for charc in a]
    return sum(result)


# In[88]:

# for col in train_dat.columns:
#     for next_col in train.columns
cols = train_dat.columns.difference(cols_not_conjunction)
total_conjunction = {}
clicked = train_dat.clicked
for col_one in cols:
    for col_two in cols:
        if col_one!=col_two and str_to_int(col_one) < str_to_int(col_two):
            key = col_one+'|'+col_two
            total_conjunction[key] = cmutinf(8,encoder_train[col_one],encoder_train[col_two],clicked, method='grassberger')
print len(total_conjunction)
print len(cols)*(len(cols)-1)/2


# In[89]:

sorted_conj = np.array(sorted(total_conjunction.items(),key=lambda d: d[1],reverse=True)[0:4])


# In[90]:

top_cols_conj = sorted_conj[:,0]


# In[100]:

for top_cols in top_cols_conj:
    col_one,col_two = top_cols.split('|')
    train_dat[top_cols] = train_dat[col_one].map(str)+'_'+train_dat[col_two].map(str)


# In[101]:

train_dat.to_csv(output_name,index=None)

