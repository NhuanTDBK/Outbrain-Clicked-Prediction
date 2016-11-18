import hashlib, csv, math, os, subprocess                             
import pandas as pd
import numpy as np
import sys

NR_BINS = 1000000                                      
def hashstr(input):
    return str(int(hashlib.md5(input.encode('utf8')).hexdigest(), 16)%(NR_BINS-1)+1)

file_name = sys.argv[1]
output_name = sys.argv[2]
train_dat = pd.read_csv(file_name)
merge_dat = train_dat
targets = merge_dat.clicked.values
train_len = train_dat.shape[0]
# merge_dat.head()
# Mapping column: Feature - Value
map_col = lambda dat,col: col+"-"+dat.map(str)
gen_hash_item = lambda field, feat: '{0}:{1}:1'.format(field,hashstr(feat))
def gen_hash_row(feats,label):
    result = []
    for idx, item in enumerate(feats):
        result.append(gen_hash_item(cols[idx],item))
    return str(label) + ' ' + ' '.join(result)+'\n'

merge_dat_val = merge_dat.drop(['display_id','clicked'],axis=1)
cols = merge_dat_val.columns
features = []
for col in merge_dat_val.columns:
    features.append(map_col(merge_dat_val[col],col))
features = np.array(features).T
with open(output_name,'w') as f_tr:
    i = 0;
    for item,label in zip(features,targets):
        if(i%200000==0):
            print i
        row = gen_hash_row(item,label)+"\n"
        f_tr.write(row)
        i+=1

