import hashlib, csv, math, os, subprocess                             
import pandas as pd
import numpy as np
import sys

# NR_BINS = 1000000                                      
# def hashstr(input):
#     return str(int(hashlib.md5(input.encode('utf8')).hexdigest(), 16)%(NR_BINS-1)+1)

# file_name = "sample_dataset/sample_train.csv"
# output_name = "train_sample.ffm"
file_name = sys.argv[1]
output_name = sys.argv[2]
train_dat = pd.read_csv(file_name,index_col=0)
merge_dat = train_dat
targets = merge_dat.clicked.values
train_len = train_dat.shape[0]
# merge_dat.head()
# Mapping column: Feature - Value
categories_field = ['uuid','ad_id']
numerical_field = ['day_of_week','hour','position']
map_col = lambda dat,col: col+":"+dat.map(str)
gen_hash_item = lambda field, feat: '{0}:{1}'.format(field,hashstr(feat))
def gen_hash_row(feats,label):
    result = []
    categories_features = []
    numerical_features = []
    for idx, item in enumerate(feats):
#         print item.split(":")[0]
        if (item.split(":")[0] in categories_field):
            categories_features.append(feats[idx])
        else:
            numerical_features.append(feats[idx])
#         result.append(gen_hash_item(cols[idx],item))
    lbl = 1
    if label == 0:
        lbl = -1
    return "{0} |i {1} |c {2}".format(lbl, ' '.join(numerical_features),' '.join(categories_features))

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
        row = gen_hash_row(item,label) + '\n'
        f_tr.write(row)
        i+=1

