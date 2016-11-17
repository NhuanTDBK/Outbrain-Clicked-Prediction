
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
import sys
argv = sys.argv
# In[43]:
true_loss = argv[1] 
dataset = pd.read_csv(true_loss)

a = np.random.normal(loc=2.0,size=(dataset.shape[0]))
if (len(argv)==3):
	pred_loss = argv[2]
	a =  pd.read_csv(pred_loss)['ctr']
if 'position' not in dataset.columns:
    position = np.zeros(dataset.shape[0],dtype=np.uint)
    s = dataset[['uuid','display_id','clicked']].values
    current_idx = 0
    skip_idx = current_idx
    i = 0
    while current_idx < len(position)-1:
        if s[current_idx][0] == s[current_idx+1][0]:
            position[current_idx]=i
            i+=1
        else:
            i=0
        current_idx+=1
        position[current_idx] = i
    dataset['position'] = position

# In[48]:

dataset['ctr'] = a
y_true = dataset[dataset.clicked==1][['display_id','ad_id']]


# In[49]:

loc_dat = dataset.sort_values(['display_id','ctr'],ascending=[True,True])


# In[54]:

y_pred_loc = loc_dat[loc_dat.isin(y_true)['ad_id']]['position']+1


# In[57]:

mape = sum([1.0/i for i in y_pred_loc])/(len(y_pred_loc))
print "MAP@12 = %s"%mape


# In[ ]:



