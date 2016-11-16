
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit


# In[2]:

dat = pd.read_csv('sample_dataset/sample_train_dat.csv')


# In[3]:

dat.head()


# In[5]:

# tr = dat[dat.day<26]
# tv = dat[dat.day>=26]


# In[32]:

tr = dat


# In[53]:

group = tr.display_id
test_size_20 = int(len(group)*1/100)


# In[54]:

test_idx_split_from_training = np.random.randint(low=group.min(),high=group.max(),size=(test_size_20,))


# In[55]:

test_dat_split = tr[tr.display_id.isin(test_idx_split_from_training)]


# In[59]:

training_split = tr[~tr.display_id.isin(test_idx_split_from_training)]


# In[66]:

training_split.to_csv("train_dat_26.csv",index=None)


# In[ ]:

# tv.append(test_dat_split).to_csv("val_dat_26.csv",index=None)


# In[ ]:



