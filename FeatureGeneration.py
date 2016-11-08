
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import StratifiedKFold, train_test_split, cross_val_score
from sklearn.metrics import roc_auc_score, confusion_matrix, f1_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.datasets import dump_svmlight_file


# # Preparing LIBSVM format 

# In[6]:
feature_train_path = "data/ads_time_platform_geo_week_impress.csv"
label_path = "data/clicks_train.csv"
ctr_path = "data/CTR_per_ad.csv"
features = pd.read_csv(feature_train_path)
labels = pd.read_csv(label_path)
ctr_per_ad = pd.read_csv(ctr_path)
clicks_train = labels.shape[0]
# In[7]:

dataset = pd.merge(ctr_per_ad,labels,on=['display_id','ad_id']).merge(features,
        on=['display_id','ad_id']).sort_values(by=['display_id'],ascending=True)

# In[8]:

label_encoder = LabelEncoder()
dataset.geo_location = label_encoder.fit_transform(dataset.geo_location)

# In[12]:

t = dataset.clicked
p = dataset.prob
dataset.drop(['clicked','prob','uuid'],axis=1,inplace=True)
#dataset.drop(['clicked','prob'],axis=1,inplace=True)
dataset['prob'] = p
dataset['clicked'] = t
#dataset.head()
# In[17]:
onehot_encoder = OneHotEncoder(n_values=[12,7,3,label_encoder.classes_.shape[0],2],
                               categorical_features=[0,1,2,3,4])
# In[18]:

feature_one_hot = onehot_encoder.fit_transform(dataset.values[:,2:-1])
# In[19]:
dump_svmlight_file(X=feature_one_hot,y=dataset.clicked,f='all_feature_sparse')
dump_svmlight_file(X=feature_one_hot[:clicks_train_len],y=dataset.clicked[:clicks_train_len],f='feature_sparse')
# In[9]:
dataset.display_id.to_csv('all_feature_sparse.group',index=None,header=None)
dataset.display_id.iloc[:clicks_train_len].to_csv('feature_sparse.group',index=None,header=None)
# # XGBoost pairwise
# In[20]:
import xgboost as xgb
# In[21]:

dataset = xgb.DMatrix("feature_sparse")


# In[22]:
ratio = 0.9
train_size = int(dataset.num_row()*ratio)


# In[23]:

train = dataset.slice(range(train_size))


# In[24]:

test = dataset.slice(range(train_size,int(dataset.num_row())))


# In[25]:

param = {'bst:max_depth':50, 'bst:eta':1, 'objective':'rank:pairwise','n_estimators':400}
param['nthread'] = -1
param['eval_metric'] = ['map@12','auc']
evallist  = [(test,'eval'), (train,'train')]
# In[26]:

num_round = 400
bst = xgb.train(param, train, num_round,verbose_eval=2)


# In[27]:

print bst.eval(test)


# In[28]:

a = bst.predict(test)


# In[33]:

sigmoid = lambda d: 1.0 / (1 + np.exp(-d))


# In[35]:

b = sigmoid(a)


# In[36]:

zip(b,test.get_label())

