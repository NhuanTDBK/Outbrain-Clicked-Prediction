
# coding: utf-8

# In[66]:

import pandas as pd
import tensorflow as tf
import numpy as np
from datetime import datetime
import tempfile
import math


# In[2]:

df_train = pd.read_csv("sample_dataset/events_user_features.csv")


# In[3]:

def epoch2datetime(x):
    epoch = (x + 1465876799998) / 1000.
    tstp = datetime.fromtimestamp(epoch)
    return tstp
#     return [tstp.weekday(), tstp.hour]
# In[4]:
df_train['weekday'] = df_train.timestamp.map(lambda d: epoch2datetime(d).weekday())
# In[5]:
df_train['hour'] = df_train.timestamp.map(lambda d: epoch2datetime(d).hour)
# In[6]:

df_train_drop = df_train.drop(['timestamp'],axis=1)

# In[7]:

#df_train_drop.head()


# In[18]:

CATEGORICAL_COLUMNS = ['display_id','ad_id','document_id','uuid','geo_location','platform','weekday','hour']
CONTINUOUS_COLUMNS = ['uuid_doc_impression','total_impression','uuid_seen_ad'
                      ,'appeared']
LABEL_COLUMN = 'clicked'


# In[19]:

df_train.geo_location.fillna('US>CA>803',inplace=True)


# In[20]:

def input_fn(df):
  # Creates a dictionary mapping from each continuous feature column name (k) to
  # the values of that column stored in a constant Tensor.
  
    continuous_cols = {k: tf.constant(df[k].values)
                     for k in CONTINUOUS_COLUMNS}
  # Creates a dictionary mapping from each categorical feature column name (k)
  # to the values of that column stored in a tf.SparseTensor.
    categorical_cols = {k: tf.SparseTensor(
      indices=[[i, 0] for i in range(df[k].size)],
      values=df[k].values,
      shape=[df[k].size, 1])
                      for k in CATEGORICAL_COLUMNS}
  # Merges the two dictionaries into one.
    feature_cols = dict(continuous_cols.items() + categorical_cols.items())
  # Converts the label column into a constant Tensor.
    label = tf.constant(df[LABEL_COLUMN].values)
  # Returns the feature columns and the label.
    return feature_cols, label


# In[38]:

display_id = tf.contrib.layers.sparse_column_with_integerized_feature("display_id",bucket_size=df_train.display_id.max())
uuid = tf.contrib.layers.sparse_column_with_hash_bucket("uuid", hash_bucket_size=1000000)
ad_id = tf.contrib.layers.sparse_column_with_integerized_feature("ad_id", bucket_size=df_train.ad_id.max())
document_id = tf.contrib.layers.sparse_column_with_integerized_feature("document_id",bucket_size=df_train.document_id.max())
geo_location = tf.contrib.layers.sparse_column_with_hash_bucket('geo_location',hash_bucket_size=100)
platform = tf.contrib.layers.sparse_column_with_integerized_feature('platform',bucket_size=3)
# weekday = tf.contrib.layers.real_valued_column('weekday')
weekday = tf.contrib.layers.sparse_column_with_integerized_feature('weekday',bucket_size=7)
hour = tf.contrib.layers.sparse_column_with_integerized_feature('hour',bucket_size=24)
# hour = tf.contrib.layers.real_valued_column('hour')


# In[152]:

uuid_impress_per_day = tf.contrib.layers.real_valued_column('uuid_doc_impression')
uuid_doc_per_day = tf.contrib.layers.real_valued_column('total_impression')
uuid_doc_ad_per_day = tf.contrib.layers.real_valued_column('uuid_seen_ad')
day_appeared = tf.contrib.layers.real_valued_column('appeared')


# In[90]:

# uuid_impress_per_day_square = tf.contrib.layers.real_valued_column('uuid_doc_impression',normalizer=lambda d: d**2)
# uuid_doc_per_day_square = tf.contrib.layers.real_valued_column('total_impression',normalizer=lambda d: d**2)
# uuid_doc_ad_square = tf.contrib.layers.real_valued_column('uuid_seen_ad',normalizer=lambda d: d**2)
# day_appeared_square = tf.contrib.layers.real_valued_column('appeared',normalizer=lambda d: d**2)


# In[147]:

# uuid_impress_per_day_sqrt = tf.contrib.layers.real_valued_column('uuid_doc_impression',normalizer=lambda d: math.sqrt(d))
# uuid_doc_per_day_sqrt = tf.contrib.layers.real_valued_column('total_impression',normalizer=lambda d: math.sqrt(d))
# uuid_doc_ad_sqrt = tf.contrib.layers.real_valued_column('uuid_seen_ad',normalizer=lambda d: math.sqrt(d))
# day_appeared_sqrt = tf.contrib.layers.real_valued_column('appeared',normalizer=lambda d: math.sqrt(d))


# In[149]:

# uuid_ad = tf.contrib.layers.crossed_column(
#   [uuid,ad_id], hash_bucket_size=int(1e9))
# doc_ad = tf.contrib.layers.crossed_column(
#   [document_id,ad_id], hash_bucket_size=int(1e9))
uuid_geo = tf.contrib.layers.crossed_column(combiner='sqrtn',
    columns= [uuid,geo_location], hash_bucket_size=int(1e9))


# In[145]:
train_len = int(len(df_train_drop)*0.9)
train = df_train_drop[:train_len]
test = df_train_drop[train_len:]


# In[146]:

def train_input_fn():
    return input_fn(train)
def eval_input_fn():
    return input_fn(test)


# # Wide model

# In[150]:

import tempfile
model_dir = tempfile.mkdtemp()
linear_columns = [weekday,
hour,platform,uuid_impress_per_day,uuid_doc_per_day,uuid_doc_ad_per_day,day_appeared, display_id,ad_id,
document_id,uuid,geo_location,uuid_geo
]
m = tf.contrib.learn.LinearClassifier(feature_columns=linear_columns,
optimizer=tf.train.FtrlOptimizer(learning_rate=0.01,l2_regularization_strength=1e-7),
model_dir=model_dir)


# In[151]:

m.fit(input_fn=train_input_fn, steps=300)


# In[ ]:
print "Linear Regression results"
results = m.evaluate(input_fn=eval_input_fn,steps=1)

# # Deep model

# In[122]:
print "Deep Neural Network"
document_dims = int(math.log(len(df_train.document_id.unique()),2))
ad_dims = int(math.log(len(df_train.ad_id.unique()),2))
uuid_dims = int(math.log(len(df_train.uuid.unique()),2))

deep_columns = [
  tf.contrib.layers.embedding_column(document_id, dimension=document_dims,combiner='sqrtn'),
  tf.contrib.layers.embedding_column(ad_id, dimension=ad_dims,combiner='sqrtn'),
  tf.contrib.layers.embedding_column(uuid, dimension=uuid_dims,combiner='sqrtn'),
  tf.contrib.layers.embedding_column(geo_location, dimension=4,combiner='sqrtn'),
  uuid_impress_per_day,uuid_doc_per_day,uuid_doc_ad_per_day,day_appeared]

# In[123]:

import tempfile
model_dir = tempfile.mkdtemp()
m = tf.contrib.learn.DNNClassifier(feature_columns=deep_columns,hidden_units=[64],
                                   optimizer=tf.train.AdamOptimizer(),
model_dir=model_dir)


# In[124]:

estimator = m.fit(input_fn=train_input_fn, steps=200)


# In[ ]:

results = m.evaluate(input_fn=eval_input_fn,steps=1)


# # Wide and Deep model

# In[264]:

# In[127]:

wide_columns = [weekday,hour,platform,
                uuid_doc_ad_per_day,day_appeared,uuid_impress_per_day,uuid_doc_per_day,geo_location]
deep_columns = [
  tf.contrib.layers.embedding_column(document_id, dimension=document_dims,combiner='sqrtn'),
  tf.contrib.layers.embedding_column(ad_id, dimension=ad_dims,combiner='sqrtn'),
  tf.contrib.layers.embedding_column(uuid, dimension=uuid_dims,combiner='sqrtn'),
  uuid_impress_per_day,uuid_doc_per_day, uuid_doc_ad_per_day,day_appeared
]
print "Deep Wide Neural Network"
# In[131]:

model_dir = tempfile.mkdtemp()
m = tf.contrib.learn.DNNLinearCombinedClassifier(dnn_optimizer=tf.train.AdamOptimizer(),
    model_dir=model_dir,
    linear_feature_columns=wide_columns,
    dnn_feature_columns=deep_columns,
    dnn_hidden_units=[64])


# In[132]:

estimator = m.fit(input_fn=train_input_fn, steps=300)


# In[133]:

results = m.evaluate(input_fn=eval_input_fn,steps=1)


# In[ ]:



