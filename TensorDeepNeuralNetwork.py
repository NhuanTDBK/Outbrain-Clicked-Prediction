from TensorLoading import *
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
