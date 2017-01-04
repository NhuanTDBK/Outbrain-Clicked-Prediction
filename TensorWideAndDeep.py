from TensorLoading import *
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


