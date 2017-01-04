import tempfile
from TensorLoading import *

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
