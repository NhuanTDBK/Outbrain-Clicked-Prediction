
# coding: utf-8

# In[41]:

import csv
import numpy as np
from datetime import datetime
import pandas as pd
from sklearn.preprocessing import LabelEncoder


# In[91]:

encoder = LabelEncoder()


# In[92]:

dat = pd.read_csv('sample_dataset/merge_events.csv')


# In[93]:

dat.uuid = encoder.fit_transform(dat.uuid)


# In[94]:

def epoch2datetime(time):
    x = time
    if type(time) is str:
        x = long(time)
    epoch = (x + 1465876799998) / 1000.
    tstp = datetime.fromtimestamp(epoch)
    return tstp


# In[95]:

reader = csv.DictReader(open('sample_dataset/merge_events.csv','r'))


# In[96]:

dat['day'] = dat.timestamp.map(lambda d: epoch2datetime(d).day)
dat['hour'] = dat.timestamp.map(lambda d: epoch2datetime(d).hour)


# In[97]:

impression_per_day = dat.groupby(['uuid','day']).count()['ad_id']


# In[98]:

user_day_appeared = dat.groupby(['uuid']).day.nunique()


# In[99]:

impression_user_ad_doc_per_day = dat.groupby(['uuid','ad_id','document_id','day']).count()['hour']


# In[100]:

ad_impression_per_doc = dat.groupby(['document_id','ad_id','day']).count()['hour']


# In[101]:

# %%time
# # a = encoder.transform(['22de81701782e6'])[0]
# impression_user_ad_doc_per_day.ix[0][131995][1282987][14]
# # impression_user_ad_doc_per_day.ix[('22de81701782e6', 150863, 1056113, 14)]


# In[22]:

# add_keys = ['ad_impression_per_doc','impression_user_ad_doc_per_day','impression_per_day','user_day_appeared']


# In[102]:

del dat


# In[103]:

add_keys = ['ad_doc','uuid_doc_ad','impress_uuid_day','uuid_day_appeared']
fieldnames = reader.fieldnames.extend(add_keys)
csvWriter = csv.DictWriter(open('events_feature.csv','w'),fieldnames=reader.fieldnames)


# In[104]:

csvWriter.writeheader()


# In[105]:

uuids = []
for idx, a in enumerate(reader):
    document_id = np.int64(a['document_id'])
    ad_id = np.int64(int(a['ad_id']))
    day = epoch2datetime(a['timestamp']).day
    uuid = encoder.transform([a['uuid']])[0]
    a['ad_doc'] = ad_impression_per_doc.ix[document_id][ad_id][day]
    a['uuid_doc_ad'] = impression_per_day.ix[uuid][day]
    a['impress_uuid_day'] = impression_user_ad_doc_per_day.ix[uuid][ad_id][document_id][day]
    a['uuid_day_appeared'] = user_day_appeared.ix[uuid]
    uuids.append(a)
    if idx % 200000 == 1:
        print "Flushing..."
#         csvWriter.writerows(uuids)
        uuids = []

