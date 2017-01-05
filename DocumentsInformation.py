
# coding: utf-8

# In[194]:

import pandas as pd
import numpy as np
import hashlib
from datetime import datetime
# In[181]:

NR_BINS = long(2**28)
def hashstr(input):
    return str(int(hashlib.md5(input.encode('utf8')).hexdigest(), 16)%(NR_BINS-1)+1)


# In[117]:

documents_topic = pd.read_csv('data/documents_topics.csv.zip',compression='zip',index_col=0)
documents_topic['confidence_level'] = documents_topic.confidence_level.map(lambda d: int(np.ceil(d*100)))
topic_namespace = ['tid','tconf']


# In[82]:

documents_categories = pd.read_csv('data/documents_categories.csv.zip',compression='zip',index_col=0)
documents_categories['confidence_level'] = documents_categories.confidence_level.map(lambda d: int(np.ceil(d*100)))
category_namespace = ['cid','cconf']


# In[119]:

documents_entities = pd.read_csv('data/documents_entities.csv.zip',compression='zip',index_col=0)
documents_entities['confidence_level'] = documents_entities.confidence_level.map(lambda d: int(np.ceil(d*100)))
entity_namespace = ['eid','econf']

# In[186]:

def _get_doc_info(document_id,dat,namespace):
    doc_id = int(document_id)
    em =""
    if doc_id in dat.index:
        total_vals = dat.loc[doc_id]
        if len(total_vals==1):
            iterator = total_vals.index
            for idxcol,col in enumerate(iterator):
#                em += " |%s0 %s0_%s"%(namespace[idxcol],namespace[idxcol],col)
		 em += format(col,namespace[idxcol],TYPE)
        else:
            iterator = total_vals.columns
            for idxcol,col in enumerate(iterator):
                for idx,row in enumerate(total_vals[col].values):
#                    em += " |%s%s %s%s_%s"%(namespace[idxcol],idx,namespace[idxcol],idx,row)
		     em += format(row,namespace[idxcol]+idx,TYPE)
    return em
def write_doc_info_type(document_id,doc_type):
    dat = []
    namespace = []
    if doc_type == 'entity':
        dat = documents_entities
        namespace = entity_namespace
    elif doc_type =='category':
        dat = documents_categories
        namespace = category_namespace
    else:
        dat = documents_topic
        namespace = topic_namespace
    return _get_doc_info(document_id,dat,namespace)

# In[187]:
#write_doc_info_type(12,'entity')
# In[190]:

#from csv import reader
#meta_reader = reader(open('data/documents_meta.csv'))
#meta_namespace = ['sid','puid','ptime']
#cols = meta_reader.next()


# In[192]:

#a = meta_reader.next()


# In[200]:

#b = pd.Timestamp(a[-1])
#print b.weekday(),b.hour

# In[ ]:




# In[ ]:




# In[ ]:



