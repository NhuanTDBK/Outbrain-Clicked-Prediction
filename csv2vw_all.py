
# coding: utf-8

# In[3]:

from datetime import datetime
from csv import DictReader, reader
import pandas as pd
import numpy as np
import sys
import csv
from datetime import datetime
csv.field_size_limit(sys.maxsize)

def epoch2datetime(x):
    epoch = (x + 1465876799998) / 1000.
    tstp = datetime.fromtimestamp(epoch)
    return [tstp.weekday(), tstp.hour]

print("Content..")
with open("data/promoted_content.csv") as infile:
    prcont = reader(infile)
    #prcont_header = (prcont.next())[1:]
    prcont_header = next(prcont)[1:]
    prcont_dict = {}
    for ind,row in enumerate(prcont):
        prcont_dict[int(row[0])] = row[1:]
    print(len(prcont_dict))
del prcont

# is_click = {}
# for row in events_dat.itertuples():
#     is_click[row.display_id] = {row.ad_id:1}

print("Leakage file..")
leak_uuid_dict= {}
# In[21]:

#is_click = {}
#csv_reader = DictReader(open('sample_dataset/events_user_features.csv'))
#for row in csv_reader:
#    is_click[row['display_id']] = {row['ad_id']:1}

# In[23]:

#is_click[row['display_id']]


# In[12]:

def convert_event(row):
    tlist = row[2:4] + row[5:6]
    time_stamp = pd.Timestamp(row[4])
    loc = row[6].split('>')
    user_feature = row[7:-1]
    if len(loc) == 3:
        tlist.extend(loc[:])
    elif len(loc) == 2:
        tlist.extend( loc[:]+[''])
    elif len(loc) == 1:
        tlist.extend( loc[:]+['',''])
    else:
        tlist.append(['','',''])
#     tlist.extend(epoch2datetime(time_stamp))
    tlist.extend([time_stamp.weekday(),time_stamp.hour])
    tlist.extend(user_feature)
    return tlist


# In[15]:

loc_csv='sample_dataset/events_user_features.csv'
loc_output='demo.w'
train=True
with open(loc_output,"wb") as outfile:
    start = datetime.now()
    r = reader(open(loc_csv))
    r.next()
    for t, row in enumerate(r):
        disp_id = row[0]
        ad_id = row[1]
        # if t >= 1:break
        #print t, row
        ids_features = "|a ad_%s |b disp_%s"% (ad_id, disp_id)
        ### Promoted content
        row_content = prcont_dict.get(ad_id, [])
        # build promoted vars
        # headers: x=document_id, y=campaign_id, z=advertiser_id
        promoted_namespaces = ['x', 'y', 'z']
        promoted_features = ""
        ad_doc_id = -1
        for i,v in enumerate(row_content):
            #print i, v
            if i == 0:
                ad_doc_id = int(v)
            promoted_features += " |%s %s_%s" % (promoted_namespaces[i], promoted_namespaces[i], v)
        #print promoted_features  
        
        
        # Events row
#         row_events = event_dict.get(disp_id, [])
#         #print row_events
#         # Create cat vars for events
#         uuid, display_id, platform,geo_location,loc_state,loc_dma,weekday,hour
        events_namespaces = ['u', 'd', 'p', 'c', 's', 'l', 'w', 'h','uipd','udpd','uad','da']
#         disp_doc_id = -1
        event_features = ""
#         if len(row_events) == 0:
#             for n in events_namespaces:
#                 event_features += " |%s na" % (n)
        row_events = convert_event(row)
        for i,v in enumerate(row_events):
            if len(row_events) == 0:
                print 'null events'
            if i == 0:
                uuid_val = v
            if i == 1:
                disp_doc_id = int(v)
            event_features += " |%s %s_%s" % (events_namespaces[i], events_namespaces[i], v)
        #print categorical_features

        features = ids_features + event_features + promoted_features
        # print features
        # Creating the labels
        if train:
            if row[-1] == 1:
                label = 1
            else:
                label = -1
            label = 1
            outfile.write( "%s '%s %s\n" % (label, t+1, features ) )
        else:
            outfile.write( "1 '%s %s\n" % (t+1, features) )
        #Reporting progress
        if t % 100000 == 0:
            print("%s\t%s"%(t, str(datetime.now() - start)))


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:



