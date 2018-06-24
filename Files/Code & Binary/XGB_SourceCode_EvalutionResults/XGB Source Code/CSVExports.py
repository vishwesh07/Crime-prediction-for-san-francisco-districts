
# coding: utf-8

# In[1]:


#DATA TAKEN ON 5/1


# In[2]:


import pandas as pd
import numpy as np
from sodapy import Socrata


# In[3]:


client = Socrata("data.sfgov.org", None)
# results = client.get("cuks-n6tp", limit = 2191368)
data = client.get("cuks-n6tp", limit = 3000000)


# In[4]:


data_df = pd.DataFrame.from_records(data)#here


# In[5]:


print(data_df.shape)


# In[6]:


data_df.columns


# In[8]:


data_df.to_csv('CITY_data.csv', index=False, header=True, encoding='utf-8')


# In[9]:


import pickle


# In[9]:


print(data_df.columns.values)


# In[10]:


districts = data_df['pddistrict'].unique().tolist()
print(districts)


# In[11]:


mask = data_df['pddistrict'] == 'BAYVIEW'


# In[12]:


df_BAYVIEW = data_df[mask]


# In[13]:


print(df_BAYVIEW.shape)


# In[14]:


df_BAYVIEW.to_csv('BAYVIEW_data.csv', index=False, header=True, encoding='utf-8')


# In[15]:


#check if it was exported properly by importing and checking shape/columns for only bayview
df_bayview_check = pd.read_csv('BAYVIEW_data.csv')


# In[16]:


print(df_bayview_check.shape)


# In[17]:


print(df_BAYVIEW.columns.values)


# In[18]:


print(df_bayview_check.columns.values)


# In[19]:


#reuse this for all csvs, only doing bayview for now...
mask = data_df['pddistrict'] == 'RICHMOND'
df_RICHMOND = data_df[mask]
df_RICHMOND.to_csv('RICHMOND_data.csv', index=False, header=True, encoding='utf-8')


# In[20]:


print(df_RICHMOND.shape)


# In[21]:


mask = data_df['pddistrict'] == 'MISSION'
df_MISSION = data_df[mask]
df_MISSION.to_csv('MISSION_data.csv', index=False, header=True, encoding='utf-8')


# In[22]:


#reuse this for all csvs, only doing bayview for now...
mask = data_df['pddistrict'] == 'CENTRAL'
df_CENTRAL = data_df[mask]
df_CENTRAL.to_csv('CENTRAL_data.csv', index=False, header=True, encoding='utf-8')


# In[23]:


#reuse this for all csvs, only doing bayview for now...
mask = data_df['pddistrict'] == 'TARAVAL'
df_TARAVAL = data_df[mask]
df_TARAVAL.to_csv('TARAVAL_data.csv', index=False, header=True, encoding='utf-8')


# In[24]:


#reuse this for all csvs, only doing bayview for now...
mask = data_df['pddistrict'] == 'NORTHERN'
df_NORTHERN = data_df[mask]
df_NORTHERN.to_csv('NORTHERN_data.csv', index=False, header=True, encoding='utf-8')


# In[25]:


#reuse this for all csvs, only doing bayview for now...
mask = data_df['pddistrict'] == 'SOUTHERN'
df_SOUTHERN = data_df[mask]
df_SOUTHERN.to_csv('SOUTHERN_data.csv', index=False, header=True, encoding='utf-8')


# In[26]:


#reuse this for all csvs, only doing bayview for now...
mask = data_df['pddistrict'] == 'PARK'
df_PARK = data_df[mask]
df_PARK.to_csv('PARK_data.csv', index=False, header=True, encoding='utf-8')


# In[27]:


#reuse this for all csvs, only doing bayview for now...
mask = data_df['pddistrict'] == 'INGLESIDE'
df_INGLESIDE = data_df[mask]
df_INGLESIDE.to_csv('INGLESIDE_data.csv', index=False, header=True, encoding='utf-8')


# In[28]:


#reuse this for all csvs, only doing bayview for now...
mask = data_df['pddistrict'] == 'TENDERLOIN'
df_TENDERLOIN = data_df[mask]
df_TENDERLOIN.to_csv('TENDERLOIN_data.csv', index=False, header=True, encoding='utf-8')


# In[ ]:


df_district = pd.read_csv('BAYVIEW_data.csv') #change this city for csv for whatever district being done
df_district = df_district.drop(columns=['pddistrict', 'incidntnum', 'pdid', 'location', 'descript'])
df_y = df_bayview['category']
df_x = df_district.drop(columns=['category'])
labelencoder = LabelEncoder()
labelencoder = labelencoder.fit(df_y)
labelencoded_y = labelencoder.transform(df_y)
df_x['day'] = df_x.date.apply(lambda x: convert_date_to_day(x))
df_x['month'] = df_x.date.apply(lambda x: convert_date_to_month(x))
df_x['year'] = df_x.date.apply(lambda x: convert_date_to_year(x))
df_x['hour'] = df_x.time.apply(lambda x: convert_time_to_hour(x))
df_x = df_x.drop(columns=['date', 'time'])
df_x['day'] = (df_x['day']).astype(int)
df_x['month'] = (df_x['month']).astype(int)
df_x['year'] = (df_x['year']).astype(int)
df_x['hour'] = (df_x['hour']).astype(int)
label_encoder_addr = LabelEncoder()
addr_feature = label_encoder_addr.fit_transform(df_x_int.address.iloc[:].values)
addr_feature = addr_feature.reshape(df_x_int.shape[0], 1)
onehot_encoder_addr = OneHotEncoder(sparse = False)
addr_feature = onehot_encoder_addr.fit_transform(addr_feature)
label_encoder_DoW = LabelEncoder()
DoW_feature = label_encoder_DoW.fit_transform(df_x_int.dayofweek.iloc[:].values)
DoW_feature = DoW_feature.reshape(df_x_int.shape[0], 1)
onehot_encoder_DoW = OneHotEncoder(sparse = False)
DoW_feature = onehot_encoder_DoW.fit_transform(DoW_feature)
label_encoder_res = LabelEncoder()
res_feature = label_encoder_res.fit_transform(df_x_int.resolution.iloc[:].values)
res_feature = res_feature.reshape(df_x_int.shape[0], 1)
onehot_encoder_res = OneHotEncoder(sparse = False)
res_feature = onehot_encoder_res.fit_transform(res_feature)

day = df_x.day.values
month = df_x.month.values
year = df_x.year.values
hour = df_x.hour.values
x = df_x.x.values
y = df_x.y.values

columns = []
columns.append(addr_feature)
columns.append(DoW_feature)
columns.append(res_feature)
columns.append(x)
columns.append(y)
columns.append(day)
columns.append(month)
columns.append(year)
columns.append(hour)
encoded_feats = column_stack(columns)
sparse_features = sparse.csr_matrix(encoded_feats)

X_train, X_test, y_train, y_test = train_test_split(sparse_features, labelencoded_y, test_size=0.20, random_state=random_seed)

model = XGBClassifier(nthread = n_threads) #or -1
kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_seed)
param_grid = {'n_estimators': stats.randint(100,500), #random int btwn 100 and 500
              'learning_rate': stats.uniform(0.01, 0.08), #.01 + loc, range of .01+/-.08
              'max_depth': [2, 4, 6, 8], #tree depths to check
              'colsample_bytree': stats.uniform(0.5, 0.4) #btwn .1 and 1.0    
}
rand_search = RandomizedSearchCV(model, param_distributions = param_grid, scoring = 'f1_micro', n_iter = 5, n_jobs=-1, verbose = 10, cv=kfold)
rand_result = rand_search.fit(X_train, y_train)
print("Best: %f using %s" % (rand_result.best_score_, rand_result.best_params_))
best_XGB_parameters = rand_result.best_estimator_
#INSERT CITY NAME FOR .DAT FILE
pickle.dump(best_XGB_LE_estimator, open("xgb_CITYHERE.pickle.dat, "wb""))
#test on test set
best_XGB_parameters.fit(X_train, y_train)
#CSV append best score after test set


