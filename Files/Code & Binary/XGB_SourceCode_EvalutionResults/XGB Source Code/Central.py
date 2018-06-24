
# coding: utf-8

# In[1]:


#libraries
import pandas as pd
import numpy as np
from numpy import column_stack
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from scipy import stats
from time import time
import matplotlib.pyplot as plt
from xgboost import plot_importance
import re
from scipy import sparse
from datetime import datetime
import pickle
import seaborn as sns


# In[2]:


n_threads = 36
#set random_seed for reproduction purposes..
random_seed = 10


# In[3]:


#convert date to seperate values for day, month, hour
def convert_date_to_day(dt):

   result = re.findall(r'\d{4}-(\d{2})-(\d{2})T00:00:00.000',dt)

   return result[0][1]

   

def convert_date_to_month(dt):

   result = re.findall(r'\d{4}-(\d{2})-(\d{2})T00:00:00.000',dt)

   return result[0][0]


def convert_date_to_year(dt):
    
    result = re.findall(r'(\d{4})-(\d{2})-(\d{2})T00:00:00.000',dt)

    return result[0][0]

def convert_time_to_hour(tm):

   result = re.findall(r'(\d{2}):\d{2}',tm)

   return result[0]


# In[4]:


df_district = pd.read_csv('/home/ubuntu/CSVs/CENTRAL_data.csv') #change this city for csv for whatever district being done
df_district = df_district.drop(columns=['pddistrict', 'incidntnum', 'pdid', 'location', 'descript'])
df_y = df_district['category']
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
addr_feature = label_encoder_addr.fit_transform(df_x.address.iloc[:].values)
addr_feature = addr_feature.reshape(df_x.shape[0], 1)
onehot_encoder_addr = OneHotEncoder(sparse = False)
addr_feature = onehot_encoder_addr.fit_transform(addr_feature)
label_encoder_DoW = LabelEncoder()
DoW_feature = label_encoder_DoW.fit_transform(df_x.dayofweek.iloc[:].values)
DoW_feature = DoW_feature.reshape(df_x.shape[0], 1)
onehot_encoder_DoW = OneHotEncoder(sparse = False)
DoW_feature = onehot_encoder_DoW.fit_transform(DoW_feature)
label_encoder_res = LabelEncoder()
res_feature = label_encoder_res.fit_transform(df_x.resolution.iloc[:].values)
res_feature = res_feature.reshape(df_x.shape[0], 1)
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


# In[5]:


X_train, X_test, y_train, y_test = train_test_split(sparse_features, labelencoded_y, test_size=0.20, random_state=random_seed)

model = XGBClassifier(nthread = n_threads) #or -1
kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_seed)
param_grid = {'n_estimators': [120, 240, 360, 480], #random int btwn 100 and 500 - removed
              'learning_rate': stats.uniform(0.01, 0.08), #.01 + loc, range of .01+/-.08
              'max_depth': [2, 4, 6, 8], #tree depths to check
              'colsample_bytree': stats.uniform(0.3, 0.7) #btwn .1 and 1.0    
}
rand_search = RandomizedSearchCV(model, param_distributions = param_grid, scoring = 'f1_micro', n_iter = 3, n_jobs=-1, verbose = 10, cv=kfold)
rand_result = rand_search.fit(X_train, y_train)
print("Best: %f using %s" % (rand_result.best_score_, rand_result.best_params_))
best_XGB_parameters = rand_result.best_estimator_
#INSERT CITY NAME FOR .DAT FILE
pickle.dump(best_XGB_parameters, open("xgb_CENTRAL.pickle.dat", 'wb')) #change pickle


# In[6]:


#test on test set
best_XGB_parameters.fit(X_train, y_train)
preds = best_XGB_parameters.predict(X_test)
f1score = f1_score(y_test, preds, average = 'micro')
#CSV append best score after test set
f1_score = []
f1_score.append(('Central', f1score))
export_df = pd.DataFrame(f1_score)
#change csv name
export_df.to_csv("Central_results.dat", index = False, header = False)

