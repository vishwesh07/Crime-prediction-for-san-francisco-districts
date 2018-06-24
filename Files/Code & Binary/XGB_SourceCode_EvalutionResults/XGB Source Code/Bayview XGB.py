
# coding: utf-8

# In[70]:


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
#from scipy.stats import uniform
#from scipy.stats import randint
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


#tune per how many available per your system... 16 is max on this EC2 instance, while -1 is technically
#supposed to choose the max amount of cores it does not max work always.. see documentation link
n_threads = 36
model_scores = []
#track time for script to run


# In[3]:


#set random_seed for reproduction purposes..
random_seed = 10


# In[4]:


df_bayview = pd.read_csv('/home/ubuntu/CSVs/BAYVIEW_data.csv')


# In[5]:


print(df_bayview.shape)


# In[6]:


for col in df_bayview.columns:
    print(col)


# In[7]:


print(df_bayview.resolution.unique())
print(df_bayview.resolution.nunique())


# In[8]:


#Only going to drop columns with 100% no relevance or correlation and then use
#OHE, label encoding on non numeric types...  Will test with only labelencoder as well(all ordinal)
#Drop Columns pdid, pddistrict(Already split by district), incidentnum, pdid, location, description(Can't use b/c its cheating)
df_bayview = df_bayview.drop(columns=['pddistrict', 'incidntnum', 'pdid', 'location', 'descript'])


# In[9]:


print(df_bayview.shape)


# In[10]:


for col in df_bayview.columns:
    print(col)


# In[11]:


df_y = df_bayview['category']
#put category column into seperate target DF


# In[12]:


#drop category column from feature dataframe.
df_x = df_bayview.drop(columns=['category'])
for col in df_x.columns:
    print(col)


# In[13]:


#39 UNIQUE CATEGORIES BASED ON THIS NUMBER
print(df_y.nunique())


# In[14]:


#use labelencoder to convert to numeric/integer values
labelencoder = LabelEncoder()
labelencoder = labelencoder.fit(df_y)
#list(labelencoder.classes_)


# In[15]:


labelencoded_y = labelencoder.transform(df_y)


# In[16]:


labelencoded_y.shape


# In[17]:


#integer versions of targets
#print(labelencoded_y)


# In[18]:


#this lists all the integer versions of categories as strings again..
#for my own use to test labelencoder
#list(labelencoder.inverse_transform(labelencoded_y))


# In[19]:


#drop category column from feature dataframe.
###test up to here should be address, date, dow, resolution, time, x and y as columns


# In[20]:


#address, dayofweek, resolution need to be onehotencoded..
#date needs to be seperated 
#time needs to be 
#might need to chop off loc as it is very specific
#first step is to convert date to 3 seperate columns


# In[21]:


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


# In[22]:


df_x['day'] = df_x.date.apply(lambda x: convert_date_to_day(x))
df_x['month'] = df_x.date.apply(lambda x: convert_date_to_month(x))
df_x['year'] = df_x.date.apply(lambda x: convert_date_to_year(x))
df_x['hour'] = df_x.time.apply(lambda x: convert_time_to_hour(x))


# In[23]:


# print(df_x.year.unique())
# print(df_x.month.unique())
# print(df_x.day.unique())
# print(df_x.hour.unique())


# In[24]:


#drop year column from feature dataframe.
df_x = df_x.drop(columns=['date', 'time'])
for col in df_x.columns:
    print(col)


# In[25]:


#check which input features are string types/non-integers so I know which ones I need
#to one hot encode for use in XGB
print("address", type(df_x.address.iloc[1]))
print("dayofweek", type(df_x.dayofweek.iloc[1]))
print("resolution", type(df_x.resolution.iloc[1]))
print("day", type(df_x.day.iloc[1]))
print("month", type(df_x.month.iloc[1]))
print("year", type(df_x.year.iloc[1]))
print("hour", type(df_x.hour.iloc[1]))
print("x", type(df_x.x.iloc[1]))
print("y", type(df_x.y.iloc[1]))


# In[26]:


#need to OHE address, dayofweek, resolution
#convert time, day, month, year to ints? or OHE?


# In[27]:


#df_x will be with used for all labelencoded/none OHE'd
#df_x_int will be with day/month/year/hour converted to integers,
#logically since day/month/year/hour are ordinal they shouldn't have to be one
#hot encoded as they should share a relationship(e.g. hour 23 and 00) but 
#the relationship may be hard for XGB model to understand... so test w/ both
df_x_int = df_x


# In[28]:


df_x['day'] = (df_x['day']).astype(int)
df_x['month'] = (df_x['month']).astype(int)
df_x['year'] = (df_x['year']).astype(int)
df_x['hour'] = (df_x['hour']).astype(int)


# In[29]:


df_x_int['day'] = (df_x_int['day']).astype(int)
df_x_int['month'] = (df_x_int['month']).astype(int)
df_x_int['year'] = (df_x_int['year']).astype(int)
df_x_int['hour'] = (df_x_int['hour']).astype(int)


# In[30]:


#df_x_int values before OHE
print("address", type(df_x_int.address.iloc[1]))
print("dayofweek", type(df_x_int.dayofweek.iloc[1]))
print("resolution", type(df_x_int.resolution.iloc[1]))
print("day", type(df_x_int.day.iloc[1]))
print("month", type(df_x_int.month.iloc[1]))
print("year", type(df_x_int.year.iloc[1]))
print("hour", type(df_x_int.hour.iloc[1]))
print("x", type(df_x_int.x.iloc[1]))
print("y", type(df_x_int.y.iloc[1]))


# In[31]:


#for df_x_int need to convert address, dayofweek, resolution to OHE
#for df_x need to convert address, dayofweek, resolution, day, month, year, hour to OHE
#list(df_x_int)


# In[32]:


#OHE'ing address, DoW, resolution for df_x_int first


# In[33]:


#labelencoding only for df_x so it isn't sparse, also has effect of ordinal relationships even if they don't, e.g. address
#but that is fine for us for this test.
label_encoder_addr = LabelEncoder()
addr_feature = label_encoder_addr.fit_transform(df_x.address.iloc[:].values)
addr_feature.shape
#test with this feature first to see if shape is fine.. if it is then do on rest
#-------------------------------


# In[34]:


label_encoder_DoW = LabelEncoder()
DoW_feature = label_encoder_DoW.fit_transform(df_x.dayofweek.iloc[:].values)
label_encoder_res = LabelEncoder()
res_feature = label_encoder_res.fit_transform(df_x.resolution.iloc[:].values)


# In[35]:


#need to convert pandas series' for our already numeric features into numpy arrays so we can stack 
#them onto the encoded_features so that all features are in one 2d array
day = df_x.day.values
month = df_x.month.values
year = df_x.year.values
hour = df_x.hour.values
x = df_x.x.values
y = df_x.y.values

#append all OH'd columns and the numeric columns
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
encoded_feats_ordinal = column_stack(columns)


# In[36]:


#OHE'ing address for ordinal test
label_encoder_addr = LabelEncoder()
addr_feature = label_encoder_addr.fit_transform(df_x_int.address.iloc[:].values)
#print(addr_feature)
addr_feature = addr_feature.reshape(df_x_int.shape[0], 1)
#print(addr_feature)
onehot_encoder_addr = OneHotEncoder(sparse = False)
addr_feature = onehot_encoder_addr.fit_transform(addr_feature)


# In[37]:


#OHE'ing dayofweek for ordinal test
label_encoder_DoW = LabelEncoder()
DoW_feature = label_encoder_DoW.fit_transform(df_x_int.dayofweek.iloc[:].values)
#print(DoW_feature)
DoW_feature = DoW_feature.reshape(df_x_int.shape[0], 1)
#print(DoW_feature)
onehot_encoder_DoW = OneHotEncoder(sparse = False)
DoW_feature = onehot_encoder_DoW.fit_transform(DoW_feature)


# In[38]:


#OHE'ing resolution for ordinal test
label_encoder_res = LabelEncoder()
res_feature = label_encoder_res.fit_transform(df_x_int.resolution.iloc[:].values)
#print(res_feature)
#print(res_feature.shape)
#print(df_x_int.resolution.shape)
res_feature = res_feature.reshape(df_x_int.shape[0], 1)
#print(res_feature)
onehot_encoder_res = OneHotEncoder(sparse = False)
res_feature = onehot_encoder_res.fit_transform(res_feature)


# In[39]:


#7 unique DoW's because 7 days, addr more complicated because many possible addresses
#for Resolution as shown earlier in script their are 17 possible resolutions, so this is correct
print(DoW_feature.shape)
print(addr_feature.shape)
print(res_feature.shape)


# In[40]:


#need to convert pandas series' for our already numeric features into numpy arrays so we can stack 
#them onto the encoded_features so that all features are in one 2d array
day = df_x_int.day.values
month = df_x_int.month.values
year = df_x_int.year.values
hour = df_x_int.hour.values
x = df_x_int.x.values
y = df_x_int.y.values


# In[41]:


#append all OH'd columns and the numeric columns
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


# In[42]:


#shape matches up.. 3809 from OH'd + 6 numeric = 3815 columns/features with 219336 total rows
print(encoded_feats.shape)
print(encoded_feats_ordinal.shape)


# In[43]:


sprase_features_ordinal = sparse.csr_matrix(encoded_feats_ordinal)


# In[44]:


#list(encoded_feats)


# In[45]:


#type(encoded_feats)


# In[46]:


#change array to sparse format for memory efficiency, otherwise XGB is too slow/intensive on EC2 instance
sparse_features = sparse.csr_matrix(encoded_feats)


# In[47]:


#verifying shape is still fine
#sparse_features


# In[48]:


#feature list in sparse format
#type(sparse_features)


# In[49]:


#labels in numpy array
#type(labelencoded_y)


# In[50]:


#how imbalanced the label categories are
from scipy.stats import itemfreq
print(itemfreq(labelencoded_y))


# In[51]:


#our label targets are labelencoded_y and the features are encoded_feats
#now we have to split into train/test set for XGB training and testing..
#or use k-fold CV to verify.. i am only using simple kfold here to verify that features/label works properly
#in a generic model.. not for final usage, still need to TUNE
#Use stratified kfold because there are many classes and it enforces the same distribution of classes
#in each fold as is in the whole dataset when performing CV..


# In[52]:


#train test split for one hot encoded so we can compare models on training set and tune(w/ CV) then test best model on test set to get accuracy
X_train, X_test, y_train, y_test = train_test_split(sparse_features, labelencoded_y, test_size=0.20, random_state=random_seed)


# In[53]:


#train test split for non one hot encoded so we can compare models on training set and tune(w/ CV) then test best model on test set to get accuracy
X_train_LE, X_test_LE, y_train_LE, y_test_LE = train_test_split(sprase_features_ordinal, labelencoded_y, test_size=0.20, random_state=random_seed)


# In[54]:


#XGB ON NONSPARSE/NON-OHE'D/ORDINAL DATASET
model = XGBClassifier(nthread = n_threads)
kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_seed)
#use over the max, e.g. 1.1 instead of 1.0 for colsample since last value isnt used for randomsearch
#verbose messages in ipython terminal not notebook.
param_grid = {'n_estimators': [120, 240, 368, 480], #random int btwn 100 and 500
              'learning_rate': stats.uniform(0.01, 0.08), #.01 + loc, range of .01+/-.08
              'max_depth': [2, 4, 6, 8], #tree depths to check
              'colsample_bytree': stats.uniform(0.3, 0.7) #btwn .1 and 1.0    
}
rand_search = RandomizedSearchCV(model, param_distributions = param_grid, scoring = 'f1_micro', n_iter = 3, n_jobs=-1, verbose = 10, cv=kfold)


# In[55]:


rand_result_LE = rand_search.fit(X_train_LE, y_train_LE)


# In[56]:


print("Best: %f using %s" % (rand_result_LE.best_score_, rand_result_LE.best_params_))
best_XGB_LE_score = rand_result_LE.best_score_
best_XGB_LE_estimator = rand_result_LE.best_estimator_
pickle.dump(best_XGB_LE_estimator, open("xgb_le_bayview.pickle.dat", 'wb'))
model_scores.append(('xg-non-onehot', best_XGB_LE_score))


# In[57]:


#using kfold CV with randomizedgridsearch in order to hypertune 
#parameters n_est, learningrate, max_depth and colsample_bytree
#colsample_bytree performs stochastic gradient boosting by 
#splitting by choosing a subset of the features for each model/tree
#tried using nthread in model as well as n_jobs in randsearch
model = XGBClassifier(nthread = n_threads)
kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_seed)
#use over the max, e.g. 1.1 instead of 1.0 for colsample since last value isnt used for randomsearch
#verbose messages in ipython terminal not notebook.
param_grid = {'n_estimators': [120, 240, 368, 480], #stats.randint(100,500), #random int btwn 100 and 500
              'learning_rate': stats.uniform(0.01, 0.08), #.01 + loc, range of .01+/-.08
              'max_depth': [2, 4, 6, 8], #tree depths to check
              'colsample_bytree': stats.uniform(0.3, 0.7) #btwn .1 and 1.0    
}
rand_search = RandomizedSearchCV(model, param_distributions = param_grid, scoring = 'f1_micro', n_iter = 3, n_jobs=-1, verbose = 10, cv=kfold)




# In[58]:


rand_result_XGB_only = rand_search.fit(X_train, y_train)


# In[59]:


print("Best: %f using %s" % (rand_result_XGB_only.best_score_, rand_result_XGB_only.best_params_))
best_XGB_only_score = rand_result_XGB_only.best_score_
best_XGB_only_estimator = rand_result_XGB_only.best_estimator_
pickle.dump(best_XGB_only_estimator, open("xgb_only_bayview.pickle.dat", 'wb'))
model_scores.append(('xgbALONE', best_XGB_only_score))


# In[60]:


# sole_xgb = rand_result.best_estimator_
# type(sole_xgb)


# In[61]:


# plot_importance(sole_xgb)
# pyplot.show()
#plot is messed/awkward looking because there is a huge amount of features.


# In[62]:


#trying with tSVD for speedup
from sklearn.pipeline import Pipeline
from sklearn.decomposition import TruncatedSVD
#____________________________________________________________________
pipe = Pipeline(steps = [
        ('svd', TruncatedSVD(n_iter=15, random_state=random_seed)),
        ('xgb', XGBClassifier(nthread = n_threads))
        ])

#hyperparameter options

kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_seed)
#use over the max, e.g. 1.1 instead of 1.0 for colsample since last value isnt used for randomsearch
#verbose messages in ipython terminal not notebook.
param_grid = {'xgb__n_estimators': [120, 240, 368, 480], #random int btwn 100 and 500
              'xgb__learning_rate': stats.uniform(0.01, 0.08), #.01 + loc, range of .01+/-.08
              'xgb__max_depth': [2, 4, 6, 8], #tree depths to check
              'xgb__colsample_bytree': stats.uniform(0.3, 0.7), #btwn .1 and 1.0   
              'svd__n_components': stats.randint(5,1000)
}

rand_search = RandomizedSearchCV(pipe, param_distributions = param_grid, scoring = 'f1_micro', n_iter = 3, n_jobs=-1, verbose = 10, cv=kfold)



# In[63]:


rand_result_svd = rand_search.fit(X_train, y_train)


# In[64]:


print("Best: %f using %s" % (rand_result_svd.best_score_, rand_result_svd.best_params_))
best_XGB_svd_score = rand_result_svd.best_score_
best_XGB_svd_estimator = rand_result_svd.best_estimator_
pickle.dump(best_XGB_svd_estimator, open("xgb_SVD_bayview.pickle.dat", 'wb'))
model_scores.append(('xgbSVD', best_XGB_svd_score))


# In[65]:


#compare best scores:
print "XGB Label Encoded(ordinal) score:", best_XGB_LE_score
print "XGB One hot encoded score:", best_XGB_only_score
print "SVD best score:", best_XGB_svd_score
#export model scores to a data frame
mscore_df = pd.DataFrame(model_scores, columns = ['model', 'f1_score'])


# In[76]:


sns.set_style("whitegrid")
msplt = sns.barplot(x = "model", y = "f1_score", data=mscore_df)
_ = plt.xlabel('Model')
_ = plt.ylabel('f1 score')
_ = plt.savefig('model_selection_bar')
_ = plt.show()


# In[71]:


#test on test set here.
#XGBoost non one-hot-encoded was the best model but only very slightly, XGB one-hot-encoded alone without
#feature reduction performed nearly as well and was much more efficient in time computations, thus I will use
#XGB w/o feature selection and with one-hot-encoding for my final modelling on test set as well as in other
#districts and whole city
#train the best estimator on the train set again then test on test set for results
best_XGB_only_estimator.fit(X_train, y_train)
preds = best_XGB_only_estimator.predict(X_test)
f1score = f1_score(y_test, preds, average = 'micro')


# In[72]:


f1_score = []
f1_score.append(('Bayview', f1score))
export_df = pd.DataFrame(f1_score)
export_df.to_csv("bayviewresults.dat", index = False, header = False)

