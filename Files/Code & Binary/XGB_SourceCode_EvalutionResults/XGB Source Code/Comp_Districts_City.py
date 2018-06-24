
# coding: utf-8

# In[11]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[12]:


#need to add city after finishing
df_city = pd.read_csv("cityresults.dat", header=None)
df_bayview = pd.read_csv("bayviewresults.dat", header=None)
df_ingleside = pd.read_csv("Ingleside_results.dat", header=None)
df_park = pd.read_csv("Park_results.dat", header=None)
df_taraval = pd.read_csv("Taraval_results.dat", header=None)
df_central = pd.read_csv("Central_results.dat", header=None)
df_mission = pd.read_csv("Mission_results.dat", header=None)
df_richmond = pd.read_csv("Richmond_results.dat", header=None)
df_tenderloin = pd.read_csv("Tenderloin_results.dat", header=None)
df_northern = pd.read_csv("Northern_results.dat", header=None)
df_southern = pd.read_csv("Southern_results.dat", header=None)


# In[13]:


f1_score = []
f1_score.append(df_city)
f1_score.append(df_bayview)
f1_score.append(df_ingleside)
f1_score.append(df_park)
f1_score.append(df_taraval)
f1_score.append(df_central)
f1_score.append(df_mission)
f1_score.append(df_richmond)
f1_score.append(df_tenderloin)
f1_score.append(df_northern)
f1_score.append(df_southern)


# In[14]:


df_dist = pd.concat(f1_score, axis = 0, ignore_index=True)


# In[15]:


df_dist.columns = ['model', 'f1']


# In[16]:


df_dist


# In[17]:


df_dist.shape


# In[18]:


sns.set_style("whitegrid")
#might need to convert data from list to pandas DF..
#try above first
msplt = sns.barplot(x = "model", y = "f1", data=df_dist)
msplt.set_xticklabels(msplt.get_xticklabels(), rotation =40, ha="right")
_ = plt.xlabel('Model')
_ = plt.ylabel('f1 score')
_ = plt.savefig('district_comparisons')
_ = plt.show()

