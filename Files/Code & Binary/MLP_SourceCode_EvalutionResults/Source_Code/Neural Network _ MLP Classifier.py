
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
results_df = pd.read_csv('SFPD_dataset.csv',encoding = "ISO-8859-1")


# In[2]:


print(len(results_df))


# In[3]:


results_df.index


# In[4]:


results_df.columns


# In[5]:


results_df.values


# In[6]:


address = results_df.address.unique()
category = results_df.category.unique()
dayofweek = results_df.dayofweek.unique()
descript = results_df.descript.unique()
# location = results_df.location.unique()
pddistrict = results_df.pddistrict.unique()
date = results_df.date.unique()
time = results_df.time.unique()
X = results_df.x.unique()
Y = results_df.y.unique()
incedent_number = results_df.incidntnum.unique()
pdid = results_df.pdid.unique()
resolution = results_df.resolution.unique()


# In[7]:


print("Unique values of attributes: ")
print("Address", len(address))
print("Category", len(category))
print("Day of week", len(dayofweek))
print("Descript", len(descript))
print("Pd district", len(pddistrict))
print("Date",len(date))
print("Time", len(time))
print("X", len(X))
print("Y", len(Y))
print("Incedent Number", len(incedent_number))
print("Pd Id", len(pdid))
print("Resolution", len(resolution))


# In[8]:


results_df.info()


# In[9]:


print(len(category))


# In[10]:


df_list = set()
for v in results_df['pddistrict']:
    if isinstance(v,str):
        df_list.add(v)


# In[11]:


df_list


# In[12]:


# Save data for whole city in 'SFPD_dataset'
# results_df.to_csv('SFPD_dataset.csv',index=False,header=True, columns=['address', 'category', 'date', 'dayofweek', 'descript', 'incidntnum',
#        'location', 'pddistrict', 'pdid', 'resolution', 'time', 'x', 'y'])


# In[14]:


district_data = dict()


# In[15]:


# Save data for different district in different dataframe


# In[16]:


mask = results_df['pddistrict'] == 'BAYVIEW'
district_data['BAYVIEW'] = results_df[mask]
# df_BAYVIEW.to_csv('BAYVIEW_data', index=False, header=True, columns= ['address', 'category', 'date', 'dayofweek', 'descript', 'incidntnum',
#        'location', 'pddistrict', 'pdid', 'resolution', 'time', 'x', 'y'])


# In[17]:


mask = results_df['pddistrict'] == 'CENTRAL'
district_data['CENTRAL'] = results_df[mask]
# df_CENTRAL.to_csv('CENTRAL_data', index=False, header=True, columns= ['address', 'category', 'date', 'dayofweek', 'descript', 'incidntnum',
#        'location', 'pddistrict', 'pdid', 'resolution', 'time', 'x', 'y'])


# In[18]:


mask = results_df['pddistrict'] == 'INGLESIDE'
district_data['INGLESIDE'] = results_df[mask]
# df_INGLESIDE.to_csv('INGLESIDE_data', index=False, header=True, columns= ['address', 'category', 'date', 'dayofweek', 'descript', 'incidntnum',
#        'location', 'pddistrict', 'pdid', 'resolution', 'time', 'x', 'y'])


# In[19]:


mask = results_df['pddistrict'] == 'MISSION'
district_data['MISSION'] = results_df[mask]
# df_MISSION.to_csv('MISSION_data', index=False, header=True, columns= ['address', 'category', 'date', 'dayofweek', 'descript', 'incidntnum',
#        'location', 'pddistrict', 'pdid', 'resolution', 'time', 'x', 'y'])


# In[20]:


mask = results_df['pddistrict'] == 'NORTHERN'
district_data['NORTHERN'] = results_df[mask]
# df_NORTHERN.to_csv('NORTHERN_data', index=False, header=True, columns= ['address', 'category', 'date', 'dayofweek', 'descript', 'incidntnum',
#        'location', 'pddistrict', 'pdid', 'resolution', 'time', 'x', 'y'])


# In[21]:


mask = results_df['pddistrict'] == 'SOUTHERN'
district_data['SOUTHERN'] = results_df[mask]
# df_SOUTHERN.to_csv('SOUTHERN_data', index=False, header=True, columns= ['address', 'category', 'date', 'dayofweek', 'descript', 'incidntnum',
#        'location', 'pddistrict', 'pdid', 'resolution', 'time', 'x', 'y'])


# In[22]:


mask = results_df['pddistrict'] == 'TARAVAL'
district_data['TARAVAL'] = results_df[mask]
# df_TARAVAL.to_csv('TARAVAL_data', index=False, header=True, columns= ['address', 'category', 'date', 'dayofweek', 'descript', 'incidntnum',
#        'location', 'pddistrict', 'pdid', 'resolution', 'time', 'x', 'y'])


# In[23]:


mask = results_df['pddistrict'] == 'TENDERLOIN'
district_data['TENDERLOIN'] = results_df[mask]
# df_TENDERLOIN.to_csv('TENDERLOIN_data', index=False, header=True, columns= ['address', 'category', 'date', 'dayofweek', 'descript', 'incidntnum',
#        'location', 'pddistrict', 'pdid', 'resolution', 'time', 'x', 'y'])


# In[24]:


mask = results_df['pddistrict'] == 'PARK'
district_data['PARK'] = results_df[mask]
# df_PARK.to_csv('PARK_data', index=False, header=True, columns= ['address', 'category', 'date', 'dayofweek', 'descript', 'incidntnum',
#        'location', 'pddistrict', 'pdid', 'resolution', 'time', 'x', 'y'])


# In[25]:


mask = results_df['pddistrict'] == 'RICHMOND'
district_data['RICHMOND'] = results_df[mask]
# df_RICHMOND.to_csv('RICHMOND_data', index=False, header=True, columns= ['address', 'category', 'date', 'dayofweek', 'descript', 'incidntnum',
#        'location', 'pddistrict', 'pdid', 'resolution', 'time', 'x', 'y'])


# In[159]:


city_data = dict()
city_data['City'] = results_df


# In[27]:


# Identify and Remove null values
print("Null values in attributes: ")
null_columns=results_df.columns[results_df.isnull().any()]
results_df[null_columns].isnull().sum()


# # Build District wise Models and City wise Model

# In[30]:


print("District wise incidents -\n")
for key in district_data:
    district = district_data[key]
    print(key,"-",district.shape[0])
print(" \nMax incedent district - Southern - 398271")
print("\nMin incedent district - PARK - 124997")

# Plot incedents in each district
# from matplotlib import pyplot as plt
# plt.plot()


# # Program Strating Point for Data Science Life Cycle Phases

# In[178]:


accuracy_dictionary = dict()

for key in district_data:
    district = district_data[key]
    process_district(district, key)
#     break


# In[161]:


process_district(city_data['City'], 'city')


# In[162]:


# Execute only after executing first step
# results = accuracy_dictionary.copy()
print(accuracy_dictionary.copy())


# In[156]:


# district_model_accuracies = accuracy_dictionary.copy()


# In[174]:


def process_district(district, key):
    preprocessed_district = preprocess_district(district, key)
    district_X, district_y = feature_engineer_district(preprocessed_district, key, dimentionality_reduction=True)
    accuracy_dictionary[key] = train_test_district_model(district_X, district_y, key)
    return


# # Data Preprocessing and Data Exploration

# In[100]:


def preprocess_district(district, key):
    
    print("\nDataframe info -")
    print(district.info())
    
    print("\nFeatures -", district.columns)
    
    # Explore Unique values of features
    print("\nUnique values of features for", key,"district: \n")
    print("Address", len(district.address.unique()))
    print("Category", len(district.category.unique()))
    print("Day of week", len(district.dayofweek.unique()))
    print("Descript", len(district.descript.unique()))
    print("Pd district", len(district.pddistrict.unique()))
    print("Date",len(district.date.unique()))
    print("Time", len(district.time.unique()))
    print("X", len(district.x.unique()))
    print("Y", len(district.y.unique()))
    print("Incedent Number", len(district.incidntnum.unique()))
    print("Pd Id", len(district.pdid.unique()))
    print("Resolution", len(district.resolution.unique()))
    
    # Explore Unique discription for each crime type
#     print("Unique descriptions for each crime: \n")
#     unique_crime_types = district['category'].unique()
#     for crime in unique_crime_types:
# #         unique_crime_descriptions = district['description'].unique()
#         temp_df = district.loc[district['category'] == crime]
#         print(crime, "-", len(temp_df['descript'].unique()), "\n")
#         print(temp_df['descript'].unique(), "\n\n")

    # Drop unwanted features - 'resolution', 'pdid', 'incidntnum', 'descript', 'pddistrict', 'location', 'x', 'y'
    district = district.drop(['resolution', 'pdid', 'incidntnum', 'descript', 'pddistrict', 'location', 'x', 'y'], axis=1)
#     print(district)

    print("\nBefore removing null and duplicates - Rows -", district.shape[0])
    print("Before removing null and duplicates - Columns -", district.shape[1])
    
    # Remove Duplicate values for same incident reporting
    district = district.drop_duplicates()
    
    # Remove Null values
    district = district.dropna()
    
    print("\nAfter removing null and duplicates - Rows -", district.shape[0])
    print("After removing null and duplicates - Columns -", district.shape[1])
    
    # Capitalize Address
    district['address'] = district.address.apply(lambda x: x.upper())
    
    print("\nRemaining features -", district.columns)
    
#     print("\nPreprocessed", key,"district Dataframe -\n\n", district)
    
    # Save preprocessed district data frame
    district.to_csv('./Preprocessed_DataFrames/Preprocessed_'+key+'_dataframe.csv', index=False, header=True, columns= ['address', 'category', 'date', 'dayofweek', 'time'])
    
    return district


# # Feature Engineering (Extraction, Transformation, Reduction)

# In[101]:


import re
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


# In[102]:


from scipy.sparse import csr_matrix

def dataframetoCSRmatrix(df):
    nrows = len(df)
    nc = len(df.columns)
    idx = {}
    tid = 0
    nnz = nc * nrows
    
    cols= df.columns
    
    for col in cols:
        df[col] = df[col].apply(str)
        for name in df[col].unique():
            idx[col+name] = tid
            tid += 1
    
    ncols = len(idx)
    
    ind = np.zeros(nnz, dtype=np.int)
    val = np.zeros(nnz, dtype=np.int)
    ptr = np.zeros(nrows+1, dtype=np.int)
    
    i=0
    n=0
    
    for index,row in df.iterrows():
        for j,col in enumerate(cols):
            ind[j+n] = idx[col+row[col]]
            val[j+n] = 1
        ptr[i+1] = ptr[i] + nc
        n += nc
        i += 1
    
    mat = csr_matrix((val,ind,ptr), shape=(nrows,ncols), dtype=np.int)
#     mat.sort_indices()   
    
    return mat


# In[103]:


from sklearn.decomposition import RandomizedPCA, TruncatedSVD
import matplotlib.pyplot as plt

def reduce_dimentions_CSR_matrix(matrix, n_components = 30, n_iter = 20):
    
    tsvd = TruncatedSVD(n_components = n_components, n_iter = n_iter, random_state=42)
    
    reduced_matrix = tsvd.fit_transform(matrix)

    print("\nShape of matrix -", matrix.shape)
    print("\nShape of reduced_matrix -", reduced_matrix.shape)

    components = tsvd.explained_variance_
    
    print("Components values -", components)
    
    # Plot component's explained variance graph to find inflection point
    plt.plot(components)
    plt.show()
    
    return reduced_matrix


# In[126]:


def feature_engineer_district(district, key, dimentionality_reduction = False):
    
    # Extract 'day', 'month' and 'year' from 'date' feature and 'hour' from 'time' feature
    
    # Extract day
    district['day'] = district.date.apply(lambda x: convert_date_to_day(x))
    
    # Extract month
    district['month'] = district.date.apply(lambda x: convert_date_to_month(x))
    
    # Extract hour
    district['hour'] = district.time.apply(lambda x: convert_time_to_hour(x))
    
    # Extract year
    district['year'] = district.date.apply(lambda x: convert_date_to_year(x))
    
    print("\nAfter adding features - Rows -", district.shape[0])
    print("After adding features - Columns -", district.shape[1])
    
    print("\nNew features -", district.columns)
    
#     print("\nFeature Engineered", key,"district Dataframe -\n\n", district)
    
    # Removing unwanted features based on random forest's feature importance values
    district = district.drop(['date', 'time', 'year', 'day'], axis=1)
    
    # Mapping crime category to index and non-index crime based on FBI standards
    # http://gis.chicagopolice.org/clearmap_crime_sums/crime_types.html#N26
    
    print("\nUnique crime categories -", district['category'].unique())
    
#     print("\n Unique crime category incedents count -\n",pd.DataFrame({'Category_Count' : district.groupby(['category']).size()}).reset_index())
    
    # Define Index Crimes (More Serious)
    Index_crimes = ['ROBBERY', 'VEHICLE THEFT', 'BURGLARY', 'LARCENY/THEFT', 'ASSAULT', 'ARSON', 'SEX OFFENSES, FORCIBLE', 'SECONDARY CODES',  'RECOVERED VEHICLE']
#     print("\nIndex Crimes -", Index_crimes,"\nCount -",len(Index_crimes))
    
    # Define Non-Index Crimes (Less Serious)
    Non_Index_crimes = ['OTHER OFFENSES', 'NON-CRIMINAL', 'SUSPICIOUS OCC', 'FRAUD', 'FORGERY/COUNTERFEITING', 'WARRANTS', 'VANDALISM', 'MISSING PERSON', 'DISORDERLY CONDUCT', 'TRESPASS', 'WEAPON LAWS', 'DRUG/NARCOTIC', 'STOLEN PROPERTY', 'DRUNKENNESS', 'EMBEZZLEMENT', 'LOITERING', 'DRIVING UNDER THE INFLUENCE', 'PROSTITUTION', 'LIQUOR LAWS', 'EXTORTION', 'RUNAWAY', 'SUICIDE', 'BAD CHECKS', 'KIDNAPPING', 'FAMILY OFFENSES', 'BRIBERY', 'GAMBLING', 'SEX OFFENSES, NON FORCIBLE', 'PORNOGRAPHY/OBSCENE MAT', 'TREA']
#     print("\nNon Index Crimes -", Non_Index_crimes,"\nCount -",len(Non_Index_crimes))
    
    # Map category to Index crimes
    district.loc[district.category.isin(Index_crimes), 'Crime_Severity'] = "Index Crime (More Serious)"
    
    # Map category to Non Index crimes
    district.loc[district.category.isin(Non_Index_crimes), 'Crime_Severity'] = "Non Index Crime (Less Serious)"
    
    # Groupby category and examin Crime_Severity
#     district_gb = district.groupby(['category'])
#     for name, group in district_gb:
#         print("\n",name)
#         print("\n",group)

    # Removing unwanted feature category
    district = district.drop(['category'], axis=1)
        
    print("\nRemaining features -", district.columns)    
        
#     print("\nFeature Engineered", key,"district Dataframe -\n\n", district)
        
    # Identifying criteria for hotspots
    # Feature Importance - address > Crime_Severity > month > day > hour - based on random forest
    
    # Group by based on 'address', 'Crime_Severity'
    district_ac = district.groupby(['address', 'Crime_Severity']).size().reset_index(name='counts')
#     print("\n", district_ac)
    
    # Group by based on 'address', 'Crime_Severity', 'month'
    district_acm = district.groupby(['address', 'Crime_Severity', 'month']).size().reset_index(name='counts')
#     print("\n", district_acm)
    
    # Group by based on 'address', 'Crime_Severity', 'month', 'dayofweek'
    district_acmd = district.groupby(['address', 'Crime_Severity','month', 'dayofweek']).size().reset_index(name='counts')
#     print("\n", district_acmd)
    
    # Group by based on 'address', 'Crime_Severity', 'month', 'dayofweek', 'hour' 
    district_acmdh = district.groupby(['address', 'Crime_Severity', 'month', 'dayofweek', 'hour']).size().reset_index(name='counts')
#     print("\n", district_acmdh)
    
    # Identify address as hotspot if number of incidents happend at that address is above mean of incidents
    
    # Seperate Index Crime data and find hotspots for Index Crime
    mask = district_acmdh['Crime_Severity'] == 'Index Crime (More Serious)'
    district_acmdh_Index_crime = district_acmdh[mask]
    mean_Index_Crime = district_acmdh_Index_crime['counts'].mean()
    mask = district_acmdh_Index_crime['counts'] >= int(mean_Index_Crime)+1
    district_hotspot_Index_crime = district_acmdh_Index_crime[mask]
    print("\n district_acmdh - Index Crime - Mean -", mean_Index_Crime)
#     print("\n",district_hotspot_Index_crime) 
    
    # Seperate Non Index Crime data and find hotspots for Non Index Crime
    mask = district_acmdh['Crime_Severity'] == 'Non Index Crime (Less Serious)'
    district_acmdh_Non_Index_crime = district_acmdh[mask]
    mean_Non_Index_Crime = district_acmdh_Non_Index_crime['counts'].mean()
    mask = district_acmdh_Non_Index_crime['counts'] >= int(mean_Non_Index_Crime)+1
    district_hotspot_Non_Index_crime = district_acmdh_Non_Index_crime[mask]
    print("\n district_acmdh - Non Index Crime - Mean -", mean_Non_Index_Crime)
#     print("\n",district_hotspot_Non_Index_crime)
    
    # Find hotspot 'addresses' for district at hand
    hotspots = set()
    Index_hotspots = district_hotspot_Index_crime['address'].unique()
    Non_Index_hotspots = district_hotspot_Non_Index_crime['address'].unique()
    print("\nIndex Crime Hotspots -", len(Index_hotspots))
    print("\nNon Index Crime Hotspots -", len(Non_Index_hotspots))
    for ih in Index_hotspots:
        hotspots.add(ih)
    for nih in Non_Index_hotspots:
        hotspots.add(nih)
    print("\nTotal Index and Non Index Crime Hotspots -", len(hotspots))
    
    datasize_before = district.shape[0]
    print("\nBefore considering hotspots - Rows -", district.shape[0])
    print("Before considering hotspots - Columns -", district.shape[1])
    
    # Filter data based on 'address' - only consider hotspot 'address'
    mask = district['address'].isin(hotspots)
    district = district[mask]
    
    datasize_after = district.shape[0]
    print("\nAfter considering hotspots - Rows -", district.shape[0])
    print("After considering hotspots - Columns -", district.shape[1])
    
    print("\nData shrunk from", datasize_before," rows to", datasize_after, "rows - By ~", 100 - (datasize_after/datasize_before), "% - for ", key,"district.")
    
    print("\nRemaining features -", district.columns)    
        
#     print("\nFeature Engineered", key,"district Dataframe -\n\n", district)
    
    # Save feature engineered district data frame
    district.to_csv('./Feature_Engineered_DataFrames/Feature_Engineered_'+key+'_dataframe.csv', index=False, header=True, columns= ['address', 'category', 'dayofweek', 'month', 'hour'])
    
    # Read feature engineered district data frame
    # district = pd.read_csv('/Feature_Engineered_DataFrames/Feature_Engineered_'+key+'_dataframe.csv',encoding = "ISO-8859-1")
    
    # Target
    district_y_crime_severity = district['Crime_Severity']
    print("\nTarget shape", district_y_crime_severity.shape)
#     print("\nTarget -\n", district_y_crime_severity)
    
    # Features
    district_X_features = district.drop(['Crime_Severity'],axis=1)
    print("\nFeatures shape", district_X_features.shape)
#     print("\nFeatures -\n", district_X_features)
    
    district_y_crime_severity_count = pd.value_counts(district_y_crime_severity.values, sort=False)
    print("\nIndex - Non Index crime counts -\n",district_y_crime_severity_count)
    
    print("\nUnique values of attributes for ", key," district: \n")
    print("Address", len(district_X_features.address.unique()))
    print("Day of week", len(district_X_features.dayofweek.unique()))
    print("Month", len(district_X_features.month.unique()))
    print("Hour", len(district_X_features.hour.unique()))
    print("Total unique values of categorical features -", len(district_X_features.hour.unique()) + len(district_X_features.month.unique()) + len(district_X_features.dayofweek.unique()) + len(district_X_features.address.unique()))
    
    # Convert categorical data to one-hot encoded data frame and convert that sparse data frame to CSR Matrix
    matrix = dataframetoCSRmatrix(district_X_features)

    print("\n Shape of district_X_features's CSR Matrix -", matrix.shape)
    
    if dimentionality_reduction:
        matrix = reduce_dimentions_CSR_matrix(matrix)
    
    # Target
    print("\nTarget shape", district_y_crime_severity.shape)
#     print("\nTarget -\n", district_y_crime_severity)
    
    # Features
    print("\nFeatures shape", matrix.shape)
#     print("\nFeatures -\n", matrix)
    
    return matrix, district_y_crime_severity.values


# # Model Training, Testing and Evaluation

# In[105]:


# Report best candidate function with optimal parameters for RandomizedSerach
def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")


# In[175]:


from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import RandomizedSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score

def train_test_district_model(district_X, district_y, key):
    
#     # Simple MLP
    
#     X_train, X_test, y_train, y_test = train_test_split(district_X, district_y, test_size=0.25)
    
#     mlpclf = MLPClassifier(solver='adam', alpha=0.001,hidden_layer_sizes=(5, 5), random_state=1, max_iter=200)
    
#     mlpclf.fit(X_train, y_train)

#     predicted_y_train = mlpclf.predict(X_train)
        
#     train_f1_score = f1_score(y_train, predicted_y_train, average='micro')
#     print("Simple MLP Train accuracy:",train_f1_score)
        
#     predicted_y_test = mlpclf.predict(X_test)
        
#     test_f1_score = f1_score(y_test, predicted_y_test, average='micro')
#     print("Simple MLP Test accuracy:",test_f1_score)
    
#     return train_f1_score, test_f1_score
    
    # MLP with Stritified K=2 fold
    
    n_splits = 2
    skf = StratifiedKFold(n_splits = n_splits)
    skf.get_n_splits(district_X, district_y)
    
    print("\nStratified K folds -",skf)
    
    count = 0
    
    train_accuracy = []
    test_accuracy = []
    
    for train, test in skf.split(district_X, district_y):
        
        print("\nTRAIN :", train, "\nTEST :", test)
        
        X_train, X_test = district_X[train], district_X[test]
        y_train, y_test = district_y[train], district_y[test]
        
        # Tried randomized search but it's talking too long even on AWS EC2 instance
        
#         param_dist = {
#             "solver": ['sgd', 'adam', 'lbfgs'],
#             "alpha": [0.01,0.001,1e-4,1e-5],
#             "hidden_layer_sizes": [(5, 5), (10, 5), (5, 7, 10), (10, 7, 5)],
#             "max_iter": [50, 100, 150, 200],
#             "learning_rate_init": [0.01,0.001,1e-4,1e-5]
#         }
#         n_iter_search = 20
#         random_search = RandomizedSearchCV(mlclf, param_distributions=param_dist, n_iter=n_iter_search)
#         start = time()
#         random_search.fit(X, y)
#         print("RandomizedSearchCV took %.2f seconds for %d candidates parameter settings." % ((time() - start), n_iter_search))
#         report(random_search.cv_results_)

#         mlpclf = MLPClassifier(solver='sgd', alpha=1e-5,hidden_layer_sizes=(5, 5), random_state=1)
#         mlpclf = MLPClassifier(solver='sgd', alpha=1e-5,hidden_layer_sizes=(10, 5), random_state=1)
#         mlpclf = MLPClassifier(solver='sgd', alpha=1e-5,hidden_layer_sizes=(5, 7, 10), random_state=1)
#         mlpclf = MLPClassifier(solver='sgd', alpha=1e-5,hidden_layer_sizes=(10, 7, 5), random_state=1)
        
        mlpclf = MLPClassifier(solver='adam', alpha=0.001,hidden_layer_sizes=(5, 5), random_state=1, max_iter=200)
#         mlpclf = MLPClassifier(solver='adam', alpha=1e-5,hidden_layer_sizes=(10, 5), random_state=1)
#         mlpclf = MLPClassifier(solver='adam', alpha=1e-5,hidden_layer_sizes=(5, 7, 10), random_state=1)
#         mlpclf = MLPClassifier(solver='adam', alpha=1e-5,hidden_layer_sizes=(10, 7, 5), random_state=1)
        
#         mlpclf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 5), random_state=1)
#         mlpclf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(10, 5), random_state=1)
#         mlpclf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 7, 10), random_state=1)
#         mlpclf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(10, 7, 5), random_state=1)
        
        mlpclf.fit(X_train, y_train)

        predicted_y_train = mlpclf.predict(X_train)
        
        train_f1_score = f1_score(y_train, predicted_y_train, average='micro')
        train_accuracy.append(train_f1_score)
        print("Train accuracy", count,":",train_f1_score)
        
        predicted_y_test = mlpclf.predict(X_test)
        
        test_f1_score = f1_score(y_test, predicted_y_test, average='micro')
        test_accuracy.append(test_f1_score)
        print("Test accuracy", count,":",test_f1_score)
        
#         print("Classification report -", classification_report(y_test,predictions))

        # Store predictions to a file
#         with open('predictions_Crime_Type_',key,'.dat','a') as file:
#             for p in predicted_y_train:
#                 file.write(str(p)+"\n")
#             for p in predicted_y_test:
#                 file.write(str(p)+"\n")
                
        count += 1
    
    total_tain_acccuracy = 0
    for ta in train_accuracy:
        total_tain_acccuracy += ta
        
    total_test_acccuracy = 0
    for ta in test_accuracy:
        total_test_acccuracy += ta
    
    avg_train_accuracy = total_tain_acccuracy/n_splits
    avg_test_accuracy = total_test_acccuracy/n_splits
    
    print("Average accuracies - Train -", avg_train_accuracy, "Test -", avg_test_accuracy)
    
    return avg_train_accuracy, avg_test_accuracy


# # Knowledge Mining and Visualizations

# In[151]:


# Tried parameter tuning with random different values of solver function, hidden layers & nodes, alpha, max_iter for BAYVIEW district


# In[130]:


# MLP

# Solver selection

# 1.
# mlpclf = MLPClassifier(solver='sgd', alpha=1e-5,hidden_layer_sizes=(5, 5), random_state=1) - 
# Average accuracies - Train - 0.579875931922 Test - 0.579875931922

# 2.
# mlpclf = MLPClassifier(solver='adam', alpha=1e-5,hidden_layer_sizes=(5, 5), random_state=1)
# Average accuracies - Train - 0.690259253642 Test - 0.600436676233

# 3.
# mlpclf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 5), random_state=1)
# Average accuracies - Train - 0.671918101601 Test - 0.607617946048


# In[ ]:


# Hidden Layers selection

# 1.
# mlpclf = MLPClassifier(solver='adam', alpha=1e-5,hidden_layer_sizes=(5, 5), random_state=1)
# Average accuracies - Train - 0.690259253642 Test - 0.600436676233

# 2.
# mlpclf = MLPClassifier(solver='adam', alpha=0.001,hidden_layer_sizes=(10, 5), random_state=1)
# Average accuracies - Train - 0.742054304791 Test - 0.588138538267

# 3.
# mlpclf = MLPClassifier(solver='adam', alpha=1e-5,hidden_layer_sizes=(5, 7, 10), random_state=1)
# Average accuracies - Train - 0.7099973349 Test - 0.594414365124

# 4.
# mlpclf = MLPClassifier(solver='adam', alpha=1e-5,hidden_layer_sizes=(10, 7, 5), random_state=1)
# Average accuracies - Train - 0.739989942155 Test - 0.588154043622


# In[138]:


# Alpha (regularization parameter selection - prevents overfitting)

# 1.
# mlpclf = MLPClassifier(solver='adam', alpha=0.01,hidden_layer_sizes=(5, 5), random_state=1)
# Average accuracies - Train - 0.66968292426 Test - 0.608502621902

# 2.
# mlpclf = MLPClassifier(solver='adam', alpha=0.001,hidden_layer_sizes=(5, 5), random_state=1)
# Average accuracies - Train - 0.683890305034 Test - 0.601735296633

# 3.
# mlpclf = MLPClassifier(solver='adam', alpha=1e-4,hidden_layer_sizes=(5, 5), random_state=1)
# Average accuracies - Train - 0.686694491125 Test - 0.598744851996

# 4.
# mlpclf = MLPClassifier(solver='adam', alpha=1e-5,hidden_layer_sizes=(5, 5), random_state=1)
# Average accuracies - Train - 0.690259253642 Test - 0.600436676233


# In[148]:


# max_iter (epochs)

# 1.
# mlpclf = MLPClassifier(solver='adam', alpha=0.001,hidden_layer_sizes=(5, 5), random_state=1, max_iter= 200)
# Average accuracies - Train - 0.690259253642 Test - 0.600436676233

# 2.
# mlpclf = MLPClassifier(solver='adam', alpha=0.001,hidden_layer_sizes=(5, 5), random_state=1, max_iter=50)
# Average accuracies - Train - 0.66968292426 Test - 0.608502621902

# 3. 
# mlpclf = MLPClassifier(solver='adam', alpha=0.001,hidden_layer_sizes=(5, 5), random_state=1, max_iter=100)
# Average accuracies - Train - 0.681386165907 Test - 0.605258671259

# 4.
# mlpclf = MLPClassifier(solver='adam', alpha=0.001,hidden_layer_sizes=(5, 5), random_state=1, max_iter=150)
# Average accuracies - Train - 0.683890305034 Test - 0.601735296633


# In[197]:


# Comparision for BAYVIEW district

# 1) Simple MLP - Train - 0.581402210109 Test - 0.577940927304
# 2) MLP with K-fold - Average accuracies - Train - 0.690259253642 Test - 0.600436676233
# 3) MLP with K-fold and TruncatedSVD - Average accuracies - Average accuracies - Train - 0.67009004209 Test - 0.607983906856


# In[199]:


# data to plot
n_groups = 3
train_accuracy = (0.581402210109, 0.690259253642, 0.67009004209)
test_accuracy = (0.577940927304, 0.600436676233, 0.607983906856)
 
# create plot
fig, ax = plt.subplots()
index = np.arange(n_groups)
bar_width = 0.20
opacity = 0.8
 
rects1 = plt.bar(index, train_accuracy, bar_width,
                 alpha=opacity,
                 color='b',
                 label='Train')
 
rects2 = plt.bar(index + bar_width, test_accuracy, bar_width,
                 alpha=opacity,
                 color='g',
                 label='Test')
 
plt.xlabel('Type of Accuracy')
plt.ylabel('Scores')
plt.title('Accuracy comparision for Simple MLP, MLP with K -fold, MLP with K-fold and TruncatedSVD')
plt.xticks(index + bar_width, ('Simple MLP', 'MLP with K -fold', 'MLP with K-fold and TruncatedSVD'))
plt.legend()
plt.rcParams["figure.figsize"] = [7,5]
# plt.tight_layout()
plt.show()


# In[204]:


# district wise model's accuracies
district_model_accuracies = accuracy_dictionary.copy()
district_model_accuracies


# In[196]:


district_names = []
train_accuracy_list = []
test_accuracy_list = []

for key in district_model_accuracies:
    district_names.append(key)
    train_accuracy_list.append(district_model_accuracies[key][0])
    test_accuracy_list.append(district_model_accuracies[key][1])

# data to plot
n_groups = 11
train_accuracy = tuple(train_accuracy_list)
test_accuracy = tuple(test_accuracy_list)
 
# create plot
fig, ax = plt.subplots()
index = np.arange(n_groups)
bar_width = 0.20
opacity = 0.8
 
rects1 = plt.bar(index, train_accuracy, bar_width,
                 alpha=opacity,
                 color='b',
                 label='Train')
 
rects2 = plt.bar(index + bar_width, test_accuracy, bar_width,
                 alpha=opacity,
                 color='g',
                 label='Test')
 
plt.xlabel('Name of District/City')
plt.ylabel('Accuracy')
plt.title('Accuracy Comparision among Districts of SF and City of SF')
plt.xticks(index + bar_width, tuple(district_names))
plt.legend()
plt.rcParams["figure.figsize"] = [15,10]
# plt.tight_layout()
plt.show()


# In[214]:


# data to plot
n_groups = 2
train_accuracy = (0.206146366553, 0.690259253642)
test_accuracy = (0.205679751649, 0.600436676233)
 
# create plot
fig, ax = plt.subplots()
index = np.arange(n_groups)
bar_width = 0.10
opacity = 0.8
 
rects1 = plt.bar(index, train_accuracy, bar_width,
                 alpha=opacity,
                 color='b',
                 label='Train')
 
rects2 = plt.bar(index + bar_width, test_accuracy, bar_width,
                 alpha=opacity,
                 color='g',
                 label='Test')
 
plt.xlabel('Type of Approach')
plt.ylabel('Accuracy - F1 Score')
plt.title('Comparison(f1 score) between Appr. 1 (without domain know.) vs Appr. 2 (with domain know.) on BAYVIEW district')
plt.xticks(index + bar_width, ('Approach 1', 'Approach 2'))
plt.legend()
plt.rcParams["figure.figsize"] = [7,5]
# plt.tight_layout()
plt.show()

