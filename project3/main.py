#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


def load_insulin_data():
    insulin_df = pd.read_csv("InsulinData.csv", parse_dates=[['Date', 'Time']], \
                             keep_date_col=True, low_memory=False)
    insulin_df = insulin_df[['Date_Time', 'BWZ Carb Input (grams)']]
    insulin_df = insulin_df.rename(columns={'BWZ Carb Input (grams)': 'Carb_Input'})
    print('extracting insulin df: ' + str(insulin_df.shape))
    return insulin_df


# In[3]:


def load_cgm_data(): 
    cgm_df = pd.read_csv('CGMData.csv', parse_dates=[['Date', 'Time']], keep_date_col=True, low_memory=False)
    cgm_df = cgm_df[['Date_Time', 'Index','Sensor Glucose (mg/dL)', 'Date', 'Time']]
    cgm_df = cgm_df.rename(columns={'Sensor Glucose (mg/dL)': 'Sensor_Glucose'})
    print('extracting cgm df: ' + str(cgm_df.size))
    return cgm_df


# In[4]:


insulin_df, cgm_df = load_insulin_data(), load_cgm_data()
display(insulin_df.size)
display(cgm_df.size)


# In[5]:


import datetime as dt


# In[6]:


inf = dt.datetime(2100, 12, 12)


# In[124]:


def extract_meal_start_times(insulin_df, cgm_df):
    
    insulin_df = insulin_df[(insulin_df['Carb_Input'].notna()) & (insulin_df['Carb_Input'] != 0)]
    insulin_df = insulin_df.set_index('Date_Time').sort_index().reset_index()
    
    # include only those meal periods for which the next carb intake is atleast after 2 hrs
    mask = (insulin_df['Date_Time'].shift(-1, fill_value=inf) - insulin_df['Date_Time'] \
            >= dt.timedelta(hours=2))
    
    insulin_df = insulin_df[mask]
    
    # column rename is required for the following merge
    insulin_df = insulin_df.rename(columns = {'Date_Time': 'Pseudo_Start_Time'})
    
    cgm_df = cgm_df[cgm_df['Sensor_Glucose'].notna()]
    cgm_df = cgm_df.set_index('Date_Time').sort_index().reset_index()
    
    meal_df = pd.merge_asof(insulin_df, cgm_df, left_on='Pseudo_Start_Time', \
                            right_on='Date_Time', direction='forward')[['Date_Time', 'Carb_Input']]
    
    min, max = meal_df['Carb_Input'].min(), meal_df['Carb_Input'].max()
    
    # binning BWZ Carb Input (grams) into bins of range 20
    meal_df['Carb_Input'] = (meal_df['Carb_Input'] - min) // 20
    
    return meal_df
    
meal_df = extract_meal_start_times(insulin_df, cgm_df)


# In[423]:


def compute_meal_data_matrix(cgm_df, meal_df):
    
    meal_data_list = []
    ground_truth = []
    
    for _, row in meal_df.iterrows():
        
        meal_start = row['Date_Time'] - pd.DateOffset(minutes=30)
        meal_end = row['Date_Time'] + pd.DateOffset(hours=2)
        meal = cgm_df.loc[(cgm_df['Date_Time'] >= meal_start) & (cgm_df['Date_Time'] < meal_end)]
        
        # remove meal periods with <30 readings
        if (meal_df.shape[0] < 30):
            continue
            
        meal = meal[meal['Sensor_Glucose'].notna()]
        meal = meal.set_index('Date_Time').sort_index().reset_index()

        # remove readings <300 seconds apart
        mask = (meal['Date_Time'].shift(-1, fill_value=inf) - meal['Date_Time'] \
                >= dt.timedelta(seconds=300))
        
        meal = meal[mask]

        # only include meal_period if it has exactly 30 readings
        if (meal.shape[0] == 30):
            meal_data_list.append(meal['Sensor_Glucose'])
            ground_truth.append(row['Carb_Input'])
        
    feature_matrix = pd.concat(meal_data_list, axis=1).transpose()
    feature_matrix['label'] = ground_truth 
    return feature_matrix


# In[424]:


meal_data_matrix = compute_meal_data_matrix(cgm_df, meal_df)


# In[215]:


from scipy.stats import entropy, iqr
from scipy.signal import periodogram


# In[267]:


def compute_meal_feature_matrix(meal_data_matrix):
    
    # exclude ground truth
    _input = meal_data_matrix.iloc[:, :-1]
    
    features = pd.DataFrame()
    
    velocity = _input.diff(axis=1).dropna(axis=1, how='all')
    features['velocity_min'] = velocity.min(axis=1)
    features['velocity_max'] = velocity.max(axis=1)
    features['velocity_mean'] = velocity.mean(axis=1)

    acceleration = velocity.diff(axis=1).dropna(axis=1, how='all')
    features['acceleration_min'] = acceleration.min(axis=1)
    features['acceleration_max'] = acceleration.max(axis=1)
    features['acceleration_mean'] = acceleration.mean(axis=1)

    features['entropy'] = _input.apply(lambda row: entropy(row, base=2), axis=1)
    features['iqr'] = _input.apply(lambda row: entropy(row, base=2), axis=1)
    
    fft_values = _input.apply(lambda row: np.fft.fft(row), axis=1)

    # get the indices of the frequencies sorted by decreasing amplitude
    fft_indices = fft_values.apply(lambda row: np.argsort(np.abs(row))[::-1])

    # select the first 6 peaks of each row
    fft_peaks = fft_indices.apply(lambda row: row[:6])
    fft_peaks = fft_peaks.apply(pd.Series)
    fft_peaks.columns = ['fft_max_' + str(i+1) for i in fft_peaks.apply(pd.Series).columns]
    
    features = pd.concat([features, fft_peaks], axis=1)
    
    _input = meal_data_matrix.iloc[:, :-1]
    psd = _input.apply(lambda row: periodogram(row)[1], axis=1)
    psd = psd.apply(lambda row: [np.mean(row[0:5]), np.mean(row[5:10]), np.mean(row[10:16])])
    psd = psd.apply(pd.Series)
    psd.columns = ['psd1', 'psd2', 'psd3']
 
    features = pd.concat([features, psd], axis=1)
    
    return features


# In[317]:


meal_feature_matrix = compute_meal_feature_matrix(meal_data_matrix)


# In[319]:


from sklearn.preprocessing import StandardScaler


# In[322]:


scaler = StandardScaler()
meal_feature_matrix_scaled = scaler.fit_transform(meal_feature_matrix)


# 
# ## KMeans Clustering

# In[ ]:


from sklearn.cluster import KMeans


# In[323]:


kmeans = KMeans(n_clusters=7, n_init=20, max_iter=100)

# Fit the model to the data
kmeans.fit(meal_feature_matrix_scaled)

# Get the cluster labels for each data point
labels = kmeans.labels_
cluster_centers = kmeans.cluster_centers_


# In[329]:


from sklearn.metrics.cluster import contingency_matrix


# In[331]:


ground_truth = meal_data_matrix['label']
cont_matrix = contingency_matrix(ground_truth, labels)


# In[ ]:


num_samples = len(meal_data_matrix)


# In[480]:


# Calculate the SSE
kmeans_sse = kmeans.inertia_

# Calculate the purity
kmeans_purity = np.sum(np.amax(cont_matrix, axis=0)) / num_samples

# Calculate the entropy
kmeans_entropy = -np.sum((np.sum(cont_matrix, axis=1) / num_samples) *
                  np.log2(np.sum(cont_matrix, axis=1) / num_samples))

print("KMeans SSE: ", kmeans_sse)
print("KMeans Purity: ", kmeans_purity)
print("KMeans Entropy: ", kmeans_entropy)


# 
# ## DBSCAN Clustering

# In[467]:


from sklearn.cluster import DBSCAN
from sklearn.metrics import mean_squared_error


# In[443]:


dbscan = DBSCAN(eps=2.4, min_samples=5).fit(meal_feature_matrix_scaled)


# In[451]:


labels = dbscan.labels_.astype(float)


# In[453]:


labels[labels == -1] = np.nan


# In[456]:


cont_matrix = pd.crosstab(ground_truth, labels, dropna = False)


# In[481]:


# Calculate the SSE
centroids = meal_feature_matrix_scaled.copy()

# Change np.nan back to -1
labels = np.nan_to_num(labels, nan=-1)

for label in np.unique(labels[1:]):
    centroids[labels == label] =  centroids[labels == label].mean(axis=0)
    
dbscan_sse = mean_squared_error(meal_feature_matrix_scaled, centroids)

# Calculate the purity
dbscan_purity = np.sum(np.amax(cont_matrix, axis=0)) / num_samples

# Calculate the entropy
dbscan_entropy = -np.sum((np.sum(cont_matrix, axis=1) / num_samples) *
                  np.log2(np.sum(cont_matrix, axis=1) / num_samples))

print("DBSCAN SSE: ", dbscan_sse)
print("DBSCAN Purity: ", dbscan_purity)
print("DBSCAN Entropy: ", dbscan_entropy)


# In[482]:


# Save the clustering evaluation results in a CSV file
results = np.array([[kmeans_sse, dbscan_sse, kmeans_entropy, dbscan_entropy, kmeans_purity, dbscan_purity]])
np.savetxt("Results.csv", results, delimiter=",")

