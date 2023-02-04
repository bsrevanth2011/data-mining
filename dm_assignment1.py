#!/usr/bin/env python
# coding: utf-8

# In[302]:


import pandas as pd
import datetime
import numpy as np


# In[303]:


# data the provided in reverse chornological order, load it in chronological order
cgm_df = pd.read_csv('./CGMData.csv', low_memory=False, usecols = ['Date', 'Time', 'Sensor Glucose (mg/dL)'])[::-1]


# In[304]:


cgm_df['DateTime'] = pd.to_datetime(cgm_df['Date'] + ' ' + cgm_df['Time'], format='%m/%d/%Y %H:%M:%S')


# In[305]:


cgm_df['Date'] = cgm_df['DateTime'].dt.date


# In[306]:


cgm_df = cgm_df[cgm_df['Sensor Glucose (mg/dL)'].notna()]


# In[307]:


ip_df = pd.read_csv('./InsulinData.csv', low_memory=False, usecols = ['Date', 'Time', 'Alarm'])


# In[308]:


""" 
Using the last matched row as this is the first occurence of 
the Alarm since the Insulin Pump data is in reverse chronological order.
"""
row = ip_df.loc[ip_df['Alarm'] == 'AUTO MODE ACTIVE PLGM OFF'].iloc[-1]
auto_start_datetime = pd.to_datetime(row['Date'] + ' ' + row['Time'], format='%m/%d/%Y %H:%M:%S')


# In[309]:


# separate out auto and manaul mode data

auto_cgm_df = cgm_df[cgm_df['DateTime'] <= auto_start_datetime]
manual_cgm_df = cgm_df[cgm_df['DateTime'] > auto_start_datetime]


# In[310]:


# daytime interval

daytime_start = datetime.time(6,0,0)
daytime_end = datetime.time(23,59,59)


# In[311]:


#overnight interval

overnight_start = datetime.time(0,0,0)
overnight_end = datetime.time(5,59,59)


# In[312]:


# removing entries for days with reading count < 80% of 288

dates_to_drop = auto_cgm_df.set_index('Date').groupby('Date').count()\
    .where(lambda g: g < 0.8 * 288).dropna().index

auto_cgm_df = auto_cgm_df.set_index('Date').drop(dates_to_drop, axis = 'index')


# In[313]:


auto_cgm_df = auto_cgm_df.reset_index().set_index('DateTime')


# In[314]:


auto_cgm_daytime_df = auto_cgm_df.between_time(daytime_start, daytime_end)


# In[315]:


auto_cgm_overnight_df = auto_cgm_df.between_time(overnight_start, overnight_end)


# In[316]:


auto_cgm_df = auto_cgm_df.reset_index().set_index('Date')
auto_cgm_daytime_df = auto_cgm_daytime_df.reset_index().set_index('Date')
auto_cgm_overnight_df = auto_cgm_overnight_df.reset_index().set_index('Date')


# In[317]:


# removing entries for days with reading count < 80% of 288

dates_to_drop = manual_cgm_df.set_index('Date').groupby('Date').count()\
    .where(lambda g: g < 0.8 * 288).dropna().index

manual_cgm_df = manual_cgm_df.set_index('Date').drop(dates_to_drop, axis = 'index')


# In[318]:


manual_cgm_df = manual_cgm_df.reset_index().set_index('DateTime')


# In[319]:


manual_cgm_daytime_df = manual_cgm_df.between_time(daytime_start, daytime_end)


# In[320]:


manual_cgm_overnight_df = manual_cgm_df.between_time(overnight_start, overnight_end)


# In[321]:


manual_cgm_df = manual_cgm_df.reset_index().set_index('Date')
manual_cgm_daytime_df = manual_cgm_daytime_df.reset_index().set_index('Date')
manual_cgm_overnight_df = manual_cgm_overnight_df.reset_index().set_index('Date')


# In[322]:


pcnt_time_hyperglycemia_wholeday_auto = auto_cgm_df[auto_cgm_df['Sensor Glucose (mg/dL)'] > 180].groupby('Date').agg(lambda x: x.count() * 100 / 288)['Sensor Glucose (mg/dL)'].mean()

pcnt_time_hyperglycemia_critical_wholeday_auto = auto_cgm_df[auto_cgm_df['Sensor Glucose (mg/dL)'] > 250].groupby('Date').agg(lambda x: x.count() * 100 / 288)['Sensor Glucose (mg/dL)'].mean()

pcnt_time_hyperglycemia_range_wholeday_auto = auto_cgm_df[auto_cgm_df['Sensor Glucose (mg/dL)'] <= 180].groupby('Date').agg(lambda x: x.count() * 100 / 288)['Sensor Glucose (mg/dL)'].mean()

pcnt_time_hyperglycemia_range_secondary_wholeday_auto = auto_cgm_df[auto_cgm_df['Sensor Glucose (mg/dL)'] <= 150].groupby('Date').agg(lambda x: x.count() * 100 / 288)['Sensor Glucose (mg/dL)'].mean()

pcnt_time_hyperglycemia_hypoglycemia_lvl1_wholeday_auto = auto_cgm_df[auto_cgm_df['Sensor Glucose (mg/dL)'] < 70].groupby('Date').agg(lambda x: x.count() * 100 / 288)['Sensor Glucose (mg/dL)'].mean()

pcnt_time_hyperglycemia_hypoglycemia_lvl2_wholeday_auto = auto_cgm_df[auto_cgm_df['Sensor Glucose (mg/dL)'] < 54].groupby('Date').agg(lambda x: x.count() * 100 / 288)['Sensor Glucose (mg/dL)'].mean()


# In[323]:


pcnt_time_hyperglycemia_daytime_auto = auto_cgm_daytime_df[auto_cgm_daytime_df['Sensor Glucose (mg/dL)'] > 180].groupby('Date').agg(lambda x: x.count() * 100 / 288)['Sensor Glucose (mg/dL)'].mean()

pcnt_time_hyperglycemia_critical_daytime_auto = auto_cgm_daytime_df[auto_cgm_daytime_df['Sensor Glucose (mg/dL)'] > 250].groupby('Date').agg(lambda x: x.count() * 100 / 288)['Sensor Glucose (mg/dL)'].mean()

pcnt_time_hyperglycemia_range_daytime_auto = auto_cgm_daytime_df[auto_cgm_daytime_df['Sensor Glucose (mg/dL)'] <= 180].groupby('Date').agg(lambda x: x.count() * 100 / 288)['Sensor Glucose (mg/dL)'].mean()

pcnt_time_hyperglycemia_range_secondary_daytime_auto = auto_cgm_daytime_df[auto_cgm_daytime_df['Sensor Glucose (mg/dL)'] <= 150].groupby('Date').agg(lambda x: x.count() * 100 / 288)['Sensor Glucose (mg/dL)'].mean()

pcnt_time_hyperglycemia_hypoglycemia_lvl1_daytime_auto = auto_cgm_daytime_df[auto_cgm_daytime_df['Sensor Glucose (mg/dL)'] < 70].groupby('Date').agg(lambda x: x.count() * 100 / 288)['Sensor Glucose (mg/dL)'].mean()

pcnt_time_hyperglycemia_hypoglycemia_lvl2_daytime_auto = auto_cgm_daytime_df[auto_cgm_daytime_df['Sensor Glucose (mg/dL)'] < 54].groupby('Date').agg(lambda x: x.count() * 100 / 288)['Sensor Glucose (mg/dL)'].mean()


# In[324]:


pcnt_time_hyperglycemia_overnight_auto = auto_cgm_overnight_df[auto_cgm_overnight_df['Sensor Glucose (mg/dL)'] > 180].groupby('Date').agg(lambda x: x.count() * 100 / 288)['Sensor Glucose (mg/dL)'].mean()

pcnt_time_hyperglycemia_critical_overnight_auto = auto_cgm_overnight_df[auto_cgm_overnight_df['Sensor Glucose (mg/dL)'] > 250].groupby('Date').agg(lambda x: x.count() * 100 / 288)['Sensor Glucose (mg/dL)'].mean()

pcnt_time_hyperglycemia_range_overnight_auto = auto_cgm_overnight_df[auto_cgm_overnight_df['Sensor Glucose (mg/dL)'] <= 180].groupby('Date').agg(lambda x: x.count() * 100 / 288)['Sensor Glucose (mg/dL)'].mean()

pcnt_time_hyperglycemia_range_secondary_overnight_auto = auto_cgm_overnight_df[auto_cgm_overnight_df['Sensor Glucose (mg/dL)'] <= 150].groupby('Date').agg(lambda x: x.count() * 100 / 288)['Sensor Glucose (mg/dL)'].mean()

pcnt_time_hyperglycemia_hypoglycemia_lvl1_overnight_auto = auto_cgm_overnight_df[auto_cgm_overnight_df['Sensor Glucose (mg/dL)'] < 70].groupby('Date').agg(lambda x: x.count() * 100 / 288)['Sensor Glucose (mg/dL)'].mean()

pcnt_time_hyperglycemia_hypoglycemia_lvl2_overnight_auto = auto_cgm_overnight_df[auto_cgm_overnight_df['Sensor Glucose (mg/dL)'] < 54].groupby('Date').agg(lambda x: x.count() * 100 / 288)['Sensor Glucose (mg/dL)'].mean()


# In[325]:


pcnt_time_hyperglycemia_wholeday_manual = manual_cgm_df[manual_cgm_df['Sensor Glucose (mg/dL)'] > 180].groupby('Date').agg(lambda x: x.count() * 100 / 288)['Sensor Glucose (mg/dL)'].mean()

pcnt_time_hyperglycemia_critical_wholeday_manual = manual_cgm_df[manual_cgm_df['Sensor Glucose (mg/dL)'] > 250].groupby('Date').agg(lambda x: x.count() * 100 / 288)['Sensor Glucose (mg/dL)'].mean()

pcnt_time_hyperglycemia_range_wholeday_manual = manual_cgm_df[manual_cgm_df['Sensor Glucose (mg/dL)'] <= 180].groupby('Date').agg(lambda x: x.count() * 100 / 288)['Sensor Glucose (mg/dL)'].mean()

pcnt_time_hyperglycemia_range_secondary_wholeday_manual = manual_cgm_df[manual_cgm_df['Sensor Glucose (mg/dL)'] <= 150].groupby('Date').agg(lambda x: x.count() * 100 / 288)['Sensor Glucose (mg/dL)'].mean()

pcnt_time_hyperglycemia_hypoglycemia_lvl1_wholeday_manual = manual_cgm_df[manual_cgm_df['Sensor Glucose (mg/dL)'] < 70].groupby('Date').agg(lambda x: x.count() * 100 / 288)['Sensor Glucose (mg/dL)'].mean()

pcnt_time_hyperglycemia_hypoglycemia_lvl2_wholeday_manual = manual_cgm_df[manual_cgm_df['Sensor Glucose (mg/dL)'] < 54].groupby('Date').agg(lambda x: x.count() * 100 / 288)['Sensor Glucose (mg/dL)'].mean()


# In[326]:


pcnt_time_hyperglycemia_daytime_manual = manual_cgm_daytime_df[manual_cgm_daytime_df['Sensor Glucose (mg/dL)'] > 180].groupby('Date').agg(lambda x: x.count() * 100 / 288)['Sensor Glucose (mg/dL)'].mean()

pcnt_time_hyperglycemia_critical_daytime_manual = manual_cgm_daytime_df[manual_cgm_daytime_df['Sensor Glucose (mg/dL)'] > 250].groupby('Date').agg(lambda x: x.count() * 100 / 288)['Sensor Glucose (mg/dL)'].mean()

pcnt_time_hyperglycemia_range_daytime_manual = manual_cgm_daytime_df[manual_cgm_daytime_df['Sensor Glucose (mg/dL)'] <= 180].groupby('Date').agg(lambda x: x.count() * 100 / 288)['Sensor Glucose (mg/dL)'].mean()

pcnt_time_hyperglycemia_range_secondary_daytime_manual = manual_cgm_daytime_df[manual_cgm_daytime_df['Sensor Glucose (mg/dL)'] <= 150].groupby('Date').agg(lambda x: x.count() * 100 / 288)['Sensor Glucose (mg/dL)'].mean()

pcnt_time_hyperglycemia_hypoglycemia_lvl1_daytime_manual = manual_cgm_daytime_df[manual_cgm_daytime_df['Sensor Glucose (mg/dL)'] < 70].groupby('Date').agg(lambda x: x.count() * 100 / 288)['Sensor Glucose (mg/dL)'].mean()

pcnt_time_hyperglycemia_hypoglycemia_lvl2_daytime_manual = manual_cgm_daytime_df[manual_cgm_daytime_df['Sensor Glucose (mg/dL)'] < 54].groupby('Date').agg(lambda x: x.count() * 100 / 288)['Sensor Glucose (mg/dL)'].mean()


# In[327]:


pcnt_time_hyperglycemia_overnight_manual = manual_cgm_overnight_df[manual_cgm_overnight_df['Sensor Glucose (mg/dL)'] > 180].groupby('Date').agg(lambda x: x.count() * 100 / 288)['Sensor Glucose (mg/dL)'].mean()

pcnt_time_hyperglycemia_critical_overnight_manual = manual_cgm_overnight_df[manual_cgm_overnight_df['Sensor Glucose (mg/dL)'] > 250].groupby('Date').agg(lambda x: x.count() * 100 / 288)['Sensor Glucose (mg/dL)'].mean()

pcnt_time_hyperglycemia_range_overnight_manual = manual_cgm_overnight_df[manual_cgm_overnight_df['Sensor Glucose (mg/dL)'] <= 180].groupby('Date').agg(lambda x: x.count() * 100 / 288)['Sensor Glucose (mg/dL)'].mean()

pcnt_time_hyperglycemia_range_secondary_overnight_manual = manual_cgm_overnight_df[manual_cgm_overnight_df['Sensor Glucose (mg/dL)'] <= 150].groupby('Date').agg(lambda x: x.count() * 100 / 288)['Sensor Glucose (mg/dL)'].mean()

pcnt_time_hyperglycemia_hypoglycemia_lvl1_overnight_manual = manual_cgm_overnight_df[manual_cgm_overnight_df['Sensor Glucose (mg/dL)'] < 70].groupby('Date').agg(lambda x: x.count() * 100 / 288)['Sensor Glucose (mg/dL)'].mean()

pcnt_time_hyperglycemia_hypoglycemia_lvl2_overnight_manual = manual_cgm_overnight_df[manual_cgm_overnight_df['Sensor Glucose (mg/dL)'] < 54].groupby('Date').agg(lambda x: x.count() * 100 / 288)['Sensor Glucose (mg/dL)'].mean()


# In[328]:


results_df = pd.DataFrame({
    'Overnight Percentage time in hyperglycemia (CGM > 180 mg/dL)': [pcnt_time_hyperglycemia_overnight_manual, pcnt_time_hyperglycemia_overnight_auto],
    'Overnight percentage of time in hyperglycemia critical (CGM > 250 mg/dL)': [pcnt_time_hyperglycemia_critical_overnight_manual, pcnt_time_hyperglycemia_critical_overnight_auto],
    'Overnight percentage time in range (CGM >= 70 mg/dL and CGM <= 180 mg/dL)': [pcnt_time_hyperglycemia_range_overnight_manual, pcnt_time_hyperglycemia_range_overnight_auto],
    'Overnight percentage time in range secondary (CGM >= 70 mg/dL and CGM <= 150 mg/dL)': [pcnt_time_hyperglycemia_range_secondary_overnight_manual, pcnt_time_hyperglycemia_range_secondary_overnight_auto],
    'Overnight percentage time in hypoglycemia level 1 (CGM < 70 mg/dL)': [pcnt_time_hyperglycemia_hypoglycemia_lvl1_overnight_manual, pcnt_time_hyperglycemia_hypoglycemia_lvl1_overnight_auto],
    'Overnight percentage time in hypoglycemia level 2 (CGM < 54 mg/dL)': [pcnt_time_hyperglycemia_hypoglycemia_lvl2_overnight_manual, np.nan_to_num(pcnt_time_hyperglycemia_hypoglycemia_lvl2_overnight_auto)],
    'Daytime Percentage time in hyperglycemia (CGM > 180 mg/dL)': [pcnt_time_hyperglycemia_daytime_manual, pcnt_time_hyperglycemia_daytime_auto],
    'Daytime percentage of time in hyperglycemia critical (CGM > 250 mg/dL)': [pcnt_time_hyperglycemia_critical_daytime_manual, pcnt_time_hyperglycemia_critical_daytime_auto],
    'Daytime percentage time in range (CGM >= 70 mg/dL and CGM <= 180 mg/dL)': [pcnt_time_hyperglycemia_range_daytime_manual, pcnt_time_hyperglycemia_range_daytime_auto],
    'Daytime percentage time in range secondary (CGM >= 70 mg/dL and CGM <= 150 mg/dL)': [pcnt_time_hyperglycemia_range_secondary_daytime_manual, pcnt_time_hyperglycemia_range_secondary_daytime_auto],
    'Daytime percentage time in hypoglycemia level 1 (CGM < 70 mg/dL)': [pcnt_time_hyperglycemia_hypoglycemia_lvl1_daytime_manual, pcnt_time_hyperglycemia_hypoglycemia_lvl1_daytime_auto],
    'Daytime percentage time in hypoglycemia level 2 (CGM < 54 mg/dL)': [pcnt_time_hyperglycemia_hypoglycemia_lvl2_daytime_manual, pcnt_time_hyperglycemia_hypoglycemia_lvl2_daytime_auto],
    'Whole Day Percentage time in hyperglycemia (CGM > 180 mg/dL)': [pcnt_time_hyperglycemia_wholeday_manual, pcnt_time_hyperglycemia_wholeday_auto],
    'Whole day percentage of time in hyperglycemia critical (CGM > 250 mg/dL)': [pcnt_time_hyperglycemia_critical_wholeday_manual, pcnt_time_hyperglycemia_critical_wholeday_auto],
    'Whole day percentage time in range (CGM >= 70 mg/dL and CGM <= 180 mg/dL)': [pcnt_time_hyperglycemia_range_wholeday_manual, pcnt_time_hyperglycemia_range_wholeday_auto],
    'Whole day percentage time in range secondary (CGM >= 70 mg/dL and CGM <= 150 mg/dL)': [pcnt_time_hyperglycemia_range_secondary_wholeday_manual, pcnt_time_hyperglycemia_range_secondary_wholeday_auto],
    'Whole day percentage time in hypoglycemia level 1 (CGM < 70 mg/dL)': [pcnt_time_hyperglycemia_hypoglycemia_lvl1_wholeday_manual, pcnt_time_hyperglycemia_hypoglycemia_lvl1_wholeday_auto],
    'Whole Day percentage time in hypoglycemia level 2 (CGM < 54 mg/dL)': [pcnt_time_hyperglycemia_hypoglycemia_lvl2_wholeday_manual, pcnt_time_hyperglycemia_hypoglycemia_lvl2_wholeday_auto]
}, index = ['Manual Mode', 'Auto Mode'])


# In[329]:


results_df.to_csv('./Results.csv')

