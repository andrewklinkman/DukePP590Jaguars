from scipy.stats import ttest_ind
from scipy.special import stdtr
from __future__ import division
from pandas import Series, DataFrame
import pandas as pd
import numpy as np
import os
import time
from datetime import datetime



#IMPORT AND CLEAN CER DATA----------------------------------------------------------------------------
main_dir = "/Users/andrewklinkman/Dropbox/Duke/School Stuff 14-15/Spring 2015/PUBPOL 590/CER Stuff/raw/" #mac
#main_dir = "C:/Users/ask46/Dropbox/Duke/School Stuff 14-15/Spring 2015/PUBPOL 590/CER Stuff/raw/" #windows
csv_file = "SME and Residential allocations.csv"
time_file = "timeseries_correction_dst.csv"


#IMPORT-------------------------------------------------------------------------------------
#build file list
all_files = [os.path.join(main_dir, vdh) for vdh in os.listdir(main_dir) if vdh.startswith("File")]

#pull in definitions file (limit to first 4 cols)
df_def = pd.read_csv(os.path.join(main_dir, csv_file), usecols=[0,1,2,3],na_values=[''])

#pull in timeseries helper file
df_time = pd.read_csv(os.path.join(main_dir, time_file), parse_dates = [1])
df_time.drop('Unnamed: 0', axis = 1, inplace = True)

df_time['timecode_cer'] = df_time['day_cer']*100 + df_time['hour_cer']
df_time['datetime'] = pd.to_datetime(df_time['date']) #get the date in datetime, so we can sort later
df_time['monthyear'] = df_time['datetime'].apply(lambda x: datetime(x.year, x.month, 1, 0, 0)) #get just the first day of the month for each date


#rename columns to match existing DF
df_def.columns = ['meterid', 'Code', 'ResTariff', 'ResStim']

#limit def data to only res customers --> drops total to 4225 rows
df_def = df_def[df_def.Code == 1].copy()
    
#limit def data to the other criteria --> drops total to 1210 rows
#either control group or (Tariff = a and stimulus = 1)
df_def = df_def[(df_def['ResStim']=='E') | ((df_def['ResTariff']=='A') & (df_def['ResStim']=='1'))].copy()

#turn all files in list into dataframes
#and concat all of them
#AND merge to def file AND then to time file...all at once
df_combo = pd.merge(df_time, pd.merge(pd.concat([pd.read_table(v, sep = ' ', names = ['meterid', 'timecode_cer', 'consump']) for v in all_files],
            ignore_index = True), df_def, on = 'meterid'), on = 'timecode_cer', how = 'inner')

del df_def
del df_time

#CLEAN-----------------------------------------------------------------------------------

#check datatypes
df_combo.dtypes 

#all good, no random periods or dashes. 

#look for null values
rows1 = df_combo['meterid'].isnull()
rows2 = df_combo['timecode_cer'].isnull()
rows3 = df_combo['consump'].isnull()

#len(df_combo[rows1])
#len(df_combo[rows2])
#len(df_combo[rows3])
#
#rows1 = None
#rows2 = None
#rows3 = None
##doesn't look like we have any


#GROUP-----------------------------------------------------------------------------------
###group by day-----------------------------
grp1 = df_combo.groupby(['ResTariff', 'datetime', 'meterid'])

agg1 = grp1['consump'].sum()
agg1 = agg1.reset_index()

grp2 = agg1.groupby(['ResTariff', 'datetime'])

###group by month---------------------------
grp3 = df_combo.groupby(['ResTariff', 'monthyear', 'meterid'])

agg2 = grp3['consump'].sum()
agg2 = agg2.reset_index()

grp4 = agg2.groupby(['ResTariff', 'monthyear'])


#T-TESTS----------------------------------------------------------------------------------

###build dicts of all days and the individual, meter-level sums
trt_day = { k[1]: agg1.consump[v].values for k, v in grp2.groups.iteritems() if k[0]=='A'}
ctrl_day = { k[1]: agg1.consump[v].values for k, v in grp2.groups.iteritems() if k[0]=='E'}
keys_day = trt_day.keys()

###build dicts of all months and the individual, meter-level sums
trt_mth = { k[1]: agg2.consump[v].values for k, v in grp4.groups.iteritems() if k[0]=='A'}
ctrl_mth = { k[1]: agg2.consump[v].values for k, v in grp4.groups.iteritems() if k[0]=='E'}
keys_mth = trt_mth.keys()

#memory management - delete unused variables
del df_combo
del grp1
del grp2
del agg1
del grp3
del grp4
del agg2

#run the t-test for days
tstats_day = DataFrame([(k, np.abs(ttest_ind(trt_day[k], ctrl_day[k], equal_var = False)[0])) for k in keys_day],
                    columns = ['date', 'tstat'])

pvals_day = DataFrame([(k, np.abs(ttest_ind(trt_day[k], ctrl_day[k], equal_var = False)[1])) for k in keys_day],
                    columns = ['date', 'pval'])
                    
#run the t-test for months
tstats_mth = DataFrame([(k, np.abs(ttest_ind(trt_mth[k], ctrl_mth[k], equal_var = False)[0])) for k in keys_mth],
                    columns = ['date', 'tstat'])

pvals_mth = DataFrame([(k, np.abs(ttest_ind(trt_mth[k], ctrl_mth[k], equal_var = False)[1])) for k in keys_mth],
                    columns = ['date', 'pval'])

#mem mgmt
del trt_day               
del ctrl_day
del trt_mth                
del ctrl_mth
                   
#put results into dataframe - nice and neat 
tp_day = pd.merge(tstats_day, pvals_day)
tp_day = tp_day.sort('date')
tp_day = tp_day.reset_index(drop = True)

tp_mth = pd.merge(tstats_mth, pvals_mth)
tp_mth = tp_mth.sort('date')
tp_mth = tp_mth.reset_index(drop = True)


###GRAPHZZZZZZZZZZ###################################

