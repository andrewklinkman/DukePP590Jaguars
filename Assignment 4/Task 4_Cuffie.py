from __future__ import division
import pandas as pd
import numpy as np
import os

main_dir = "/Users/louiswinkler/Desktop/GitHub/Task-4/"

# change working directory (wd)
os.chdir(main_dir)
from logit_functions import *

df = pd.read_csv(main_dir + 'task_4_kwh_w_dummies_wide.csv')
df = df.dropna(axis=0, how ='any')

################################################
######Section I  ########################
################################################

#get traiffs
tariffs = [v for v in pd.unique(df['tariff']) if v != 'E']
stimuli = [v for v in pd.unique(df['stimulus']) if v != 'E']
tariffs.sort()
stimuli.sort()

# run logit
drop = [v for v in df.columns if v.startswith('kwh_2010')]
df_pretrial = df.drop(drop, axis=1)

for i in tariffs: 
    for j in stimuli:
        # dummy vars must start with "D_" and consumption vars with "kwh_"; specific to this function; 
        #mc stands for multi colinearity
        logit_results, df_logit = do_logit(df_pretrial, i, j, add_D=None, mc= False)

#quick means comparison with a T-Test by hand--------------

#create means
df_mean = df_logit.groupby('tariff').mean().transpose()

df_mean.C - df_mean.E

#do a t-test by hand
df_s = df_logit.groupby('tariff').std().transpose()
df_n = df_logit.groupby('tariff').count().transpose().mean()
top = df_mean['C'] - df_mean['E']
bottom = np.sqrt(df_s['C']**2/df_n['C'] + df_s['E']**2/df_n['E'])
tstats = top/bottom
sig = tstats[np.abs(tstats) > 2]
sig.name = 't-stats'

################################################
######Section II  ########################
################################################

df_logit['p_val'] = logit_results.predict()
p_hat = df_logit['p_val']
df_logit['trt'] = 0 + (df_logit['tariff'] == 'C')
top = df_logit['trt']

df_logit['w'] = np.sqrt((top/p_hat)+(1-top)/(1-p_hat))
df_w = df_logit[['ID', 'trt', 'w']]

################################################
######Section III  ########################
################################################

#Set up
from fe_functions import *

#Import and Merge
df_task4 = pd.read_csv(main_dir+ 'task_4_kwh_long.csv')
df_merge = pd.merge(df_task4, df_w)

#### 3.3i, Create necessary vars ------------------

treat-trial = 
# --->need to make treatment variable with log_kwh and ids?

## 3.3ii Log kwh consumption var and YM var

df_merge['log_kwh'] = (df_merge['kwh'] + 1).apply(np.log)

# 3.3iii create month string `mo_str` that adds "0" to single digit integers
df_merge['mo_str'] = np.array(["0" + str(v) if v < 10 else str(v) for v in df_merge['month']])
# concatenate to make ym string values
df_merge['ym'] = df_merge['year'].apply(str) + "_" + df_merge['mo_str']

# 3.4 set up regression variables

y = ['log_kwh']
T = ['0', '1']
TP = df[treat-trial']
w = df['w']
mu = pd.get_dummies(df['ym'], prefix = 'ym').iloc[:, 1:-1]

X = pd.concat([TP, P, mu], axis=1)

# 3.5
ids = df['ID']

y = demean(y, ids)

# 3.6 Run FE with and without weights 

## WITHOUT WEIGHTS
fe_model = sm.OLS(y, X) # linearly prob model
fe_results = fe_model.fit() # get the fitted values
print(fe_results.summary()) # print pretty results (no results given lack of obs)

# WITH WEIGHTS
## apply weights to data
y = y*w # weight each y
nms = X.columns.values # save column names
X = np.array([x*w for k, x in X.iteritems()]) # weight each X value
X = X.T # transpose (necessary as arrays create "row" vectors, not column)
X = DataFrame(X, columns = nms) # update to dataframe; use original names

fe_w_model = sm.OLS(y, X) # linearly prob model
fe_w_results = fe_w_model.fit() # get the fitted values
print(fe_w_results.summary()) # print pretty results (no results given lack of obs)



