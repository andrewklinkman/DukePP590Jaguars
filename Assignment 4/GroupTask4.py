from __future__ import division
import pandas as pd
import numpy as np
import os
import statsmodels.api as sm

main_dir = '/Users/jseidenfeld/Documents/School/Classes/Big Data for Energy 590/Data for Exercises/'

## Change working directory (wd)

os.chdir(main_dir)
from logit_functions import *

# Import Data

df = pd.read_csv(main_dir+ 'task_4_kwh_w_dummies_wide.csv')
df = df.dropna()

## get tariffs -------
tariffs = [v for v in pd.unique(df['tariff']) if v !='E']
stimuli = [v for v in pd.unique(df['stimulus']) if v !='E'] 

tariffs.sort()
stimuli.sort()

##drop actual trial data -----------
drop = [v for v in df.columns if v.startswith("kwh_2010")]
df_pretrial = df.drop(drop, axis=1)

## RUN LOGIT
for i in tariffs:
    for j in stimuli:
            logit_results, df_logit = do_logit(df_pretrial, i, j, add_D=None, mc=False)
            
## RUN QUICK MEANS

grp = df_logit.groupby('tariff')

df_mean = grp.mean().transpose()
df_mean.C - df_mean.E

df_s = grp.std().transpose()
df_n = grp.count().transpose().mean()

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

#### Create necessary vars ------------------

df_merge['TP']=df_merge['trial']*df_merge['trt']


## Log kwh consumption var and YM var

df_merge['log_kwh'] = (df_merge['kwh'] + 1).apply(np.log)

# create month string `mo_str` that adds "0" to single digit integers
df_merge['mo_str'] = np.array(["0" + str(v) if v < 10 else str(v) for v in df_merge['month']])
# concatenate to make ym string values
df_merge['ym'] = df_merge['year'].apply(str) + "_" + df_merge['mo_str']

###################################
#########  Section III.4  #########
###################################

Josh_is_a_genius = df_merge['log_kwh']
P = df_merge['trial']
TP = df_merge['TP']
w = df_merge['w']
mu = pd.get_dummies(df_merge['ym'], prefix = 'ym').iloc[:, 1:-1]

X = pd.concat([TP, P, mu], axis=1)

ids = df_merge['ID']
y = demean(Josh_is_a_genius, ids)
X = demean(X, ids)

## WITHOUT WEIGHTS
fe_model = sm.OLS(y, X) # linearly prob model
fe_results = fe_model.fit() # get the fitted values
print(fe_results.summary()) # print pretty results (no results given lack of obs)

# WITH WEIGHTS
## apply weights to data
yw = y*w # weight each y
nms = X.columns.values # save column names
Xw = np.array([x*w for k, x in X.iteritems()]) # weight each X value
Xw = Xw.T # transpose (necessary as arrays create "row" vectors, not column)
Xw_jake_is_a_genius = pd.DataFrame(Xw, columns = nms) # update to dataframe; use original names

fe_w_model = sm.OLS(yw, Xw_jake_is_a_genius) # linearly prob model
fe_w_results = fe_w_model.fit() # get the fitted values
print(fe_w_results.summary()) # print pretty results (no results given lack of obs)

