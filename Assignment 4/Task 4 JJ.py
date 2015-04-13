#=========== WHAT JAKE AND JOSH COULDN'T DO / DIDN'T GiT TO
#Analysis from Section 1
#Section 3.3.i
#Section 3.4-7
#

from __future__ import division
import pandas as pd
import numpy as np
import os

main_dir = '/Users/Jake/Desktop/Duke/Energy and Big Data/task 4/'

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



## Log kwh consumption var and YM var

df_merge['log_kwh'] = (df_merge['kwh'] + 1).apply(np.log)

# create month string `mo_str` that adds "0" to single digit integers
df_merge['mo_str'] = np.array(["0" + str(v) if v < 10 else str(v) for v in df_merge['month']])
# concatenate to make ym string values
df_merge['ym'] = df_merge['year'].apply(str) + "_" + df_merge['mo_str']

