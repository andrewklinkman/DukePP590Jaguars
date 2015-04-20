from __future__ import division
from pandas import Series, DataFrame
import pandas as pd
import numpy as np
import statsmodels.api as sm
import os

main_dir = "/Users/andrewklinkman/Dropbox/Duke/School Stuff 14-15/Spring 2015/PUBPOL 590/Data/"

alloc_file = "allocation_subsamp.csv"

df_alloc = pd.read_csv(os.path.join(main_dir, alloc_file))

control = df_alloc.ID[df_alloc['tariff']=='E']
A1 = df_alloc.ID[(df_alloc['tariff'] == 'A') & (df_alloc['stimulus'] == '1')]
A3= df_alloc.ID[(df_alloc['tariff'] == 'A') & (df_alloc['stimulus']=='3')]
B1= df_alloc.ID[(df_alloc['tariff'] == 'B') & (df_alloc['stimulus']=='1')]
B3= df_alloc.ID[(df_alloc['tariff'] == 'B') & (df_alloc['stimulus']=='3')]

#set seed
np.random.seed(seed=1789)

#pick randoms out
control_redux = DataFrame(np.random.choice(control, 300, replace = False), columns = ['ID'])
A1_redux = DataFrame(np.random.choice(A1, 150, replace = False), columns = ['ID'])
A3_redux = DataFrame(np.random.choice(A3, 150, replace = False), columns = ['ID'])
B1_redux = DataFrame(np.random.choice(B1, 50, replace = False), columns = ['ID'])
B3_redux = DataFrame(np.random.choice(B3, 50, replace = False), columns = ['ID'])

#give group names (for aggregation later) Not the most elegant solution, but it works
control_redux['Group'] = 'Control'
A1_redux['Group'] = 'A1'
A3_redux['Group'] = 'A3'
B1_redux['Group'] = 'B1'
B3_redux['Group'] = 'B3'

df_samp = pd.concat([control_redux,A1_redux,A3_redux, B1_redux, B3_redux], ignore_index = True)

#pull in consump data
consump_file = "kwh_redux_pretrail.csv"

df_cons = pd.read_csv(os.path.join(main_dir, consump_file), parse_dates = [2])

df = pd.merge(df_samp, df_cons)

len(df_cons)-len(df) #shows that we dropped a bunch by merging
del df_cons

#add columns for month/year
df['year'] = df['date'].apply(lambda x: x.year)
df['month'] = df['date'].apply(lambda x: x.month)

#aggregation
grp1 = df.groupby(['Group','ID','year', 'month'])
df=grp1['kwh'].sum().reset_index()

#new cols before pivoting
df['mo_str'] = ['0' + str(v) if v < 10 else str(v) for v in df['month']]
df['kwh_ym'] = 'kwh_' + df.year.apply(str) + '_' + df.mo_str

#now pivot
df_wide = df.pivot('ID', 'kwh_ym', 'kwh')
df_wide.reset_index(inplace = True)
df_wide.columns.name = None

#merge id file back on
df_alloc = pd.merge(df_alloc, df_samp) #first merge with df_samp so that we retain the group names
df_wide = pd.merge(df_wide, df_alloc)

#dummy vars for groups/clean up df
df_wide = pd.get_dummies(df_wide, columns = ['Group'])
df_wide.drop(['code', 'tariff', 'stimulus'], axis = 1, inplace = True)

#SET UP DATA FOR LOGIT
kwh_cols=[v for v in df_wide.columns.values if v.startswith('kwh')] #list of all vars you want from regression

##SET UP Y, X and RUN LOGIT
#group A1
df_wideA1 = df_wide[(df_wide.Group_A1 == 1) | (df_wide.Group_Control == 1)]
yA1 = df_wideA1.Group_A1
XA1 = df_wideA1[kwh_cols]
XA1 = sm.add_constant(XA1)
#logit
logit_model = sm.Logit(yA1, XA1)
logit_results = logit_model.fit()
print(logit_results.summary())

#group A3
df_wideA3 = df_wide[(df_wide.Group_A3 == 1) | (df_wide.Group_Control == 1)]
yA3 = df_wideA3.Group_A3
XA3 = df_wideA3[kwh_cols]
XA3 = sm.add_constant(XA3)
#logit
logit_model = sm.Logit(yA3, XA3)
logit_results = logit_model.fit()
print(logit_results.summary())

#group B1
df_wideB1 = df_wide[(df_wide.Group_B1 == 1) | (df_wide.Group_Control == 1)]
yB1 = df_wideB1.Group_B1
XB1 = df_wideB1[kwh_cols]
XB1 = sm.add_constant(XB1)
#logit
logit_model = sm.Logit(yB1, XB1)
logit_results = logit_model.fit()
print(logit_results.summary())

#group B3
df_wideB3 = df_wide[(df_wide.Group_B3 == 1) | (df_wide.Group_Control == 1)]
yB3 = df_wideB3.Group_B3
XB3 = df_wideB3[kwh_cols]
XB3 = sm.add_constant(XB3)
#logit
logit_model = sm.Logit(yB3, XB3)
logit_results = logit_model.fit()
print(logit_results.summary())



