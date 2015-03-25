from __future__ import division
from pandas import Series, DataFrame
import pandas as pd
import numpy as np
import statsmodels.api as sm
import os

main_dir = "/Users/andrewklinkman/Dropbox/Duke/School Stuff 14-15/Spring 2015/PUBPOL 590/Data/"

alloc_file = "allocation_subsamp.csv"

df_alloc = pd.read_csv(os.path.join(main_dir, alloc_file))

treat = df_alloc.ID[df_alloc['tariff']=='E']
A1 = df_alloc.ID[(df_alloc['tariff'] == 'A') & (df_alloc['stimulus'] == '1')]
A3= df_alloc.ID[(df_alloc['tariff'] == 'A') & (df_alloc['stimulus']=='3')]
B1= df_alloc.ID[(df_alloc['tariff'] == 'B') & (df_alloc['stimulus']=='1')]
B3= df_alloc.ID[(df_alloc['tariff'] == 'B') & (df_alloc['stimulus']=='3')]

#set seed
np.random.seed(seed=1789)

#pick randoms out
treat_redux = np.random.choice(treat, 300, replace = False)
A1_redux = np.random.choice(A1, 150, replace = False)
A3_redux = np.random.choice(A3, 150, replace = False)
B1_redux = np.random.choice(B1, 50, replace = False)
B3_redux = np.random.choice(B3, 50, replace = False)

df_samp = DataFrame(np.concatenate([A1_redux,A3_redux, B1_redux, B3_redux, treat_redux], axis = 0))

#pull in consump data


