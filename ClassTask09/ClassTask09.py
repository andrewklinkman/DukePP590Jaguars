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
A2= df_alloc.ID[(df_alloc['tariff'] == 'A') & (df_alloc['stimulus']=='2')]
B1= df_alloc.ID[(df_alloc['tariff'] == 'B') & (df_alloc['stimulus']=='1')]
B2= df_alloc.ID[(df_alloc['tariff'] == 'B') & (df_alloc['stimulus']=='2')]