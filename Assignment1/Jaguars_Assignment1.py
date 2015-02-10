from pandas import Series, DataFrame
import pandas as pd
import numpy as np
import os
import datetime as dt

#IMPORT AND CLEAN CER DATA
main_dir = "C:/Users/ask46/Dropbox/Duke/School Stuff 14-15/Spring 2015/PUBPOL 590/CER Stuff/raw/"
csv_file = "SME and Residential allocations.csv"

#build file list
all_files = [os.path.join(main_dir, v) for v in os.listdir(main_dir) if v.startswith("File")]

#turn all files in list into dataframes
#and build list out of all of them
all_dfs = [pd.read_table(v, sep = '\s', names = ['meterid', 'timedate', 'consump']) for v in all_files]

#all_dfs[1]
#df1 = pd.read_table(all_files, sep = '\s', names = ['meterid', 'timedate', 'consump'])

#merge all the existing dataframes from the text files
df1 = pd.concat(all_dfs, ignore_index = True)
all_dfs = None #clear up some space in memory

#pull in definitions file
df_def = pd.read_csv(os.path.join(main_dir, csv_file), na_values=[''])

#drop unused columns
df_def = df_def.drop(['Unnamed: 5','Unnamed: 6','Unnamed: 7','Unnamed: 8','Unnamed: 9'] ,1)

#rename columns to match existing DF
df_def.columns = ['meterid', 'Code', 'Residential - Tariff allocation', 'Residential - stimulus allocation','SME allocation']

#MERGE DEFINITIONS FILE TO DATA
df_combo = pd.merge(df1, df_def, on='meterid')
df1 = None #memory mgmt
df_def = None #memory mgmt

#drop all 'study drop-outs' where code = 3 (i.e. participant didn't complete study)
df_combo = df_combo[df_combo.Code != 3]

#verify that all Code 3s have been dropped
sum(df_combo.Code[df_combo.Code==3])

#check for duplicated rows on subset of meterid and timedate
sum(df_combo.duplicated(subset = ['meterid', 'timedate']))
#True = 1, False = 0, sum of array = 0, so no duplicates!


#check datatypes
df_combo.dtypes #all good, no random periods or dashes. 


#look for null values
rows1 = df_combo['meterid'].isnull()
rows2 = df_combo['timedate'].isnull()
rows3 = df_combo['consump'].isnull()

df_combo[rows1]
df_combo[rows2]
df_combo[rows3]
#doesn't look like we have any

#---------------------------------------------------------------------------
##convert dates
#base_date = dt.datetime.strptime("2009-01-01", "%Y-%m-%d")
#
##get first three chars of time field (days)
#temp1 = str(df1[11:12]['timedate'])
#intTemp1=int(temp1[5:8])
#
##get last two chars of time field (half hour increments)
#intTemp2=int(temp1[9:11])
#
#actual_date = base_date + dt.timedelta(days=intTemp1) + dt.timedelta(minutes = intTemp2*30)
