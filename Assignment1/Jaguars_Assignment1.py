from __future__ import division
from pandas import Series, DataFrame
import pandas as pd
import numpy as np
import os


#IMPORT AND CLEAN CER DATA

#this dir is for windows computers/cluster
main_dir = "C:/Users/ask46/Dropbox/Duke/School Stuff 14-15/Spring 2015/PUBPOL 590/CER Stuff/raw/"

#this dir is for my mac
#main_dir = "/Users/andrewklinkman/Dropbox/Duke/School Stuff 14-15/Spring 2015/PUBPOL 590/CER Stuff/raw/"
csv_file = "SME and Residential allocations.csv"

#build file list
all_files = [os.path.join(main_dir, v) for v in os.listdir(main_dir) if v.startswith("File")]

#turn all files in list into dataframes
#and build list out of all of them
all_dfs = [pd.read_table(v, sep = ' ', names = ['meterid', 'timedate', 'consump']) for v in all_files]

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
df_combo = df_combo[df_combo.Code != 3].copy()

#verify that all Code 3s have been dropped
sum(df_combo.Code[df_combo.Code==3])

#DROP ALL DST DATES
#make df_combo equal to itself, excluding the three DST days
df_combo = df_combo[
    ((df_combo.timedate-(df_combo.timedate % 100))/100 != [298]) &
    ((df_combo.timedate-(df_combo.timedate % 100))/100 != [452]) &
    ((df_combo.timedate-(df_combo.timedate % 100))/100 != [669]) 
].copy()

#check to make sure they dropped
df_combo[df_combo.timedate==29850]

#check for duplicated rows on subset of meterid and timedate
sum(df_combo.duplicated(subset = ['meterid', 'timedate']))
#True = 1, False = 0, sum of array = 0, so no duplicates!
#NOTE: this takes FOREVER...save this command for the cluster

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


