from pandas import Series, DataFrame
import pandas as pd
import numpy as np


main_dir = "/Users/jseidenfeld/Documents/School/Classes/Big Data for Energy 590/Panda Repositories/LiveDemo" 
txt_file = "/Data/File1_small.txt"
main_dir + txt_file

## read_table(file, sep = " ") tells computer that file is space delimited

df = pd.read_table(main_dir + txt_file, sep = " ")
type(df)
list(df)
df[60:100]
df[df.kwh > 30]