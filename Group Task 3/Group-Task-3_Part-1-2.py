## group task 3 practice
from __future__ import division
from pandas import Series, DataFrame
import pandas as pd
import numpy as np
import statsmodels.api as sm

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

# DEFINE FUNCTIONS -----------------
def ques_recode(srvy):

    DF = srvy.copy()
    import re
    q = re.compile('Question ([0-9]+):.*')
    cols = [unicode(v, errors ='ignore') for v in DF.columns.values]
    mtch = []
    for v in cols:
        mtch.extend(q.findall(v))

    df_qs = Series(mtch, name = 'q').reset_index() # get the index as a variable. basically a column index
    n = df_qs.groupby(['q'])['q'].count() # find counts of variable types
    n = n.reset_index(name = 'n') # reset the index, name counts 'n'
    df_qs = pd.merge(df_qs, n) # merge the counts to df_qs
    df_qs['index'] = df_qs['index'] + 1 # shift index forward 1 to line up with DF columns (we ommited 'ID')
    df_qs['subq'] = df_qs.groupby(['q'])['q'].cumcount() + 1
    df_qs['subq'] = df_qs['subq'].apply(str)
    df_qs.ix[df_qs.n == 1, ['subq']] = '' # make empty string
    df_qs['Ques'] = df_qs['q']
    df_qs.ix[df_qs.n != 1, ['Ques']] = df_qs['Ques'] + '.' + df_qs['subq']

    DF.columns = ['ID'] + df_qs.Ques.values.tolist()

    return df_qs, DF

def ques_list(srvy):

    df_qs, DF = ques_recode(srvy)
    Qs = DataFrame(zip(DF.columns, srvy.columns), columns = [ "recoded", "desc"])[1:]
    return Qs

# df = dataframe of survey, sel = list of question numbers you want to extract free of DVT
def dvt(srvy, sel):

    """Function to select questions then remove extra dummy column (avoids dummy variable trap DVT)"""

    df_qs, DF = ques_recode(srvy)

    sel = [str(v) for v in sel]
    nms = DF.columns

    # extract selected columns
    indx = []
    for v in sel:
         l = df_qs.ix[df_qs['Ques'] == v, ['index']].values.tolist()
         if(len(l) == 0):
            print (bcolors.FAIL + bcolors.UNDERLINE +
            "\n\nERROR: Question %s not found. Please check CER documentation"
            " and choose a different question.\n" + bcolors.ENDC) % v
         indx =  indx + [i for sublist in l for i in sublist]

    # Exclude NAs Rows
    DF = DF.dropna(axis=0, how='any', subset=[nms[indx]])

    # get IDs
    dum = DF[['ID']]
    # get dummy matrix
    for i in indx:
        # drop the first dummy to avoid dvt
        temp = pd.get_dummies(DF[nms[i]], columns = [i], prefix = 'D_' + nms[i]).iloc[:, 1:]
        dum = pd.concat([dum, temp], axis = 1)
        # print dum

        # test for multicollineary

    return dum

def rm_perf_sep(y, X):

    dep = y.copy()
    indep = X.copy()
    yx = pd.concat([dep, indep], axis = 1)
    grp = yx.groupby(dep)

    nm_y = dep.name
    nm_dum = np.array([v for v in indep.columns if v.startswith('D_')])

    DFs = [yx.ix[v,:] for k, v in grp.groups.iteritems()]
    perf_sep0 = np.ndarray((2, indep[nm_dum].shape[1]),
        buffer = np.array([np.linalg.norm(DF[nm_y].values.astype(bool) - v.values) for DF in DFs for k, v in DF[nm_dum].iteritems()]))
    perf_sep1 = np.ndarray((2, indep[nm_dum].shape[1]),
        buffer = np.array([np.linalg.norm(~DF[nm_y].values.astype(bool) - v.values) for DF in DFs for k, v in DF[nm_dum].iteritems()]))

    check = np.vstack([perf_sep0, perf_sep1])==0.
    indx = np.where(check)[1] if np.any(check) else np.array([])

    if indx.size > 0:
        keep = np.all(np.array([indep.columns.values != i for i in nm_dum[indx]]), axis=0)
        nms = [i.encode('utf-8') for i in nm_dum[indx]]
        print (bcolors.FAIL + bcolors.UNDERLINE +
        "\nPerfect Separation produced by %s. Removed.\n" + bcolors.ENDC) % nms

        # return matrix with perfect predictor colums removed and obs where true
        indep1 = indep[np.all(indep[nm_dum[indx]]!=1, axis=1)].ix[:, keep]
        dep1 = dep[np.all(indep[nm_dum[indx]]!=1, axis=1)]
        return dep1, indep1
    else:
        return dep, indep


def rm_vif(X):

    import statsmodels.stats.outliers_influence as smso
    loop=True
    indep = X.copy()
    # print indep.shape
    while loop:
        vifs = np.array([smso.variance_inflation_factor(indep.values, i) for i in xrange(indep.shape[1])])
        max_vif = vifs[1:].max()
        # print max_vif, vifs.mean()
        if max_vif > 30 and vifs.mean() > 10:
            where_vif = vifs[1:].argmax() + 1
            keep = np.arange(indep.shape[1]) != where_vif
            nms = indep.columns.values[where_vif].encode('utf-8') # only ever length 1, so convert unicode
            print (bcolors.FAIL + bcolors.UNDERLINE +
            "\n%s removed due to multicollinearity.\n" + bcolors.ENDC) % nms
            indep = indep.ix[:, keep]
        else:
            loop=False
    # print indep.shape

    return indep


def do_logit(df, tar, stim, D = None):

    DF = df.copy()
    if D is not None:
        DF = pd.merge(DF, D, on = 'ID')
        kwh_cols = [v for v in DF.columns.values if v.startswith('kwh')]
        dum_cols = [v for v in D.columns.values if v.startswith('D_')]
        cols = kwh_cols + dum_cols
    else:
        kwh_cols = [v for v in DF.columns.values if v.startswith('kwh')]
        cols = kwh_cols

    # DF.to_csv("/Users/dnoriega/Desktop/" + "test.csv", index = False)
    # set up y and X
    indx = (DF.tariff == 'E') | ((DF.tariff == tar) & (DF.stimulus == stim))
    df1 = DF.ix[indx, :].copy() # `:` denotes ALL columns; use copy to create a NEW frame
    df1['T'] = 0 + (df1['tariff'] != 'E') # stays zero unless NOT of part of control
    # print df1

    y = df1['T']
    X = df1[cols] # extend list of kwh names
    X = sm.add_constant(X)

    msg = ("\n\n\n\n\n-----------------------------------------------------------------\n"
    "LOGIT where Treatment is Tariff = %s, Stimulus = %s"
    "\n-----------------------------------------------------------------\n") % (tar, stim)
    print msg

    print (bcolors.FAIL +
        "\n\n-----------------------------------------------------------------" + bcolors.ENDC)

    y, X = rm_perf_sep(y, X) # remove perfect predictors
    X = rm_vif(X) # remove multicollinear vars

    print (bcolors.FAIL +
        "-----------------------------------------------------------------\n\n\n" + bcolors.ENDC)

    ## RUN LOGIT
    logit_model = sm.Logit(y, X) # linearly prob model
    logit_results = logit_model.fit(maxiter=10000, method='newton') # get the fitted values
    print logit_results.summary() # print pretty results (no results given lack of obs)



#####################################################################
#                           SECTION 1                               #
#####################################################################


main_dir = "/Users/louiswinkler/Desktop/data/Class-14-Task/"

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
treat_redux = DataFrame(np.random.choice(treat, 300, replace = False), columns = ['ID'])
A1_redux = DataFrame(np.random.choice(A1, 150, replace = False), columns = ['ID'])
A3_redux = DataFrame(np.random.choice(A3, 150, replace = False), columns = ['ID'])
B1_redux = DataFrame(np.random.choice(B1, 50, replace = False), columns = ['ID'])
B3_redux = DataFrame(np.random.choice(B3, 50, replace = False), columns = ['ID'])

#give group names (for aggregation later) Not the most elegant solution, but it works
treat_redux['Group'] = 'Control'
A1_redux['Group'] = 'A1'
A3_redux['Group'] = 'A3'
B1_redux['Group'] = 'B1'
B3_redux['Group'] = 'B3'

df_samp = pd.concat([A1_redux,A3_redux, B1_redux, B3_redux, treat_redux], ignore_index = True)

#df_samp = DataFrame(np.concatenate([A1_redux,A3_redux, B1_redux, B3_redux, treat_redux], axis = 0), columns = ['ID'])

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



#####################################################################
#                           SECTION 2                               #
#####################################################################

main_dir = "/Users/louiswinkler/Desktop/data/Class-14-Task/"

nas = ['', ' ', 'NA'] # set NA values so that we dont end up with numbers and text
srvy = pd.read_csv(main_dir + 'Smart meters Residential pre-trial survey data.csv', na_values = nas)
df = pd.read_csv(main_dir + 'data_section2.csv')

# list of questions
qs = ques_list(srvy)


# get dummies
sel = [200, 5414, 405]
dummies = dvt(srvy, sel)

# run logit, optional dummies
tariffs = [v for v in pd.unique(df['tariff']) if v != 'E']
stimuli = [v for v in pd.unique(df['stimulus']) if v != 'E']
tariffs.sort() # make sure the order correct with .sort()
stimuli.sort()

for i in tariffs:
    for j in stimuli:
        do_logit(df, i, j, D = dummies)







