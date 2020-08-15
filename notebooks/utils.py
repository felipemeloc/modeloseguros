import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
trg = ['TARGET_FLAG', 'TARGET_AMT']
#Variables Numericas
con = ['AGE','BLUEBOOK','CAR_AGE','CLM_FREQ','HOMEKIDS','HOME_VAL','INCOME','MVR_PTS','OLDCLAIM','TIF','TRAVTIME','YOJ']
#Variables Cateoricas
cat = ['CAR_TYPE','CAR_USE','EDUCATION','JOB','KIDSDRIV','MSTATUS','PARENT1','RED_CAR','REVOKED','SEX','URBANICITY']
def get_dummies_cols(df,need):
    col = df.columns
    col_need=[]
    for i in col:
        for j in need:
            if j in i:
                col_need.append(i)
    return col_need
def get_df(df,cols,model):
    x = []
    for i in cols:
        if i in con:
            x.append(i)
        elif i in cat:
            cat_col = get_dummies_cols(df,[i])
            for j in cat_col:
                x.append(j)
    if model == 'reg':
        y = trg[1]
        df = df[df[trg[0]]==1]
        return df[[y]],df[x]
    if model == 'log':
        y = trg[0]
        return df[[y]],df[x]

def split_check_log(x,y,test_size):
    error = 1e18
    rd_seed = 0
    for i in range(500):
        rd = np.random.randint(1,500)
        x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=rd)
        p_test = 100*sum(y_test[trg[0]]) /len(y_test[trg[0]])
        p_train = 100*sum(y_train[trg[0]]) /len(y_train[trg[0]])
        dif = abs(p_test-p_train)
        if dif < error:
            rd_seed = rd
            error = dif
            print('{:3} {}'.format(rd,dif))
    return rd