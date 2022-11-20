import pandas as pd 

# The 'toydata.csv' dataset (1000 samples, 6 features, 1 ground truth) has been generated using the linear equation 'y = 2*x1 + 5*x2 + 3*x3 + 4*x4 + x5 + 6*x6'
path = "C:/Users/P70077043/Documents/FL_Codes/HFL(LoR)/"

# The 'toydata.csv' has been divided into 'traindata.csv' (900, 7) and 'testdata.csv' (100, 7)
df_train = pd.read_csv(path+'traindata.csv')
df_test = pd.read_csv(path+'testdata.csv')

def whole_train():
    x = df_train.iloc[:,:-1].values
    y = df_train.iloc[:,-1:].values
    return x,y

def get_testdata():
    x = df_test.iloc[:,:-1].values
    y = df_test.iloc[:,-1:].values
    return x,y

# Considering that there are 3 clients we split the traindata into 3 paritions horizontally

def client1_data():
    df = df_train[:300]
    x = df.iloc[:,:-1].values
    y = df.iloc[:,-1:].values        #Local data of client 1
    return x,y


def client2_data():
    df = df_train[300:600]
    x = df.iloc[:,:-1].values
    y = df.iloc[:,-1:].values        #Local data of client 2
    return x,y

def client3_data():
    df = df_train[600:]
    x = df.iloc[:,:-1].values
    y = df.iloc[:,-1:].values       #Local data of client 3
    return x,y
