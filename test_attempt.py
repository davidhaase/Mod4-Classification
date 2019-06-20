from model_runs import Attempt
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score

iris = load_iris()

df = pd.DataFrame(data= np.c_[iris['data'], iris['target']],
                     columns= iris['feature_names'] + ['target'])

df.drop(df[df.target==0].index,inplace=True)

model=LogisticRegression
features=df.drop("target",axis=1)
target=df.target
resample=None
scaler=StandardScaler
metrics=[f1_score]
modelargs={'penalty':'l1'}
print(Attempt(model=LogisticRegression,features=df.drop("target",axis=1),target=df.target,scaler=StandardScaler,metrics=[f1_score],modelargs={'penalty':'l1'}).evaluate())
