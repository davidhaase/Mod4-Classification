from model_runs import Attempt
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

iris = load_iris()

df = pd.DataFrame(data= np.c_[iris['data'], iris['target']],
                     columns= iris['feature_names'] + ['target'])



model=LogisticRegression
features=df.drop("target",axis=1)
target=df.target
resample=None
scaler=
metrics
**modelargs
