import pandas as pd
import numpy as np


from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression

import seaborn as sns
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', 300)

import warnings
warnings.filterwarnings('ignore')

class ModelRun():
    '''
    Describes experiments and their results considering a constant.
    Manages the expiriment.
    '''
    def __init__(self, df):
        self.df = df
        self.history = []


    def get_results(self, score_type):
        max_value = 0.00
        max_trial = None
        scores = []
        trial_name = []
        trial_range = len(self.history)
        for key, result_agg in self.history:
            if score_type in result_agg:
                score = result_agg[score_type]
                scores.append(score)
                trial_name.append(key)
                if score > max_value:
                    max_value = score
                    max_trial = key

        df_scores = pd.DataFrame(scores)

        df_scores.plot.line(figsize=(16,8))

        print('***\nWINNER for {}: {}'.format(score_type, str(max_value)))
        for k in key:
            print ('{}: {}'.format(k, key[k]))
        return scores


        # plt.figure(figsize=(12, 6))
        # plt.plot(trial_range, scores, color='red', linestyle='dashed', marker='o',
        #          markerfacecolor='blue', markersize=10)
        # plt.title('{} Scores'.format(score_type))
        # plt.xlabel('Trial')
        # plt.ylabel(score_type)
        # plt.show()

    def explore_data(self):
        self.df.shape

        sns.set(style="white")

        # Compute the correlation matrix
        corr = self.df.corr()

        # Generate a mask for the upper triangle
        mask = np.zeros_like(corr, dtype=np.bool)
        mask[np.triu_indices_from(mask)] = True

        # Set up the matplotlib figure
        f, ax = plt.subplots(figsize=(11, 9))

        # Generate a custom diverging colormap
        cmap = sns.diverging_palette(220, 10, as_cmap=True)

        # Draw the heatmap with the mask and correct aspect ratio
        sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
                    square=True, linewidths=.5, cbar_kws={"shrink": .5})

        plt.savefig('images/correlation_heatmap.png')

    def plot_pairs(self, cols=[]):
        if cols == []:
            cols = self.df.columns

        sns.set(style="white")
        sns.pairplot(self.df, vars=cols)
        plt.savefig('images/pairplots.png')

    def missing_data(self):
        #missing data
        total = self.df.isnull().sum().sort_values(ascending=False)
        percent = (self.df.isnull().sum()/self.df.isnull().count()).sort_values(ascending=False)
        missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
        return missing_data.head(20)

    def get_drop_list_corrs(self, threshold=0.95):

        # Create correlation matrix
        # Select upper triangle of correlation matrix
        corr_matrix = self.df.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
        return [column for column in upper.columns if any(upper[column] > threshold)]

    # def get_drop_list_f_test(self, k=10):
    #     selector = SelectKBest(f_regression, k)
    #     selector.fit(self.X_train, self.y_train)
    #     return self.X_train.columns[~selector.get_support()]

    def show_target_balance(self,target):
        sns.set(style="white")
        plt.figure(figsize = (10,5))
        sns.countplot(self.df[target], alpha =.80, palette= ['grey','lightgreen'])
        plt.title('Non-Defaults vs Defaults')
        plt.ylabel('# Defaults')
        plt.show()
        plt.savefig('images/balance.png')


    def run_model(self, rd):
        # attempt_count = 1
        # for item in rd:
        #     attempt_count *= len(rd[item])
        # print('Warning: attempting {} variations.'.format(str(attempt_count)))
        # x = input('Continue? (y)')



        for feat in rd['features']:
            for scaler in rd['scaler']:
                for model in rd['model']:
                    for kwargs in rd['kwargs']:
                        if kwargs['name'] == model.__name__:
                            kwargscopy = kwargs.copy()
                            del kwargscopy['name']
                            target_df = self.df[rd['target']]
                            if rd['target'] in feat:
                                feat.remove(rd['target'])

                            features_df = self.df[feat]



                            attempt = Attempt(model=model,
                                                features=features_df,
                                                target=target_df,
                                                scaler=scaler,
                                                metrics=rd['metrics'],
                                                modelargs=kwargscopy)
                            key = {'features':feat,'scaler':scaler.__name__,'model':model.__name__,'kwargs':kwargscopy}
                            self.history.append((key,attempt.evaluate()))






class Attempt():
    '''
    Contains the ClassifierModel (independent variable) and RunDetails (control/constant)
    '''
    def __init__(self,model,features,target,scaler,metrics,modelargs):
        self.sklearn_model = model #ClassifierModel
        self.scaler = scaler()
        self.x = features.values
        self.y = target.values
        self.modelargs = modelargs
        self.metrics = metrics
    def evaluate(self):
        metric_agg = {}
        for metric in self.metrics:
            metric_agg[metric.__name__] = 0
        runs = 0
        for x_train,x_test,y_train,y_test in split(self.x,self.y):
            runs += 1
            self.scaler.fit(x_train)
            x_train,x_test = self.scaler.transform(x_train),self.scaler.transform(x_test)
            model = self.sklearn_model(**self.modelargs)
            model.fit(x_train,y_train)
            preds = model.predict(x_test)
            for metric in self.metrics:
                metric_agg[metric.__name__] += metric(preds,y_test)

        for metric in self.metrics:
            metric_agg[metric.__name__] /= runs
        return metric_agg

from sklearn.model_selection import KFold
def split(x,y):
    kfold = KFold(n_splits=5)
    splits = []
    for train_ind,test_ind in kfold.split(x):
        splits.append([x[train_ind],x[test_ind],y[train_ind],y[test_ind]])
    return splits
