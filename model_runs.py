class ModelRun():
    '''
    Describes experiments and their results considering a constant.
    Manages the expiriment.
    '''
    def __init__(self, df):
        self.df = df
        self.history = []

    def get_results(self, score_type):
        scores = []
        x_ticks = []
        trial_range = len(df.history)

        for trial in self.history:
            scores.append(trial['results'][score_type])
            x_ticks.append(trial['results']['model'])

        plt.figure(figsize=(12, 6))
        plt.plot(trial_range, scores, color='red', linestyle='dashed', marker='o',
                 markerfacecolor='blue', markersize=10)
        plt.title('{} Scores'.format(score_type))
        plt.xlabel('Trial')
        plt.ylabel(score_type)
        plt.xticks(np.arange(len(trial_range)), (x_ticks))
        plt.show()

        for trial_num, trial in  enumerate(self.history):



    def run_model(self, rd):
        attempt_count = 1
        for item in rd:
            attempt_count *= len(rd[item])
        print('Warning: attempting {} variations.'.format(str(attempt_count)))
        # x = input('Continue? (y)')


        trials = []

        target_df = self.df[rd['target']]
        if rd['target'] in rd['features']:
            features_df = self.df[rd['features']].drop(target_df, axis=1)
        else:
            features_df = self.df[rd['features']]

        attempt = Attempt(model=rd['model'],
                            features=features_df,
                            target=target_df,
                            scaler=rd['scaler'],
                            metrics=rd['metrics'],
                            modelargs=rd['kwargs'])

        rd['results'] = attempt.evaluate()
        self.attempts.append(rd)
        return None





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
