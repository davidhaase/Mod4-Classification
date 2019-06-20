class ModelRun():
    '''
    Describes experiments and their results considering a constant.
    Manages the expiriment.
    '''
    def __init__(self, df):
        self.df = df
        self.history = []

    def get_results(self, score_type):
        import matplotlib.pyplot as plt
        scores = []
        trial_name = []
        trial_range = len(self.history)

        for trial in self.history:
            key, result = trial
            if score_type in result:
                scores.append(result[score_type])
                trial_name.append(key)

        # plt.figure(figsize=(12, 6))
        # plt.plot(trial_range, scores, color='red', linestyle='dashed', marker='o',
        #          markerfacecolor='blue', markersize=10)
        # plt.title('{} Scores'.format(score_type))
        # plt.xlabel('Trial')
        # plt.ylabel(score_type)
        # plt.show()

        for i in  trial_name:
            print('{}: {}\n'.format(str(trial_num), trial))



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
