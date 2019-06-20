class RunDetail():
    '''
    All meta features as lists; if you want to keep constant, give only one choice
    '''
    def __init__(self, target, features, model, split, resample, scaler=None, kwargs={}):
        self.target = target
        self.split = split
        self.model = model
        self.scaler = scaler
        self.features = features
        self.resample = resample
        self.kwargs = kwargs
        self.metrics = None

class ModelRun():
    '''
    Describes experiments and their results considering a constant.
    Manages the expiriment.
    '''
    def __init__(self, df):
        self.df = df
        self.attempts = []

    def run_model(self, rd):
        attempt_count = 0
        # for item in rd:
        #     attempt_count += len(rd[item])
        # print(str(len(rd)), str(attempt_count))

        target_df = self.df[rd['target']]
        if rd['target'] in rd['features']:
            features_df = self.df[rd['features']].drop(target_df, axis=1)
        else:
            features_df = self.df[rd['features']]

        attempt = Attempt(rd['model'],features_df,target_df,rd['scaler'],rd['metrics'], rd['kwargs'])
        rd['metrics'] = attempt.evaluate()
        self.attempts.append(rd)
        return target_df, features_df


class Attempt():
    '''
    Contains the ClassifierModel (independent variable) and RunDetails (control/constant)
    '''
    def __init__(self,model,features,target,scaler,metrics,**modelargs):
        self.sklearn_model = model #ClassifierModel
        self.scaler = scaler()
        self.x = features.values
        self.y = target.values
        self.modelargs = modelargs
        self.metrics = metrics
    def evaluate(self):
        metric_agg = {}
        for metric in self.metrics:
            metric_agg[metric.__name__()] = 0
        runs = 0
        for x_train,x_test,y_train,y_test in split(self.x,self.y):
            runs += 1
            self.scaler.fit(x_train)
            x_train,x_test = self.scaler.transform(x_train),self.scaler.transform(x_test)
            model = self.sklearn_model(self.modelargs)
            model.fit(x_train,y_train)
            preds = model.predict(x_test)
            for metric in self.metrics:
                metric_agg[metric.__name__()] += metric(preds,y_test)

        for metric in self.metrics:
            metric_agg[metric.__name__()] /= runs
        return metrics

from sklearn.model_selection import KFold
def split(x,y):
    kfold = KFold(n_splits=5)
    splits = []
    for train_ind in kfold.split(x):
        test_ind = [ind for ind in range(len(y)) if ind not in train_ind]
        splits.append([x[train_ind],x[test_ind],y[train_ind],y[test_ind]])
    return splits
