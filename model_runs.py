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
        self.details = None # RunDetail

    def run_model(self, rd):
        target_df = self.df[rd['target']]
        if rd['target'] in rd['features']:
            features_df = self.df[rd['features']].drop(rd['target'], axis=1)
        else:
            features_df = self.df[rd['features']]

        #attempt = Attempt(rd.model,rd.features,rd.target,rd.resample,rd.scaler,rd.kwargs)
        rd['metrics'] = {'result':'hello'}  #attempt.evaluate()
        self.attempts.append(rd)
        return target_df, features_df


class Attempt():
    '''
    Contains the ClassifierModel (independent variable) and RunDetails (control/constant)
    '''
    def __init__(self,model,features,target,resample,scale,metrics,**modelargs):
        self.sklearn_model = model #ClassifierModel
        self.process = process
        self.x = features
        self.y = target
        self.split = split
        self.modelargs = modelargs
        self.metrics = metrics
    def evaluate(self):
        metric_agg = {}
        for metric in self.metrics:
            metric_agg[metric.__name__()] = 0
        runs = 0
        for x_train,x_test,y_train,y_test in self.split(self.x,self.y):
            runs += 1
            self.process.fit(x_train)
            x_train,x_test = self.process.transform(x_train),self.process.transform(x_test)
            model = self.sklearn_model(self.modelargs)
            model.fit(x_train,y_train)
            preds = model.predict(x_test)
            for metric in self.metrics:
                metric_agg[metric.__name__()] += metric(preds,y_test)

        for metric in self.metrics:
            metric_agg[metric.__name__()] /= runs
        return metrics
