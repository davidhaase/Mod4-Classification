class RunDetail():
    '''
    All meta features as lists; if you want to keep constant, give only one choice
    '''
    def __init__(self):
        self.metrics = []
        self.split_strategy = None
        self.target = None
        self.process = None
        self.model = None
        self.features = None
        self.target = None
        self.splitter = None
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

    def run_model(self, details):
        attempt_count = 1
        for detail in details:
            attempt_count *= len(detail)
        return attempt_count

    def get_metrics(self, i=0):
        if (i < len(self.attempts)):
            return self.attempts[i]

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
