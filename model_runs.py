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
    def __init__(self,model,features,target,resample,scale,**modelargs):
        self.sklearn_model = model #ClassifierModel
        self.process = process
        self.x = features
        self.y = target
        self.split = split
    def evaluate(self):
        metric_agg = {}
        for metric in metrics:
            metric_agg[metric[0]] = 0
        runs = 0
        for x_train,x_test,y_train,y_test in self.split(self.x,self.y):
            x_train,x_test = self.process(x_train,x_test)



class ClassifierModel():
    '''
    Represents decisions from original data all the way until results.
    '''
    def __init__(self,df,process):
        self.sklearn_model = None
        self.data_processing = None #Normalizer,Scaling, how to handle columns
