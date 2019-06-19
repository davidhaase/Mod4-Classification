class RunDetail():
    '''
    All meta features as lists; if you want to keep constant, give only one choice

    '''
    def __init__(self):
        self.metrics = []
        self.split_strategy = None
        self.target = None

class ModelRun():
    '''
    Describes experiments and their results considering a constant.
    Manages the expiriment.
    '''
    def __init__(self):
        self.attempts = []
        self.details = None # RunDetail

    def run_model(self, details):
        results = get_metrics
        self.attempts.append(details)

    def get_metrics(self, i=0):
        if (i < len(self.attempts)):
            return self.attempts[i]

class Attempt():
    '''
    Contains the ClassifierModel (independent variable) and RunDetails (control/constant)
    '''
    def __init__(self):
        self.model = None #ClassifierModel
        self.details = None

class ClassifierModel():
    '''
    Represents decisions from original data all the way until results.
    '''
    def __init__(self,df,process):
        self.sklearn_model = None
        self.data_processing = None #Normalizer,Scaling, how to handle columns
        self.?
