class RunDetail():
    '''
    All meta features as lists; if you want to keep constant, give only one choice

    '''
    def __init__(self):
        self.metrics = []
        self.split_strategy = None
        self.target = None
        self.targettype = None

class ModelRun():
    '''
    Describes experiments and their results considering a constant.
    Manages the expiriment.
    '''
    def __init__(self):
        self.attempts = []
        self.details = None # RunDetail

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
