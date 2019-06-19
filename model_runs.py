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
    def __init__(self,process,model,features,target,spliter,metrics):
        self.sklearn_model = model #ClassifierModel
        self.process = process
        self.x = features
        self.y = target
        self.splitter = splitter
    def evaluate(self):
        for x_train,x_test,y_train,y_test
        pass #will return metrics


class ClassifierModel():
    '''
    Represents decisions from original data all the way until results.
    '''
    def __init__(self,x,y,model):
        self.sklearn_model = model
        self.x = process(x) #Normalizer,Scaling, how to handle columns
        self.y = y
    def predict(self,x):
        return self.sklearn_model
