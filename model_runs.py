class RunDetail():
    def __init__(self):
        self.metrics = []
        self.split_strategy = None
        self.target = None
        self.targettype = None

class ModelRun():
    def __init__(self):
        self.attempts = []
        self.details = None # RunDetail

class Attempt():
    def __init__(self):
        self.model = None #ClassifierModel
        self.details = None

class ClassifierModel():
    def __init__(self):
        self.sklearn_model = None
        self.data_scaler = None #Normalizer,Scaling
        self.?
