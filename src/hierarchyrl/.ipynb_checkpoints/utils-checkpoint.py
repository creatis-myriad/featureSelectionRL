import sklearn.model_selection, sklearn.preprocessing
import numpy as np
import functools, inspect 

# To ignore warnings. When doing inspect, I will access to all variables of the object, including those deprecated, raising a lot of messages
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
sigmoid = lambda x: 1/(1 + np.exp(-x))

def crossvalPredictSampleWeights(model, X, y, sample_weight = None, cv = 10, method = 'predict', regression = True, fit_params = {}):
    if sample_weight is None:
        sample_weight = np.ones(len(y))
    if isinstance(cv, int):
        if not regression:
            cv = sklearn.model_selection.StratifiedKFold(cv, shuffle = True, random_state = 1)
        else:
            cv = sklearn.model_selection.KFold(cv, shuffle = True, random_state = 1)

    # Do splits
    if method == 'predict_proba':
        y_pred = np.zeros((len(y), int(np.max(y)) + 1))
    else:
        y_pred = y.copy()
    
    for train_index, test_index in  cv.split(X, y):
        model.fit(X[train_index], y[train_index], sample_weight = sample_weight[train_index], **fit_params)
        if method == 'predict':
            y_pred[test_index] = model.predict(X[test_index])
        elif method == 'predict_proba':
            try:
                y_pred[test_index] = model.predict_proba(X[test_index])
            except:
                y_pred[test_index, 1] = sigmoid(model.decision_function(X[test_index]))
                y_pred[test_index, 0] = 1 - y_pred[test_index, 1]
        else:
            raise ValueError('Unknown predict method')
    return y_pred


class ClassDecorator(object):
    def __init__(self, orig_class):
        self._orig_class = orig_class
    def __call__(self, arg=None):
        return self._orig_class.__call__()

    def __getattr__(self, name):
        return getattr(self._orig_class, name)

def make_2d(X):
    if (len(X.shape) == 1):
        return X.reshape((-1,1))
    else:
        return X
    
class PredictorStandardise(ClassDecorator):
    """
    Does the same as pipeline, but accepting sample weights (broken when using Pipeline + CV, they do not split)
    """    
    def __init__(self, orig_class, scaleY = False):
        self.list_functions_X = []
        self.scaleY = scaleY
        for c in inspect.getmembers(orig_class, predicate=inspect.ismethod):
            if c[0][0] == '_':
                continue
            if 'X' in c[1].__code__.co_varnames:
                self.list_functions_X.append(c[0])
        super().__init__(orig_class)
        
    def transform_X(self, f):
        @functools.wraps(f)
        def f_2(X, *args, **kwargs):
            X_hat = self.scaler.transform(X)
            return f(X_hat, *args, **kwargs)
        return f_2
    def fit(self, X, y,*args, **kwargs):
        self.scaler = sklearn.preprocessing.StandardScaler()
        X_hat = self.scaler.fit_transform(X)
        if self.scaleY:
            self.originalYShape = len(y.shape)
            self.scalerY = sklearn.preprocessing.StandardScaler()
            y_hat = self.scalerY.fit_transform(make_2d(y))
        else:
            y_hat = y
        return self._orig_class.fit(X_hat, y_hat, *args, **kwargs)
    
    def predict(self, X,*args, **kwargs):
        X_hat = self.scaler.transform(X)
        if self.scaleY:
            yRes = self.scalerY.inverse_transform(self._orig_class.predict(X_hat, *args, **kwargs))
            if self.originalYShape == 1:
                yRes = yRes.flatten()
            return yRes
        else:
            return self._orig_class.predict(X_hat, *args, **kwargs)

    def __getattr__(self, name):
        if name in self.list_functions_X:
            return self.transform_X(getattr(self._orig_class, name))
        else:
            return super().__getattr__(name)
