import numpy as np, collections, time, scipy
import sklearn, sklearn.dummy
from . import policy


# For the Kernel-CSL
try:
    import tensorflow as tf
    import scipy.optimize as sopt
    tensorflowLoaded = True
except:
    tensorflowLoaded = False
    
class PolicyPowersetClassification:
    """
    TODO: Sometimes there is only one action, but still models are trained. Solve that to improve efficiency a bit.
    """
    def __init__(self, X_powerset, modelClassification, modelAction,
                 max_data_acquisition = -1, acquisitionCost = None, loss = 'accuracy'):
        self.modelClassification = modelClassification
        self.modelQActionTemplate = modelAction
        self.X_powerset = X_powerset

        self.y_gt = X_powerset.y
        if self.y_gt is None:
            raise ValueError('The powerset need to be annotated')
            
        self.baseProbabilities = np.array([np.mean(self.y_gt == i) for i in range(self.X_powerset.nClasses)])
        self.nSamplesTrain = len(self.y_gt)
        self.nVariables = X_powerset.ACTION_FINISH 
        
        self.max_number_data = max_data_acquisition if max_data_acquisition != -1 else self.nVariables
        self.acquisitionCost = collections.defaultdict(lambda: 0) if acquisitionCost is None else acquisitionCost
        self.stateCost = {k: sum((self.acquisitionCost[v] for v in self.X_powerset.getVariablesFromEncoding(k) ))  for k in self.X_powerset.getAllCombinations()}
        self.weights = collections.defaultdict(lambda: None)
        self.mappingActions = self.X_powerset.mappingActions
        if loss == 'accuracy':
            self.loss = lambda y, y1: y == np.argmax(y1, axis =1)
        elif loss == 'logloss':
            self.loss = lambda y, y1: np.log(np.abs(1 - y + y1[:, 1] + 1e-9))
        else:
            raise ValueError()
    @property
    def ACTION_FINISH(self):
        return self.X_powerset.ACTION_FINISH
    
    def makePrediction(self, state):
        k = state.k & (self.X_powerset.DATA_MASK)
        if k != 0:
            return self.X_powerset.classificationModels[state.k].predict_proba(state.v.reshape((1,-1)))
        else:
            return self.baseProbabilities.reshape((1, -1))
    
    def trainIteration(self):
        self.action_model = collections.defaultdict(self.modelClassification)
        self.values = {}
        self.bestAction = {}
        
        # Generate the values of the finish states
        for k in self.X_powerset.getAllCombinationsFinish():
            if k & self.X_powerset.DATA_MASK == 0:
                pass
            else:
                # Use accuracy loss
                value =  self.loss(self.X_powerset.y,  self.yPred_s[k & self.X_powerset.DATA_MASK]) - self.stateCost[k & self.X_powerset.DATA_MASK]
                self.values[k] = value

        # Update using Bellman equation
        for k in self.X_powerset.getAllCombinations(reverse = True):
            if k == 0:
                continue
            X_k = self.X_powerset.getData(k)
            values = np.zeros((self.X_powerset.nSamples, len(self.X_powerset.mappingActions[k])))

            for i,a in enumerate(self.X_powerset.mappingActions[k]):
                if a != self.ACTION_FINISH and self.X_powerset.countActiveBits(k) >= self.max_number_data:
                    continue
                k_a = self.X_powerset.applyAction(a, k)
                values[:, i] = self.values[k_a]
                                
            self.bestAction[k] = np.argmax(values, axis = 1) 
            if True:
                self.action_model[k] = self.modelQActionTemplate()
            else:
                self.action_model[k] = sklearn.dummy.DummyClassifier()
                values = self.bestAction[k]
            self.action_model[k].fit(X_k, values, sample_weight = self.weights[k])
            action_Choosen = self.action_model[k].predict(X_k)
            
            # Probably need to do some smoothing, otherwise, values will be overoptimistic...
            self.values[k] = np.array([ values[i, a]  for i, a in enumerate(action_Choosen)])
            
        self.valuesLevel0 = [np.mean(self.values[1 << i]) for i in range(self.nVariables)] + [np.mean(self.loss(self.X_powerset.y, self.yPred_s[0]))]
        self.choice_first_action = np.argmax(self.valuesLevel0)

    
    def train(self, nIts = 10, debug = False,  offPolicyEpsilon = 0.2, cv = 10, ):
        self.X_powerset.trainClassificationModels(self.modelClassification, weights = None)
        self.yPred_s = self.X_powerset.getPredictions(self.modelClassification(), cv = cv,weights = None)
        for n_it in range(nIts):
            tStart = time.time()
            self.trainIteration()
            self.weights = self.countVisits(self.X_powerset, offPolicyEpsilon)            
            if debug:
                exec_time = time.time() -tStart
                print(f'Iteration {n_it}: time = {exec_time:.2f}')
        self.X_powerset.trainClassificationModels(self.modelClassification, weights = self.weights)
        self.choice_first_action = np.argmax(self.valuesLevel0)

    def countVisits(self, X, epsilon_explore = .1):
        visits = {k: np.zeros(X.nSamples, dtype = float) if k != 0. else  np.ones(X.nSamples, dtype = float) 
                    for k in X.getAllCombinations()}
        for k in X.getAllCombinations(reverse = False):
            w = visits[k]

            if k == 0:
                values = np.repeat(self.valuesLevel0, X.nSamples).reshape((-1, X.nSamples)).T
                possibleActions = list(range(X.ACTION_FINISH + 1))
                bestAction = np.argmax(values, axis = 1)
            else:
                possibleActions = self.mappingActions[k]
                bestAction = np.array([possibleActions[i] for i in self.action_model[k].predict(X.getData(k))])
            nActions = len(possibleActions)
            for i,a in enumerate(possibleActions):
                if a == X.ACTION_FINISH:
                    continue
                kNext_a = X.applyAction(a,k)
                visits[kNext_a] += w * epsilon_explore/nActions
                visits[kNext_a] += w * (1 - epsilon_explore) * np.where(bestAction == i, 1, 0)
        for k in X.getAllCombinations(reverse = False):
            visits[k] = visits[k]/np.sum(visits[k]) * X.nSamples
            visits[k | (1 << X.ACTION_FINISH)] = visits[k]
        return visits


    def nextAction(self, s):
        k = s.k
        if k == 0:
            return self.choice_first_action
        elif self.X_powerset.hasFinished(k):
            raise ValueError('Already finished')
        else:
            aa = self.action_model[k].predict(s.v.reshape((1,-1)))[0]
            a = self.mappingActions[k][aa]
            return a

 
            
    def simulateEvaluateInPolicy(self, X_powerset = None):
        X_powerset = self.X_powerset if X_powerset is None else X_powerset
        e = policy.EnvironmentDataAcquisitionSample(X_powerset)
        yPred = np.zeros((X_powerset.nSamples, self.X_powerset.nClasses))
        ks = np.zeros(X_powerset.nSamples, dtype = int)
        paths = []
        for i in range(X_powerset.nSamples):
            p = []
            s = e.reset(i)
            finish = False
            while not finish:
                p.append(s.k)
                a = self.nextAction(s)
                s, finish = e.applyAction(s, a)
                if finish:
                    yPred[i] = self.makePrediction(s)[0]
                    ks[i] = s.k
                    p.append(s.k)

            paths.append(p)
        return yPred, ks, paths
    
    
##
# Different algorithms to solve the CSL problem
##
#CSL using kernel ridge with L2 regularisation
class KernelCSL(sklearn.base.BaseEstimator):
    """
    Solve the CSL minimisation problem using a kernel method and a convex relaxation (using softmax)
    """
    def __init__(self, alpha = 1):
        if not tensorflowLoaded:
            raise ValueError('KernelCSL requires tensorflow')
        self.alpha = alpha
        super().__init__()
        
    def fit(self, X, W, sample_weight = None):
        dist_2 = scipy.spatial.distance_matrix(X,X)**2
        self.sigma = np.sqrt(X.shape[1])
        K = np.exp(-dist_2/self.sigma**2)
                
        if sample_weight is not None:
            W = W* sample_weight.reshape((-1, 1))
        
        @tf.function
        def loss(K, W, thetas):
            """
            loss using tensorflow

            \mean_i  < W_i ,softmax(theta *K^hat) > + \alpha ||theta||^2
            
            TODO: add scale and bias parameters
            """
            softmax = tf.math.softmax(tf.matmul(K, thetas), axis = 1)
            loss_data = tf.reduce_mean(tf.reduce_sum(softmax * (-W), axis = 1))
            return loss_data + self.alpha * tf.tensordot(thetas, thetas, axes = [(0,1), (0,1)])

        @tf.function
        def val_and_grad(x):
            with tf.GradientTape() as tape:
                tape.watch(x)
                l = loss(K, W, x)
            grad = tape.gradient(l, x)
            return l, grad

        def func(x):
            return [vv.numpy().astype(np.float64).flatten()  for vv in val_and_grad(tf.constant(x.reshape(W.shape), dtype=tf.float64))]

        self.resdd= sopt.minimize(fun=func, x0=np.zeros_like(W).flatten(),
                                              jac=True, method='L-BFGS-B')
        self.thetas = self.resdd.x.reshape(W.shape)
        self.K = K
        self.X = X
        
    def predict(self, X):
        dist_2 = scipy.spatial.distance_matrix(X, self.X)**2
        K = np.exp(-dist_2/self.sigma**2)
        return np.argmax(K@self.thetas, axis = 1)
    
    def score(self, X, W):
        pred = self.predict(X)
        return sum([W[i,j] for i,j in enumerate(pred)])
    
    
class ClassificationCSL(sklearn.base.BaseEstimator):
    """
    Solve the CSL minimisation problem using a kernel method and a convex relaxation (using softmax)
    """
    def __init__(self, classifier):
        self.classifier = classifier
        
        
    def fit(self, X, W, sample_weight = None):
        bestAction = np.argmax(W, axis = 1) 
        if  len(np.unique(bestAction)) <= 1:
            self.singleClass = True
            self.prediction = bestAction[0]
        else:
            self.singleClass = False
            self.classifier.fit(X, bestAction,sample_weight = sample_weight)
            
    def predict(self, X):
        if self.singleClass:
            return np.repeat(self.prediction, X.shape[0])
        else:
            return self.classifier.predict(X)
    
    def score(self, X, W):
        pred = self.predict(X)
        return sum([W[i,j] for i,j in enumerate(pred)])
    
class RegressionCSL(sklearn.base.BaseEstimator):
    """
    Solve the CSL minimisation problem using a kernel method and a convex relaxation (using softmax)
    """
    def __init__(self, regressor):
        self.regressor = regressor
        
    def fit(self, X, W, sample_weight = None):
        bestAction = np.argmax(W, axis = 1)
        if  len(np.unique(bestAction)) <= 1:
            self.singleClass = True
            self.prediction = bestAction[0]
        else:
            self.singleClass = False
            self.regressor.fit(X, W,sample_weight = sample_weight)
            
    def predict(self, X):
        if self.singleClass:
            return np.repeat(self.prediction, X.shape[0])
        else:
            return np.argmax(self.regressor.predict(X), axis = 1)
    
    def score(self, X, W):
        pred = self.predict(X)
        return sum([W[i,j] for i,j in enumerate(pred)])