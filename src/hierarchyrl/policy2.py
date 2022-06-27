import numpy as np, collections, time
import sklearn, sklearn.dummy

StatePartiallyObservedData = collections.namedtuple('StatePartiallyObservedData', 'k v')


class PolicyPowerset:
    """
    TODO: Sometimes there is only one action, but still models are trained. Solve that to improve efficiency a bit.
    """
    def __init__(self, X_powerset, modelClassification, modelQAction,
                 max_data_acquisition = -1, acquisitionCost = None, loss = 'accuracy'):
        self.modelClassification = modelClassification
        self.modelQActionTemplate = modelQAction

        self.X_powerset = X_powerset
        self.mappingActions = self.X_powerset.mappingActions

        self.y_gt = X_powerset.y
        if self.y_gt is None:
            raise ValueError('The powerset need to be annotated')
            
        self.baseProbabilities = np.array([np.mean(self.y_gt == i) for i in range(self.X_powerset.nClasses)])
        self.nSamplesTrain = len(self.y_gt)
        self.nVariables = X_powerset.ACTION_FINISH 
        
        self.resetCounts()
        self.max_number_data = max_data_acquisition if max_data_acquisition != -1 else self.nVariables
        self.acquisitionCost = collections.defaultdict(lambda: 0) if acquisitionCost is None else acquisitionCost
        self.stateCost = {k: sum((self.acquisitionCost[v] for v in self.X_powerset.getVariablesFromEncoding(k) ))  for k in self.X_powerset.getAllCombinations()}
                
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
    
    def trainIteration(self, useWeightsClassification = True):
        self.transitionModels =  collections.defaultdict(self.modelQActionTemplate)
        self.values = {}
        
        # Generate the values of the finish states
        for k in self.X_powerset.getAllCombinationsFinish():
            if k & self.X_powerset.DATA_MASK == 0:
                pass
            else:
                # Use accuracy loss
                self.values[k] =  self.loss(self.X_powerset.y,  self.yPred[k & self.X_powerset.DATA_MASK]) - self.stateCost[k & self.X_powerset.DATA_MASK]
        # Update using Bellman equation
        for k in self.X_powerset.getAllCombinations(reverse = True):
            if k == 0:
                continue
            X_k = self.X_powerset.getData(k)
            # Compute the best action
            values = np.zeros((self.X_powerset.nSamples, len(self.X_powerset.mappingActions[k])))
            for i, a in enumerate(self.X_powerset.mappingActions[k]):
                if a != self.ACTION_FINISH and self.X_powerset.countActiveBits(k) >= self.max_number_data:
                    continue

                k_a = self.X_powerset.applyAction(a, k)
                values[:, i] = self.values[k_a]                
            self.transitionModels[k].fit(X_k, values, sample_weight = self.visitCounts[k])
            self.values[k] = np.max(self.transitionModels[k].predict(X_k), axis = 1)
            
        self.valuesLevel0 = [np.mean(self.values[1 << i]) for i in range(self.nVariables)] + [np.mean(self.loss(self.X_powerset.y, self.yPred[0]))]
        self.choice_first_action = np.argmax(self.valuesLevel0)

    def train(self, nIts = 1, debug = False, offPolicyEpsilon = 0.2, useWeightsClassification = False):
        self.yPred = self.X_powerset.getPredictions(self.modelClassification(), cv = 20,
                                            weights = self.visitCounts if useWeightsClassification else None) 

        for n_it in range(nIts):
            tStart = time.time()
            self.trainIteration()
            if n_it != nIts - 1:
                self.visitCounts = self.countVisits(self.X_powerset,offPolicyEpsilon)
            if useWeightsClassification:
                self.yPred =self.X_powerset.trainClassificationModels(self.modelClassification, weights = self.visitCounts)

            if debug:
                exec_time = time.time() -tStart
                print(f'Iteration {n_it}: time = {exec_time:.2f}')
                
        self.X_powerset.trainClassificationModels(self.modelClassification, weights = self.visitCounts)
        self.choice_first_action = np.argmax(self.valuesLevel0)

    def getActionValues(self, k, v):
        valuesArray = self.transitionModels[k].predict(v)
        values = {a: valuesArray[i] for i,a in enumerate(self.mappingActions[k])}
        return values

    def nextAction(self, s):
        k = s.k
        if k == 0:
            return self.choice_first_action
        elif self.X_powerset.hasFinished(k):
            raise ValueError('Already finished')
        else:
            values = self.transitionModels[k].predict(s.v.reshape((1,-1)))[0]
            a = self.mappingActions[k][np.argmax(values)]
            return a
        
    def resetCounts(self, v = 1):
        self.visitCounts = { k : v*np.ones(self.X_powerset.nSamples) for k in self.X_powerset.getAllCombinationsFinish()}

        
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
                bestAction = np.array([possibleActions[i] for i in np.argmax(self.transitionModels[k].predict(X.getData(k)), axis = 1)])
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


    def simulateEvaluateInPolicy(self, X_powerset = None):
        X_powerset = self.X_powerset if X_powerset is None else X_powerset
        e = EnvironmentDataAcquisitionSample(X_powerset)
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

class EnvironmentDataAcquisitionSample:
    def __init__(self, X_powerset):
        self.X_p = X_powerset
        
    def reset(self, i = None):
        if i is None:
            self.id = np.random.randint(self.X_p.nSamples)
        else:
            self.id = i
        return StatePartiallyObservedData(0, np.zeros((1, 0)))
    
    def applyAction(self, s, a):
        kNew = self.X_p.applyAction(a, s.k)
        finish = a == self.X_p.ACTION_FINISH
        state = StatePartiallyObservedData(self.X_p.applyAction(a, s.k), self.X_p.getDataSample(kNew, self.id))
        return state, finish
    
    @property
    def y(self):
        return self.X_p.y[self.id]