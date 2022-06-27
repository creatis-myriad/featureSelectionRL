from . import powerset, utils
import collections, numpy as np, logging, time
import sklearn, sklearn.metrics

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
    
    def trainIterationDQN(self, split_Q_S_datasets = True, cv = 10, random_rng = np.random.default_rng(), useWeightsClassification = True):
        if split_Q_S_datasets:
            self.X_Q, countsQ, self.X_S, countsS = self.X_powerset.split(0.5, self.visitCounts, random_rng = random_rng)
        else:
            self.X_Q, countsQ = self.X_powerset, self.visitCounts
            self.X_S, countsS = self.X_powerset, self.visitCounts
        self.countsS = countsS
        self.countsQ = countsQ
        self.mappingActions = collections.defaultdict(list)

        self.yPred_s = self.X_S.getPredictions(self.modelClassification(), cv = cv, weights = countsS if useWeightsClassification else None) # Not sure if this is OK, or maybe I should better use the S dataset...
        self.transitionModels =  collections.defaultdict(self.modelQActionTemplate)
        self.S_model = collections.defaultdict(self.modelQActionTemplate)
        self.values = {}
        
        # Generate the values of the finish states
        for k in self.X_powerset.getAllCombinationsFinish():
            if k & self.X_S.DATA_MASK == 0:
                pass
            else:
                # Use accuracy loss
                value =  self.loss(self.X_S.y,  self.yPred_s[k & self.X_S.DATA_MASK]) - self.stateCost[k & self.X_S.DATA_MASK]
                self.S_model[k].fit(self.X_S.getData(k), value, sample_weight = countsS[k])
                self.values[k] = value
        self.max_value_s = {}
        # Update using Bellman equation
        for k in self.X_powerset.getAllCombinations(reverse = True):
            if k == 0:
                continue
            max_value_s = - 1000 * np.ones(self.X_S.y.shape, dtype = float)
            X_s_k = self.X_S.getData(k)
            # Compute the best action
            for a in self.X_powerset.mappingActions[k]:
                if a != self.ACTION_FINISH and self.X_powerset.countActiveBits(k) >= self.max_number_data:
                    continue

                k_a = self.X_powerset.applyAction(a, k)
                self.transitionModels[(k, a)].fit(self.X_Q.getData(k), self.S_model[k_a].predict(self.X_Q.getData(k_a)).flatten(), sample_weight = countsQ[k])
                np.maximum(max_value_s, self.transitionModels[(k, a)].predict(X_s_k).flatten(), out = max_value_s)
                self.mappingActions[k].append(a)
            self.max_value_s[k] = max_value_s
            self.S_model[k].fit(X_s_k, max_value_s, sample_weight = countsS[k])
            self.values[k] = max_value_s
        self.valuesLevel0 = [np.mean(self.S_model[1 << i].predict(self.X_S.getData(1 << i))) for i in range(self.nVariables)] + [np.mean(self.loss(self.X_S.y, self.yPred_s[0]))]
        self.choice_first_action = np.argmax(self.valuesLevel0)

    def trainIterationSimulateValue(self, valueEstimationMode = 'prediction_crossval'):
        """
        The estimates of value can be overoptimisting, to avoid that, double Q-learning was proposed, by using a separate data
        
        The proposed solution is  "Fitted value iteration method"
        
        Not needed, already in the previous code 
        """
        self.yPred = self.X_powerset.getPredictions(self.modelClassification())
        self.values = {}
        self.transitionModels = {}
        self.mappingActions = collections.defaultdict(list)
        # Put the values of the end states
        for k in self.X_powerset.getAllCombinations(reverse = True):
            # Use accuracy loss
            self.values[self.X_powerset.applyAction(self.ACTION_FINISH, k)] =  self.loss(self.y_gt, self.yPred[k]).astype(float) - self.stateCost[k]

        # Update using Bellman equation
        for k in self.X_powerset.getAllCombinations(reverse = True):
            if k == 0:
                continue
                
            X_k = self.X_powerset.getData(k)
            weights = self.visitCounts[k]
            
            self.values[k] = np.finfo(float).min * np.ones(self.y_gt.shape)
            best_action = np.zeros(self.y_gt.shape, dtype = int)
            best_action_value= - 1000 * np.ones(self.y_gt.shape, dtype = float)

            for a in self.X_powerset.mappingActions[k]:
                if a != self.ACTION_FINISH and self.X_powerset.countActiveBits(k) >= self.max_number_data:
                    continue
                self.transitionModels[(k, a)] = self.modelQActionTemplate()
                # TODO: cross-val predict
                
                if valueEstimationMode == 'prediction_crossval':
                    value_predicted = utils.crossvalPredictSampleWeights(self.transitionModels[(k, a)],
                                                                         X_k, 
                                                                         self.values[self.X_powerset.applyAction(a, k)], 
                                                                         sample_weight = weights)
                    self.transitionModels[(k, a)].fit(X_k, self.values[self.X_powerset.applyAction(a, k)], sample_weight = weights)

                elif valueEstimationMode == 'prediction_normal':
                    self.transitionModels[(k, a)].fit(X_k, self.values[self.X_powerset.applyAction(a, k)], sample_weight = weights)
                    value_predicted = self.transitionModels[(k, a)].predict(X_k).flatten()
                else:
                    raise ValueError('Incorrect prediction strategy [accepted prediction_crossval/prediction_normal]')

                
                k_a = self.X_powerset.applyAction(a, k)
                self.values[k] = np.where(value_predicted > best_action_value,  self.values[k_a], self.values[k])
                best_action_value = np.where(value_predicted > best_action_value,  value_predicted, best_action_value)

                # Add that it is possible to take action a in state k.
                self.mappingActions[k].append(a)
                
        self.valuesLevel0 =[np.mean(self.values[1 << i]) for i in range(self.nVariables)] + [np.mean(self.loss(self.y_gt, self.yPred[0]))]
        self.choice_first_action = np.argmax(self.valuesLevel0)
        return self.values, self.transitionModels
    
    def train(self, nIts = 10, debug = False, test_set = None, 
    valueEstimationMode = 'prediction_normal', valueUpdate = 'max', offPolicyEpsilon = 0.2, cv = 10, random_rng = np.random.default_rng()):
        for n_it in range(nIts):
            tStart = time.time()
            if valueEstimationMode == 'prediction_normal':
                self.trainIterationDQN(split_Q_S_datasets = False, cv = cv, random_rng = random_rng)
                #values, models = self.trainIteration(valueEstimationMode = valueEstimationMode, valueUpdate = valueUpdate)
            elif valueEstimationMode == 'DQN':
                self.trainIterationDQN(random_rng = random_rng)
            elif valueEstimationMode == 'simulate':
                self.trainIterationSimulateValue()#valueEstimationMode = 'prediction_normal')
            else:
                raise ValueError()
            self.resetCounts(v = offPolicyEpsilon/(1 - offPolicyEpsilon))
            self.simulateAndCountInPolicy(self.nSamplesTrain, mode = 'deterministic')            
            if debug:
                exec_time = time.time() -tStart
                print(f'Iteration {n_it}: time = {exec_time:.2f}')
        self.X_powerset.trainClassificationModels(self.modelClassification, weights = self.visitCounts)
        self.choice_first_action = np.argmax(self.valuesLevel0)

    def getActionValues(self, k, v):
        values = {a: self.transitionModels[(k, a)].predict(v) for a in self.mappingActions[k]}
        return values

    def nextAction(self, s):
        k = s.k
        if k == 0:
            return self.choice_first_action
        elif self.X_powerset.hasFinished(k):
            raise ValueError('Already finished')
        else:
            values = [self.transitionModels[(k, a)].predict(s.v.reshape((1,-1)))[0] for a in self.mappingActions[k]]
            a = self.mappingActions[k][np.argmax(values)]
            return a
        
    def resetCounts(self, v = 1):
        self.visitCounts = { k : v*np.ones(self.X_powerset.nSamples) for k in self.X_powerset.getAllCombinationsFinish()}

        
    def simulateAndCountInPolicy(self, nSamples, mode = 'random', X_powerset = None):
        self.finish_k = np.zeros(nSamples, dtype = np.int)
        e = EnvironmentDataAcquisitionSample(self.X_powerset if X_powerset is None else X_powerset)
        for i in range(nSamples):
            s = e.reset() if mode == 'random' else e.reset(i)
            finish = False
            while not finish:
                a = self.nextAction(s)
                s, finish = e.applyAction(s, a)
                self.visitCounts[s.k][e.id] += 1
            self.finish_k[i] = s.k
            
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