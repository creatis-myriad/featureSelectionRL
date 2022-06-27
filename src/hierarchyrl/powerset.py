import collections, numpy as np, sklearn, sklearn.model_selection, logging
from . import utils

def make_2d_np(x):
    if len(x.shape) == 1:
        return x.reshape((-1, 1))
    else:
        return x
    
def make_2d_row_np(x):
    if len(x.shape) == 1:
        return x.reshape((1, -1))
    else:
        return x

class Powerset:
    """
    Class for unrolling a set of variables in all its possible combinations (2^N possibilities, including the empty).
    """
    N_MAX_VARIABLES = 5 #This implementation can be superinefficient, so limited to small datasets only
    def __init__(self, variablesDict, y = None, ignoreMaxNumberVariables = False):
        variablesDict = {k: make_2d_np(v) for k,v in variablesDict.items()}
        self.variablesDict = variablesDict
        self.ACTION_FINISH = len(variablesDict)
        self.N_VARIABLES = len(variablesDict)
        self.variablesNames = list(variablesDict.keys())
        self.powers_two = [1 << i for i, _ in enumerate(variablesDict)]
        self.ignoreMaxNumberVariables = ignoreMaxNumberVariables
        if not ignoreMaxNumberVariables and len(variablesDict) > self.N_MAX_VARIABLES :
            raise ValueError("This is only thought for a small number of variables...")
        
        self.variablesSize = {k : v.shape[1] for k,v  in variablesDict.items()}
        self.inputSizes = {k : sum([self.variablesSize[varName] for varName in self.getVariablesFromEncoding(k)]) for k in self.getAllCombinations()}
        self.mappingActions = {k : np.array([ self.variablesNames.index(v) for v in self.variablesNames
                                             if (v not in self.getVariablesFromEncoding(k))] + [self.ACTION_FINISH])
                               for k in self.getAllCombinations()
                              }
        self.y = y
        if y is not None:
            self.nClasses = int(np.max(y) + 1)
        else:
            self.nClasses = 0
        self.DATA_MASK = (1 << self.ACTION_FINISH) - 1
        k = list(variablesDict.keys())[0]
        self.nSamples = variablesDict[k].shape[0]
        self.X_all = np.concatenate([make_2d_np(variablesDict[n]) for n in self.variablesNames], axis = 1)
        #Indices
        indicesVars = {}
        prevIndex = 0
        for k, s in self.variablesSize.items():
            indicesVars[k] = list(range(prevIndex, prevIndex + s))
            prevIndex += s
        self.all_indices = {k: 
                           [id for v in self.getVariablesFromEncoding(k) for id in indicesVars[v]]
                            for k in self.getAllCombinations()}
        
    def applyAction(self, a, k):
        """
        Applies the action a to state k.
        returns newState, correctAction, endState
        """
        return k | (1 << a)
        
    def getData(self, k):
        k = k & self.DATA_MASK
        return self.X_all[:,self.all_indices[k]]
    
    def hasFinished(self,k):
        return (k & (1 << self.ACTION_FINISH))
    
    def getDataSample(self, k, i):
        k = k & self.DATA_MASK
        return self.X_all[i,self.all_indices[k]]


    def getEncodingFromVariables(self, v):
        i = 0
        for j, vv in enumerate(self.variablesNames):
            if vv in v:
                i |= self.powers_two[j]
        return i
    
    def getVariablesFromEncoding(self, k):
        return [v for i, v in enumerate(self.variablesNames) if k & self.powers_two[i]] 
    
    @property
    def X_full(self):
        return self.getData(self.DATA_MASK)
    
    def getAllCombinations(self, reverse = False):
        #if self.ACTION_FINISH > self.N_MAX_VARIABLES:
        #    raise ValueError("This is only thought for a small number of variables... (5 max)")
        if not reverse:
            return list(range(1 << self.ACTION_FINISH))
        else:
            return list(range((1 << self.ACTION_FINISH) -1, 0 - 1, -1))    
        
    def getAllCombinationsFinish(self, reverse = False):
        return  self.getAllCombinations(reverse) + [s | (1 <<self.ACTION_FINISH) for s in self.getAllCombinations(reverse)]
    
    def countActiveBits(self, k):
        return len(self.getVariablesFromEncoding(k))
    
    def trainClassificationModels(self, model, weights = None):
        self.classificationModels = { k | (1 << self.ACTION_FINISH)  : model() for k in self.getAllCombinations()}
        for k, m in self.classificationModels.items():
            if k& self.DATA_MASK == 0:
                continue
            m.fit(self.getData(k), self.y, **{"sample_weight" : weights[k]} if weights is not None else {}) 
                
    def getPredictions(self, model, cv = 10, weights = None, getPredictionsMethod = 'predict_proba'):
        yPred = {}
        for k in self.getAllCombinations():
            if k == 0:
                continue
            model.fit(self.getData(k), self.y)
            yPred[k] = utils.crossvalPredictSampleWeights(model, self.getData(k), 
                                                          self.y, sample_weight = weights[k] if weights else None,
                                                          method = getPredictionsMethod, cv = cv, regression = False)
        yPred[0] =  np.array([np.mean(self.y == i) for i in range(self.nClasses)]).reshape((1, -1)) * np.ones([self.nSamples, self.nClasses])
        return yPred
        
    def split(self, p, visitCounts = None, random_rng = np.random.default_rng()):
        if self.y is not None:
            all_idx_set_1 = np.zeros((0), dtype = int)
            all_idx_set_2 = np.zeros((0), dtype = int)
            for i in range(self.nClasses):
                n_i = np.sum(self.y == i)
                p_i = int(p *n_i)

                idxi = np.where(self.y == i)[0]
                idxi = random_rng.permutation(idxi)
                idxi_1, idxi_2 = idxi[:p_i], idxi[p_i:]
                all_idx_set_1 = np.concatenate([all_idx_set_1, idxi_1])
                all_idx_set_2= np.concatenate([all_idx_set_2, idxi_2])
                
            idx1 = all_idx_set_1
            idx2 = all_idx_set_2

        else:
            p_n = int(p* self.nSamples)

            idx = random_rng.permutation(self.nSamples)
            idx1, idx2 = idx[:p_n], idx[p_n:]
        
        
        
        d1 = {k : v[idx1] for k,v in self.variablesDict.items()}
        d2 = {k : v[idx2] for k,v in self.variablesDict.items()}
        if self.y is None:
            y1 = None
            y2 = None
        else:
            y1 = self.y[idx1]
            y2 = self.y[idx2]
        if visitCounts:
            counts1 = {k : v[idx1] for k,v in visitCounts.items()}
            counts2 = {k : v[idx2] for k,v in visitCounts.items()}
        else:
            counts1, counts2 = None, None
        return Powerset(d1, y1,ignoreMaxNumberVariables = self.ignoreMaxNumberVariables), counts1, Powerset(d2, y2,ignoreMaxNumberVariables = self.ignoreMaxNumberVariables), counts2