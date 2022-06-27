import pickle, numpy as np

class GraphWithVisits:
    def __init__(self, X_p, paths = None):
        if paths is None:
            paths = np.zeros(len(X_p.y), dtype = int)
        self.namesVariables = X_p.variablesNames
        self.nodes = {k: X_p.getVariablesFromEncoding(k) for k in X_p.getAllCombinations()}
        self.paths = paths
        
        # Population description
        self.X = X_p.X_full
        self.y = X_p.y
        self.indicesDataFromK = X_p.all_indices
        self.sizes = X_p.variablesSize
        
        # For each state, the subpopulations that has finished and the one that has finished there
        self.nodesVisited = {k : np.array([k in p  for i, p in enumerate(paths)]) for k in X_p.getAllCombinations()}
        self.nodesFinished = {k : np.array([p[-2] ==k  for i, p in enumerate(paths)]) for k in X_p.getAllCombinations()}

        #List of edges, and those of the population subviseted 
        self.edgesAll = []
        self.edgesVisited = {}

        for k in X_p.getAllCombinations():
            for a,_ in enumerate(self.namesVariables):
                k1 = k | (1 <<a)
                if k != k1:
                    self.edgesAll.append((k, k1))
                    self.edgesVisited[(k, k1)] = np.logical_and(self.nodesVisited[k], self.nodesVisited[k1])
                                                                
    def filterByVisits(self, th = 0):          
        self.nodesVisited = {k:v for k,v in self.nodesVisited.items() if np.sum(v) > th}
        self.nodesFinished = {k:v for k,v in self.nodesFinished.items() if (np.sum(v) > th) or k in self.nodesVisited}
        self.edgesVisited = {k:v for k,v in self.edgesVisited.items() if np.sum(v) > th}

    def getVariablesFromX(self, X, k):
        ks = X.getVariablesFromEncoding(k)
        r = {}
        for kk in ks:
            i = X.variablesNames.index(kk)
            r[kk] = X.getData(1<<i)
        return r
    
    def variableToEncoding(self, v):
        k = 0
        for vv in v:
            k |= (1 << self.namesVariables.index(vv))
        return k
    
    def encodingToVariable(self, k):
        return self.nodes[k]
    
    def dump(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self, f)
            
    @classmethod
    def load(cls, path):
        with open(path, 'rb') as f:
            v2 =pickle.load(f)
        return v2