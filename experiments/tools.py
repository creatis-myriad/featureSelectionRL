import numpy as np, pickle, types
import matplotlib.pyplot as plt

class DataResult:
    def __init__(self, metrics, DATA_MASK,stateCosts):
        self.DATA_MASK = DATA_MASK
        self.metrics_fct = metrics
        self.stateCost = stateCosts
        
        self.y =  {}
        self.yPred =  {}
        self.ks =  {}
        self.costs_usage =  {}
        self.metrics =  {}
        
        self.names = []
        
    def addName(self, name):
        self.metrics[name] = {m: [] for m in self.metrics_fct}
        self.y[name] = []
        self.yPred[name] =  []
        self.ks[name] =  []
        self.costs_usage[name] =  []
        self.names.append(name)
        
    def addResult(self, name, y, yPred, ks):
        if name not in self.names:
            self.addName(name)
            
        self.y[name].append(y)
        self.yPred[name].append(yPred)
        self.ks[name].append(ks)
        self.costs_usage[name].append(np.mean([self.stateCost[k & self.DATA_MASK] for k in ks]))
        for k, f in self.metrics_fct.items():
            self.metrics[name][k].append(f(y, yPred))
                
    def makeDumpable(self):
        self.metrics_fct = None # It can not dump lambda functions...
        
class DataResultList:
    def __init__(self, ts, metrics, DATA_MASK,stateCosts):
        self.ts = ts
        self.results = {t: DataResult(metrics,DATA_MASK, stateCosts) for t in ts}
        
    def join(self, other):
        if len(self.ts) != len (other.ts) or np.any(self.ts != other.ts):
            raise ValueError()
        for t in self.ts:
            other_t = other[t]
            for n in other[t].names:
                for i,_ in enumerate(other_t.y[n]):
                    self.results[t].addResult(n, other_t.y[i], other_t.yPred[i], other_t.ks[i])
        
    def dump(self, fileName):
        for r in self.results.values():
            r.makeDumpable()
        with open(fileName,'wb') as f:
            pickle.dump(self, f)
    @classmethod
    def read(self, fileName):
        with open(fileName,'rb') as f:
            d = pickle.load(f)
        return d

sigmoid = lambda z: 1/(1 + np.exp(-z))
def addPredictProba(c):
    """
    SVC with predict_proba, so it is consistent with the interface.
    
    Instead of using the sklearn predict_proba, that is not deterministic and has a lot of variability in small datasets, I simply use a sigmoid of the decision function.  
    WARNING: probabilities will not be well calibrated, just for accuracy computation / AUC.
    """
    def predict_proba(self, X):
        y = np.zeros((len(X), 2))
        y[:, 1] = sigmoid(self.decision_function(X))
        y[:, 0] = 1 -  y[:, 1]
        return y
    c.predict_proba = types.MethodType(predict_proba, c)
    return c

    
def flatten(t):
    return [item for sublist in t for item in sublist]


def multiplot(results,renameDict = {}, score = 'acc', nIts = 1):
    _, (f1, f2, f3, f4) = plt.subplots(figsize = (20, 4), ncols = 4)
    ts = results.ts
    plt.sca(f1)
    for p in results.results[ts[0]].names:
        mean_score_cv = np.array([np.mean(results.results[t].metrics[p][score]) for t in ts])
        std_score_cv = np.array([np.std(results.results[t].metrics[p][score]) for t in ts])
        plt.semilogx(ts, mean_score_cv, '-o', label =  p if p not in renameDict else renameDict[p])
        plt.fill_between(ts, mean_score_cv - std_score_cv/np.sqrt(nIts), mean_score_cv + std_score_cv/np.sqrt(nIts), alpha = .25)


    plt.legend()

    plt.xlabel('Cost-weight [$\lambda$]')
    plt.ylabel('Accuracy')
    plt.sca(f2)
    for p in results.results[ts[0]].names:
        mean_acc_cv = np.array([np.mean(results.results[t].costs_usage[p]) for t in ts])
        std_acc_cv = np.array([np.std(results.results[t].costs_usage[p]) for t in ts])
        plt.semilogx(ts, mean_acc_cv, '-o', label = p if p not in renameDict else renameDict[p])
        plt.fill_between(ts, mean_acc_cv - std_acc_cv/np.sqrt(nIts), mean_acc_cv + std_acc_cv/np.sqrt(nIts), alpha = .25)


    plt.legend()

    plt.xlabel('Cost-weight [$\lambda$]')
    plt.ylabel('Cost [Dollars]')

    plt.sca(f3)
    for p in results.results[ts[0]].names:
        mean_cost_cv = np.array([np.mean(results.results[t].costs_usage[p]) for t in ts])
        mean_score_cv = np.array([np.mean(results.results[t].metrics[p][score]) for t in ts])
        std_score_cv = np.array([np.std(results.results[t].metrics[p][score]) for t in ts])

        plt.plot(mean_cost_cv, mean_score_cv, '-o', label =  p if p not in renameDict else renameDict[p])
        plt.fill_between(mean_cost_cv, mean_score_cv - std_score_cv/np.sqrt(nIts), mean_score_cv + std_score_cv/np.sqrt(nIts), alpha = .5)
        #plt.fill_between(mean_cost_cv, mean_score_cv - std_score_cv, mean_score_cv + std_score_cv, alpha = .15)

    plt.legend()

    plt.xlabel('Cost [Dollars]')
    plt.ylabel('Accuracy')

    plt.sca(f4)
    for p in results.results[ts[0]].names:
        mean_cost_cv = np.array([np.mean(results.results[t].costs_usage[p]) for t in ts])
        mean_score_cv = np.array([np.mean(results.results[t].metrics[p][score]) for t in ts])

        plt.plot(ts, mean_score_cv - mean_cost_cv * ts, '-o', label = p if p not in renameDict else renameDict[p])

    plt.legend()
    plt.xlabel('Cost-weight [$\lambda$]')
    plt.ylabel('Accuracy - $\lambda$cost')

    plt.tight_layout()