# Script for generating Figure 1, showing the cost - accuracy trade off
import sklearn, sklearn.svm, sklearn.kernel_ridge, sklearn.datasets
import hierarchyrl.powerset, hierarchyrl.utils, hierarchyrl.policy2
import numpy as np, os, functools, itertools, argparse, collections, tqdm, dill
import matplotlib.pyplot as plt, multiprocessing

from experiments import dataReading, tools

def dice(k1, k2):
    intersection = len([k for k in k1 if k in k2])
    if len(k1) == 0. and len(k2) == 0.:
        return 1
    return 2 * intersection/(len(k1) + len(k2))


classifier = lambda: tools.addPredictProba(hierarchyrl.utils.PredictorStandardise(
                                            sklearn.svm.SVC(probability = False, C = 1., class_weight = 'balanced')
                                        )
                                    )
regressor = lambda: hierarchyrl.utils.PredictorStandardise(
                                sklearn.model_selection.GridSearchCV(
                                sklearn.kernel_ridge.KernelRidge(alpha = 1, kernel = 'rbf'), 
                                param_grid = {'alpha' : np.logspace(-2, 2, num =  5)}, 
                                    cv = sklearn.model_selection.KFold(n_splits=5, shuffle= True, random_state = 0)
                        ),
                        scaleY = True)

metrics_fcts = {}
metrics_fcts['acc'] = lambda y, y1: sklearn.metrics.accuracy_score(y, np.argmax(y1, axis = 1))

def doComputations(t, nReps, costs, stateCost, X_p_2, X_p_test):
    dataResultSameTestset = tools.DataResult(metrics_fcts, X_p_2.DATA_MASK, stateCost) 

    for i in range(nReps):
        X_p_train, _,  _, _ = X_p_2.split(.90, random_rng= np.random.RandomState(i))
        X_p_train_1, _, X_p_val, _ = X_p_train.split(.8)    


        policy = hierarchyrl.policy2.PolicyPowerset(X_p_train, 
                                    modelQAction = regressor,  
                                    modelClassification = classifier,
                                    loss='accuracy',
                                        acquisitionCost = {k:v *t for k,v in costs.items()}
                                    )
        policy.train(debug = False, nIts = 1,  offPolicyEpsilon = 0.5)

        yPred_dqn, ks, _ = policy.simulateEvaluateInPolicy( X_powerset= X_p_test)
        dataResultSameTestset.addResult('Reinforcement learning', X_p_test.y, yPred_dqn,ks)

        model = classifier()
        best_val_acc = -10000
        nVars = len(X_p_train.variablesNames)
        for v in itertools.chain(*[itertools.combinations(range(nVars), i) for i in range(0, nVars + 1)]):
            k = functools.reduce(lambda x, y: x | y, [1 << vv for vv in v], 0)
            if k != 0:
                model.fit(X_p_train_1.getData(k),X_p_train_1.y) 
                val_acc = np.mean(policy.loss(X_p_val.y, model.predict_proba(X_p_val.getData(k)))) - policy.stateCost[k]
            else: 
                yPred = np.repeat([1 - np.mean(X_p_train_1.y),  np.mean(X_p_train_1.y)] ,X_p_val.nSamples).reshape((2, -1)).T
                val_acc =  np.mean(policy.loss(X_p_val.y, yPred))
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_k = k
                
        if best_k != 0:
            model.fit(X_p_train.getData(best_k),X_p_train.y)
            yPred = model.predict_proba(X_p_test.getData(best_k))
        else:
            yPred = np.repeat([1 - np.mean(X_p_train.y),  np.mean(X_p_train.y)] ,X_p_test.nSamples).reshape((2, -1)).T
        dataResultSameTestset.addResult('Populationwise feature selection', X_p_test.y, yPred, [best_k])

    meanDice = collections.defaultdict(list)
    for i,j in itertools.combinations( range(nReps), 2):
        if i == j:
            continue
        for algorithm in dataResultSameTestset.ks:
            nSamples = len(dataResultSameTestset.ks[algorithm][i])
            mean = np.mean([dice(X_p_test.getVariablesFromEncoding(dataResultSameTestset.ks[algorithm][i][k]), 
                                X_p_test.getVariablesFromEncoding(dataResultSameTestset.ks[algorithm][j][k]))
                            for k in range(nSamples)])
            meanDice[algorithm].append(mean)
    return meanDice

if __name__ == '__main__':
    # Parameters
    parser = argparse.ArgumentParser(description= 'Figure 2 of the paper, experiment in which the reproducibility of the modalities chosen by the policy is evaluated.')
    parser.add_argument('-output', default = './FiguresMICCAI')
    parser.add_argument('-nIts', default = 20, type =int)
    parser.add_argument('-nReps', default = 3, type =int)
    parser.add_argument('-numTsamples', default = 10, type = int)
    parser.add_argument('--parallel', action = 'store_true', default = False)
    parser.add_argument('--ignore-cleveland', action = 'store_true', default = False)
    parser.add_argument('--readOnly', action = 'store_true', default = False)


    args = parser.parse_args()

    num_t_samples = args.numTsamples
    nIts = args.nIts
    nReps = args.nReps
    resultPath = args.output

    # Data reading
    all_costs = {}
    all_data = {}
    if not args.ignore_cleveland:
        all_data['Cleveland'], all_costs['Cleveland'], _ = dataReading.readDataCleveland()
    all_data['Hypertension'], all_costs['Hypertension'], _ =  dataReading.readDataHypertensionCensored()
    #
    if args.readOnly:
        results = {}
        allDiceByDataset = {}
        for dataset in all_costs:
            costs = all_costs[dataset]
            X_p = all_data[dataset]
            allDice=  collections.defaultdict(lambda: collections.defaultdict(list))
            allDiceByDataset[dataset] = allDice

            ts = np.logspace(-3, -1, num = num_t_samples)
            policyTest = hierarchyrl.policy2.PolicyPowerset(X_p, 
                                    modelQAction = regressor,  
                                    modelClassification = classifier,
                                    loss='accuracy',
                                        acquisitionCost = {k:v for k,v in costs.items()}
                                    )


            results[dataset] = tools.DataResultList(ts, metrics_fcts, X_p.DATA_MASK, policyTest.stateCost) 

            for ii in tqdm.tqdm(range(nIts)):
                X_p_2, _,  X_p_test, _ = X_p.split(.90, random_rng= np.random.RandomState(ii))
                if not args.parallel:
                    for t in ts:
                        meanDice = doComputations(t, nReps, costs, policyTest.stateCost, X_p_2, X_p_test)
                        for algorithm in meanDice:
                            allDice[algorithm][t].append(meanDice[algorithm])
                else:
                    with multiprocessing.Pool() as pool:
                        rs = pool.starmap(doComputations,  zip(ts,
                                                            itertools.repeat(nReps),  itertools.repeat(costs), itertools.repeat(policyTest.stateCost),
                                                            itertools.repeat(X_p_2), itertools.repeat(X_p_test)))
                    for t, meanDice in zip(ts, rs):
                        for algorithm in meanDice:
                            allDice[algorithm][t].append(meanDice[algorithm])
    else:
        with open(os.path.join(resultPath, 'exp2.pkl'), 'rb' ) as f:
            allDiceByDataset = dill.load(f)

    # Save data and generate figures


    score = 'acc'
    _, fs = plt.subplots(ncols = max(len(allDiceByDataset), 2), figsize = (12, 2.5))

    for i, (k, allDice) in enumerate(allDiceByDataset.items()):
        plt.sca(fs[i])
        maxCosts = []
        minCosts = []

        for algo in allDice:
            if algo == 'NIPS - 2015':
                continue
            meanCost = np.array([ np.mean(allCosts[k][algo][t]) for t in ts])
            maxCosts.append(np.max(meanCost))
            minCosts.append(np.min(meanCost))
            mean = np.array([ np.mean(allDice[algo][t]) for t in ts])
            std = np.array([ np.std(allDice[algo][t]) for t in ts])
            line = plt.plot(meanCost, mean, '-o', label = algo)
            plt.fill_between(meanCost, mean - std, mean + std, alpha = .15, color = line[0].get_color())
            plt.fill_between(meanCost, mean - 1.96/np.sqrt(nIts) *std, mean + 1.96/np.sqrt(nIts) *std, alpha = .45, color = line[0].get_color())

        plt.ylabel('Dice Score', fontsize = 12)
        plt.xlabel('Average cost [Arbitrary unit]', fontsize = 12)
        plt.xlim(max(minCosts), min(maxCosts))
        plt.legend(loc = 4, fontsize = 12)
        if k == 'Hypertension':
            k = 'Hypertense'
        elif k == 'Cleveland':
            k = 'Heart Disease'

        plt.title(k, y = 1, pad = -14, fontsize = 14, bbox = dict(boxstyle='round', facecolor='lightgray', alpha=0.85))

        plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig('FiguresMICCAI/exp2.pdf')
    
    # Save data
    with open(os.path.join(resultPath, 'exp2.pkl'), 'wb' ) as f:
        dill.dump(allDiceByDataset, f)