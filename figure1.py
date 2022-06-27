# Script for generating Figure 1, showing the cost - accuracy trade off
import sklearn, sklearn.svm, sklearn.kernel_ridge, sklearn.datasets
import hierarchyrl.powerset, hierarchyrl.utils, hierarchyrl.policy2
import numpy as np, os, functools, itertools, argparse
import matplotlib.pyplot as plt
import tqdm, multiprocessing

from experiments import dataReading, tools
# Parameters

classifier = lambda: tools.addPredictProba(hierarchyrl.utils.PredictorStandardise(
                                        sklearn.model_selection.GridSearchCV(
                                            sklearn.svm.SVC(probability = False, C = 1., class_weight = 'balanced'),
                                            param_grid = {'C' : np.logspace(-2, 2, num =  5)}, 
                                            cv = sklearn.model_selection.StratifiedKFold(n_splits=5, shuffle= True, random_state = 0)
                                        )
                                    ))
classifier = lambda: tools.addPredictProba(hierarchyrl.utils.PredictorStandardise(
                                            sklearn.svm.SVC(probability = False, C = 1., class_weight = 'balanced')
                                    ))
regressor = lambda: hierarchyrl.utils.PredictorStandardise(
                                sklearn.model_selection.GridSearchCV(
                                sklearn.kernel_ridge.KernelRidge(alpha = 1, kernel = 'rbf'), 
                                param_grid = {'alpha' : np.logspace(-2, 2, num =  5)}, 
                                    cv = sklearn.model_selection.KFold(n_splits=5, shuffle= True, random_state = 0)
                        ),
                        scaleY = True)

def doComputation(X_p_train, X_p_test, costs, t):
    res = {}
    X_p_train_1, _, X_p_val, _ = X_p_train.split(.8)    

    policy = hierarchyrl.policy2.PolicyPowerset(X_p_train, 
                                   modelQAction = regressor,  
                                   modelClassification = classifier,
                                   loss='accuracy',
                                    acquisitionCost = {k:v *t for k,v in costs.items()}
                                  )
    policy.train(debug = False, nIts = 1,  offPolicyEpsilon = 0.5)

    yPred_dqn, ks, _ = policy.simulateEvaluateInPolicy( X_powerset= X_p_test)

    res['Reinforcement learning'] =  (X_p_test.y, yPred_dqn,ks)

    model = classifier()
    best_val_acc = -10000
    nVars = len(X_p_train.variablesNames)
    for v in itertools.chain(*[itertools.combinations(range(nVars), i) for i in range(0,nVars + 1)]):
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
    res['Populationwise feature selection'] =  (X_p_test.y, yPred, [best_k])
    return res

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description= 'Figure 1 of the paper, experiment in which the cost-accuracy trade-off is evaluated.')
    parser.add_argument('-output', default = './FiguresMICCAI')
    parser.add_argument('-nIts', default = 40, type =int)

    parser.add_argument('-numTsamples', default = 20, type = int)
    parser.add_argument('--ignore-cleveland', action = 'store_true', default = False)
    parser.add_argument('--parallel', action = 'store_true', default = False)
    parser.add_argument('--readOnly', action = 'store_true', default = False)

    args = parser.parse_args()

    num_t_samples = args.numTsamples
    nIts = args.nIts
    resultPath = args.output

    # Data reading
    all_costs = {}
    all_data = {}
    if not args.ignore_cleveland:
        all_data['Heart Disease'], all_costs['Heart Disease'], _ = dataReading.readDataCleveland()
    all_data['Hypertension'], all_costs['Hypertension'], _ =  dataReading.readDataHypertensionCensored()
    #


    metrics_fcts = {}
    metrics_fcts['acc'] = lambda y, y1: sklearn.metrics.accuracy_score(y, np.argmax(y1, axis = 1))
    results = {}
    if args.readOnly:
        for dataset in all_costs:
            costs = all_costs[dataset]
            X_p = all_data[dataset]

            ts = np.logspace(-3, -1, num = num_t_samples)
            policyTest = hierarchyrl.policy2.PolicyPowerset(X_p, 
                                    modelQAction = regressor,  
                                    modelClassification = classifier,
                                    loss='accuracy',
                                        acquisitionCost = {k:v for k,v in costs.items()}
                                    )


            results[dataset] = tools.DataResultList(ts, metrics_fcts, X_p.DATA_MASK, policyTest.stateCost) 

            for i in tqdm.tqdm(range(nIts)):
                X_p_train, _,  X_p_test, _ = X_p.split(.90, random_rng = np.random.RandomState(i))
                #X_p_test = X_p_train

                # If not parallel
                if not args.parallel:
                    for t in ts:
                        res = doComputation(X_p_train, X_p_test, costs, t)
                        for k, r in res.items():
                            results[dataset].results[t].addResult(k, r[0], r[1], r[2])
                # If parallel
                else:
                    with multiprocessing.Pool() as pool:
                        rs = pool.starmap(doComputation,  zip(itertools.repeat(X_p_train), itertools.repeat(X_p_test), itertools.repeat(costs), ts))
                    for t, res in zip(ts, rs):
                        for k, r in res.items():
                            results[dataset].results[t].addResult(k, r[0], r[1], r[2])

        # Save data and figures
        for dataset, r in results.items():
            r.dump(os.path.join(resultPath, 'exp1_' + dataset + '.pkl'))
    else:
        for dataset,in all_data:
            results[dataset] = tools.DataResultList.read(os.path.join(resultPath, 'exp1_' + dataset + '.pkl'))


    score = 'acc'
    _, fs = plt.subplots(ncols = max(2, len(results)), figsize = (12, 3))
    fs = fs.flatten()
    for i, (k, r) in enumerate(results.items()):
        plt.sca(fs[i])
        maxCosts = []
        minCosts = []
        for p in r.results[r.ts[0]].names:
            if p == 'NIPS - 2015':
                continue
                
            mean_cost_cv = np.array([np.mean(r.results[t].costs_usage[p]) for t in r.ts])
            mean_score_cv = np.array([np.mean(r.results[t].metrics[p][score]) for t in r.ts])
            std_score_cv = np.array([np.std(r.results[t].metrics[p][score])/np.sqrt(nIts) for t in r.ts])
            std_score_cv_orig = np.array([np.std(r.results[t].metrics[p][score]) for t in r.ts])
            maxCosts.append(np.max(mean_cost_cv))
            minCosts.append(np.min(mean_cost_cv))

            line = plt.plot(mean_cost_cv, mean_score_cv, '-o', label =  p)
            plt.fill_between(mean_cost_cv, mean_score_cv - 1.96*std_score_cv, mean_score_cv + 1.96*std_score_cv, alpha = .45, color = line[0].get_color())
            plt.fill_between(mean_cost_cv, mean_score_cv - std_score_cv_orig, mean_score_cv + std_score_cv_orig, alpha = .10, color = line[0].get_color())

        plt.legend(loc = 4, fontsize = 12)
        plt.xlim(max(minCosts), min(maxCosts))

        plt.xlabel('Average cost [Arbitrary unit]', fontsize = 12)
        plt.ylabel('Accuracy', fontsize =12)
        plt.title(k, y = 1, pad = -14, fontsize = 14, bbox = dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
    plt.tight_layout()

    plt.savefig(os.path.join(resultPath, 'exp1.pdf'))
