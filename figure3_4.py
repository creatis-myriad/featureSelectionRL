# Script for generating Figure 1, showing the cost - accuracy trade off
import sklearn, sklearn.svm, sklearn.kernel_ridge, sklearn.datasets
import hierarchyrl.powerset, hierarchyrl.utils, hierarchyrl.policy2
import experiments.graphStructure, experiments.dataReading
import numpy as np, os, argparse
import matplotlib.pyplot as plt

from experiments import tools
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

def getVariablesFromX(X, k):
    ks = X.getVariablesFromEncoding(k)
    r = {}
    for kk in ks:
        i = X.variablesNames.index(kk)
        r[kk] = X.getData(1<<i)
    return r

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description= 'Figure 3 and 4 of the paper. Train a policy, and evaluate it in the training set, keeping track of the sequence of decisions for each individual.')
    parser.add_argument('-output', default = './FiguresMICCAI')
    parser.add_argument('--plot-traces', action = 'store_true', default = False, help = 'Generate the trace figures, in case the signal data is available.')
    
    args = parser.parse_args()

    if args.plot_traces:
        X_p_hta, costs_hta, data_hta = experiments.dataReading.readDataHypertension()
    else:
        X_p_hta, costs_hta, data_hta = experiments.dataReading.readDataHypertensionCensored()

    policy5em2 = hierarchyrl.policy2.PolicyPowerset(X_p_hta, 
                                        modelQAction = regressor,  
                                        modelClassification = classifier,
                                        loss='accuracy',
                                            acquisitionCost = {k:v *5e-2 for k,v in costs_hta.items()}
                                        )

    policy5em2.train(debug = False, nIts = 1,  offPolicyEpsilon = 0.5)
    yPred_dqn, ks, paths5em2 = policy5em2.simulateEvaluateInPolicy( X_powerset= X_p_hta)

    v = experiments.graphStructure.GraphWithVisits(X_p_hta, paths5em2)
    v.dump(os.path.join(args.output, 'paths5em2Graph.pkl'))


    yPred_dqn, ks, paths5em2 = policy5em2.simulateEvaluateInPolicy( X_powerset= X_p_hta)


    # Generate the traces (Figure 4)
    if  args.plot_traces:
        _, grid = plt.subplots(nrows = 3, ncols = 4, gridspec_kw={'width_ratios': [1,.025, 1, 1]}, figsize = (18, 8))
        j = 0
        ylims = {}
        ylims['Mitral Valve'] = (0, 100)
        ylims['Aortic Valve'] = (0, 140)
        ylims['GLS'] = (-25, 5)

        for k in np.unique(ks):
            if np.sum(ks ==k) < 10:
                continue
            idx = ks ==k
            kOrig = k
            k = X_p_hta.getEncodingFromVariables(['Mitral Valve', 'Aortic Valve', 'GLS'])
            X = getVariablesFromX(X_p_hta, k)
            # _, fs = plt.subplots(ncols = len(X), nrows = 1,figsize = (5 * len(X), 4))
            if len(X) == 1:
                fs = [fs]
            label = X_p_hta.y[idx]
            if True:
                plt.sca(grid[j, 1])
                plt.axis('off')
                
                for i,d in enumerate(['Mitral Valve', 'Aortic Valve', 'GLS']):
                    x = X[d]
                # plt.sca(fs[i])
                    plt.sca(grid[j, i + (1 if i !=0 else 0)])
                    mean1 =np.mean( data_hta['dr'][d].inverse_transform(x[idx][label ==1]), axis = 0)
                    std1 =np.std( data_hta['dr'][d].inverse_transform(x[idx][label ==1]), axis = 0)
                    mean0 =np.mean( data_hta['dr'][d].inverse_transform(x[idx][label ==0]), axis = 0)
                    std0 =np.std( data_hta['dr'][d].inverse_transform(x[idx][label ==0]), axis = 0)
                    plt.plot(mean0, label = 'CTRL')
                    plt.fill_between(np.arange(len(mean0)), mean0 - std0, mean0 + std0, alpha =.25)
                    plt.plot(mean1, label = 'Hypertense')
                    plt.fill_between(np.arange(len(mean0)),mean1 - std1, mean1 + std1, alpha =.25)

                    meanCTRL =np.mean( data_hta['dr'][d].inverse_transform(x[X_p_hta.y == 0.]), axis = 0)
                    meanHTA =np.mean( data_hta['dr'][d].inverse_transform(x[X_p_hta.y == 1.]), axis = 0)

                    #plt.plot(meanCTRL, label = 'Mean CTRL [Full population]', color = 'k')

                    if d == 'Aortic Valve':
                        plt.axvline(np.argmax(mean1), c = 'orange')
                        plt.axvline(np.argmax(mean0))
                        plt.ylabel('$v_{blood}$ [mm/s]', fontsize = 14)
                        cycle = 'Systole'
                    elif d == 'Mitral Valve':
                        plt.ylabel('$v_{blood}$ [mm/s]', fontsize = 14)
                        cycle = 'Diastole'
                    elif d in ['LV Strain1', 'GLS', 'TDI Septal']:
                        print(d, np.argmin(mean0), np.argmin(mean1))
                        plt.ylabel('Strain [%]', fontsize = 14)
                        cycle = 'Full cardiac cycle'
                    #plt.title(d, fontsize = 12)
                    if j == 2:
                        plt.xlabel(f'Time [{cycle}]', fontsize = 14)
                    plt.gca().axes.xaxis.set_ticklabels([])
                    plt.ylim(*ylims[d])
                    #plt.legend()
        #        plt.suptitle(f'{str(X_p_hta.getVariablesFromEncoding(kOrig))} Proportion HTA = {np.mean(label):.2f}, N = {np.sum(idx)}')#', HR_HTA  = {np.nanmean(hr_idx[label ==1]):.1f}, HR CTRL  = {np.nanmean(hr_idx[label ==0]):.1f}')
            j += 1
        plt.tight_layout(w_pad = 2)
        plt.savefig(os.path.join(args.output, 'policyDecisions.pdf'))