import numpy as np, scipy, collections, sklearn, sklearn.decomposition
import pandas
import hierarchyrl.powerset

def readDataHypertension():
    data = scipy.io.loadmat(path)
    gls = (np.mean([data['Features'][i][0][0][0][0].T for i in range(-2, -8, -1)], axis = 0))
    dataProcessed = {}
    for i, _  in enumerate(data['Features']):
        name = data['Features'][i][0][0][0][1][0]
        if name != 'Temp Transf':
            X = data['Features'][i][0][0][0][0].T
            dataProcessed[name] =X
    dataProcessed['GLS'] = gls
    y = (data['Outcomes'] == 1).flatten() # data['Outcomes'] == 1  => Hypertense 

    dr =collections.defaultdict(lambda : sklearn.decomposition.PCA(n_components= 5))
    X = {}
    #X['lvm'] = lvm

    X['Aortic Valve'] = dr['Aortic Valve'].fit_transform(dataProcessed['Aortic Valve'])
    X['Mitral Valve'] =  dr['Mitral Valve'].fit_transform(dataProcessed['Mitral Valve'])
    X['LV Strain1'] =  dr['LV Strain1'].fit_transform(dataProcessed['LV Strain1'])
    X['GLS'] = dr['GLS'].fit_transform(dataProcessed['GLS'])
    X['TDI Septal'] = dr['TDI Septal'].fit_transform(dataProcessed['TDI Septal'])


    X_p = hierarchyrl.powerset.Powerset(X, y)

    costs = {}
    costs['Aortic Valve'] = 1.
    costs['Mitral Valve'] =  1.
    costs['LV Strain1'] = 10.
    costs['GLS'] = 5.
    costs['TDI Septal'] = 2.5
    return X_p, costs,{'dr' : dr, 'data' : data}

def readDataHypertensionCensored(path = './Data/hypertensionScores.npz'):
    """
    Read the PCA scores from a Python file, when there is no access to the original traces.
    """
    data = np.load(path)
    X = {}
    #X['lvm'] = lvm

    X['Aortic Valve'] =data['Aortic Valve']
    X['Mitral Valve'] =  data['Mitral Valve']
    X['LV Strain1'] =data['LV Strain1']
    X['GLS'] = data['GLS']
    X['TDI Septal'] = data['TDI Septal']


    X_p = hierarchyrl.powerset.Powerset(X, data['y'])

    costs = {}
    costs['Aortic Valve'] = 1.
    costs['Mitral Valve'] =  1.
    costs['LV Strain1'] = 10.
    costs['GLS'] = 5.
    costs['TDI Septal'] = 2.5
    return X_p, costs,{}


def parseICUClevelandICUDataset():
    cols = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'label']
    df1 = pandas.read_csv('Data/clevelandICU/processed.cleveland.data', header = None, na_values='?', names = cols)
    df1['center'] = 0
    df2 = pandas.read_csv('Data/clevelandICU/processed.va.data', header = None, na_values='?', names = cols)
    df2['center'] = 1
    df3 = pandas.read_csv('Data/clevelandICU/processed.switzerland.data', header = None, na_values='?', names = cols)
    df3['center'] = 2
    df4 = pandas.read_csv('Data/clevelandICU/processed.hungarian.data', header = None, na_values='?', names = cols)
    df4['center'] = 3
    df = pandas.concat([df1, df2, df3, df4]) 
    return df

def readDataCleveland():
    df = parseICUClevelandICUDataset()
    df = df.drop('thal', axis = 1).drop('ca', axis = 1).dropna()
    df = df.dropna()
    X = {}
    X['demographics'] = df[['age', 'sex']].values
    X['admission'] = df[['cp', 'trestbps', 'restecg']].values
    X['lab'] = df[['chol', 'fbs']].values
    X['exercise'] = df[['thalach', 'exang', 'oldpeak', 'slope']].values
    #X['contrastImaging'] = df[['thal', 'ca']].values

    X_p = hierarchyrl.powerset.Powerset(X, df['label'].values >= 1)
    
    costs = {}
    costs['demographics'] = 1.
    costs['admission'] =  15
    costs['lab'] = 7.27
    costs['exercise'] = 85.
    costs['contrastImaging'] = 100
    return X_p, costs, {}
