 # Hierarchy reinforcement learning
   
   This document contains instructions on how to reproduce the results
   of the submitted paper "Reinforcement learning for active modality
   selection during diagnosis", as well as a small description of the
   different files and code on the implementation of the reinforcement
   learning method for selecting modalities during diagnosis proposed in
   the manuscript.
   
   ## Setup 
   To install a conda environment to reproduce the experiments or try the framework with new data, use the following instructions:
   ```
 conda create -n hierarchyRL --file requirements.txt 
 conda activate hierarchyRL pip install -e .
 ``` 
   The principal dependencies are the scikit-learn and numpy libraries, and dash and matplotlib for visualisation.
   
   ## Data
   We use the public Heart Disease dataset (including the different centers), a copy of which can be found in our data folder.
   The data can be also found in https://archive.ics.uci.edu/ml/datasets/heart+disease.
   
   For the private hypertense dataset, we don't have the right to publicly share
   the raw images / traces, but we include the PCA scores of the
   processed data, which are enough for reproducing all the figures
   except Figure 4. 
   
   ## Hardware environment and computational footprint:
   
   - The experiments had an approximate maximal memory footprint of 500 MB
   - The training time of a policy is ~15s the proposed datasets, in a MacBook Pro 2020 (2 GHz Quad-Core Intel Core i5)
   
   ## Code description:
   
   #### src/hierarchyRL 
   The main classes are *Powerset( (in  **`hierarchyrl.powerset.py`** )  and policy (in **`hierarchyrl.powerset.py`**  ), which include respectively the handling of the powerset of measurements, and the training and
   prediction of the policy.
   
   *Powerset* is a class used to obtain different subsets of features. It is initialised by a dictionary including the measurement names and
   its values, of ```{modalityName: measurements} ```, where
   measurements is a numpy array of size (```Nsamples x
   dimensionality```). Each subset is identified by an integer “k”,
   where each bit of its binary representation indicates whether a
   modality is present or not in the subset. In the powerset are also
   encoded the classification estimators at each subset, that take the
   decision using the available data.
   
   The policy class includes the training of the state-action value
   estimators, as well as the predictors. The training is done in the
   *train* function, which calls the *train_iteration*  function that includes the ordered visit of the superstates.
   *simulateEvaluateInPolicy* function evaluates the policy in a new dataset, acquiring at each step a modality as proposed by the policy.
   For each individual it returns the final prediction, the final
   superstate, indicating the acquired modalities, and also the sequence
   of decisions taken. 
   
   **Note:** our formulation allows the use and estimation of sample weights, associated to each individual and superstate (combination of
   measurements), representing the probability that the individual
   reaches such superstate, even if it is not described nor used in the
   paper. The idea is to use weighted classifiers/value estimators at
   each superstate, assigning higher weight to the samples that are
   likely to visit it. It can be implemented by iteratively training the
   policy, and then evaluating the policy to . Our preliminary
   experiments have not found an influence in the result, but further
   examinations are needed  to confirm it and they belong to future
   work. To avoid their use, simply keep the argument nIts  = 1 in the
   train function.  The *offPolicyEpsilon* argument of the train
   function is also related to the sample weights, and has to be ignored
   for this version.
   **Note 2:** The hyperparemeter $\lambda$ in the manuscript corresponds to $t$ in the code 
   
   ####  Experiments 
   Utilities for experiments, mainly data reading and saving results of executions and visualisation
   
   - *dataReading* : utilities for loading datasets and generating *Powerset* classes that can be used to train a policy
   - *tools* Data structures to store results of experiments 1 and 2, trying multiple $\lambda$ and samples of train/test splits with
   different algorithms.
   - *graphStructure* Given paths of actions, generates a graph counting the number of individuals that reach every combination of
   measurements, and also of the actions that are taken from every node.
   
   - *viewGraphFromFile* dash application that allows visualising the graphs generated with the previous class.
   
   
   ## Reproduction of experiments and figures
   
   ###  Figures 1 and 2
   
   To generate the first two figures, run the following commands: ```
   python exp1.py --parallel python exp2.py --parallel ```.  Given the
   high computational time, since it needs to test different seeds for
   the train/test split, and different values of $\lambda$ , we provide
   precomputed results stored in the pkl format. You can use the
   –readOnly flag to use that data to generate the plots.
   
   ###  Figures 3 and 4
   
   Figures 3 and 4, regarding the interpretation of a policy in the hypertense dataset, are lighter to compute, and can be done with thecommand: ``` python exp3_4.py ``` And, to visualise the resulting graph (Figure 3), we used dash and cytoscape. A small web app that reads the results of the execution paths (generated in the previous experiment, exp3_4) is provided.  Use the following instruction to start the dash server: 
  ``` 
   python experiments/viewGraphFromFile.py FiguresMICCAI/paths5em2Graph.pkl –showOnlyVisited 
  ``` 
  After running the instruction, you will  need to open a browser and visit the url localhost:8050 to find the visual representation of the graph.
   
   Figure 3 and 4 were edited to add a colour bar and annotations respectively.
