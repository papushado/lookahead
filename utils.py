import time
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold

"""
Binary Grid Search Code Adapted from
https://github.com/mattalhonte/binary-grid-search
"""

    
def vprint(v, *args):
    if v: print(*args)
        
def get_l2_reg(model, alpha, n):
    assert alpha>=0
    l2_reg = 0.
    if alpha > 0:
        for name, prm in model.named_parameters():
            if 'weight' in name:
                l2_reg = l2_reg + (prm ** 2).sum()/n
    return l2_reg

def get_l1_reg(model, alpha, n):
    assert alpha>=0
    l2_reg = 0.
    if alpha > 0:
        for name, prm in model.named_parameters():
            if 'weight' in name:
                l2_reg = l2_reg + prm.abs().sum()/n
    return l2_reg

def null(*args, **kwargs):
    return 0.


def setSearchStepAndArgs(lowerArg, upperArg, decimals):
    #Some hyperparameters only accept ints
    if decimals==0:
        return np.array([(upperArg-lowerArg)/2, 
                         lowerArg, 
                         upperArg],
                        dtype=np.int)
    else:
        return (np.round((upperArg-lowerArg)/2, 
                         decimals),
                lowerArg,
                upperArg)
        

def compareVals(X, y, model, params, var, decimals, newArg, lastArg, lastVal, timesAndScores):
    if np.abs(newArg-lastArg)<=10**(-decimals):
        return pd.DataFrame(timesAndScores)
    
    if lastArg>newArg:
        lowerArg = newArg
        upperArg = lastArg
        
        
        searchStep, lowerArg, upperArg = setSearchStepAndArgs(lowerArg, upperArg, decimals)

        start_time = time.perf_counter()
        lowerVal = model(X, 
                         y,
                         {**params,
                          **{var:lowerArg}})
        end_time = time.perf_counter()
        run_time = end_time - start_time
        timesAndScores = timesAndScores + [{var: lowerArg,
                                           "score": lowerVal,
                                           "time": run_time}]
        
        upperVal = lastVal
        
    if newArg>lastArg:
        
        lowerArg = lastArg
        upperArg = newArg
        
        
        searchStep, lowerArg, upperArg = setSearchStepAndArgs(lowerArg, upperArg, decimals)
        

        
        lowerVal = lastVal
        
        start_time = time.perf_counter()
        upperVal = model(X, 
                         y,
                         {**params, 
                          **{var:upperArg}}, 
                         )
        end_time = time.perf_counter()
        run_time = end_time - start_time
        timesAndScores = timesAndScores + [{var: upperArg,
                                           "score": upperVal,
                                           "time": run_time}]
        
    if lowerVal==upperVal:
        return pd.DataFrame(timesAndScores)
        
    if lowerVal>upperVal:
        return compareVals(X, 
                           y, 
                           model,
                           params,
                           var,
                           decimals,
                           upperArg - searchStep,
                           lowerArg,
                           lowerVal, 
                           timesAndScores)
       
    if upperVal>lowerVal:
        return compareVals(X, 
                           y, 
                           model,
                           params, 
                           var,
                           decimals,
                           lowerArg + searchStep,
                           upperArg,
                           upperVal, 
                           timesAndScores)

    
def compareValsBaseCase(X, y, model, params, var, decimals, lowerArg, upperArg):
    """Run the binary search
    
    Parameters
    ----------
    X : NumPy array
        Training data
    y : 1d NumPy array
        Training data answers
    model : function
        A function that takes in some arguments and returns a metric for an ML algo
    params : dictionary
        Parameters for our ML algo that we want in every run
    var: string
        The parameter we want to optimize
    decimals: int
        How many decimals of difference we want between test values of the parameter
        For instance, some things have to be whole numbers, but others are floats
    lowerArg: int or float
        Lower limit to search
    upperArg: int or float
        Upper limit to search
        
    Returns
    -------
    pandas.DataFrame
        Contains the values that were tested, the performance, and the time it took
        to run
    
    """
    
    searchStep, lowerArg, upperArg = setSearchStepAndArgs(lowerArg, upperArg, decimals)


    timesAndScores = []
    
    start_time = time.perf_counter()
    lowerVal = model(X, 
                     y,
                     {**params,
                      **{var:lowerArg}},
                     )
    end_time = time.perf_counter()
    run_time = end_time - start_time
    timesAndScores = timesAndScores + [{var: lowerArg,
                                        "score": lowerVal,
                                        "time": run_time}]

    start_time = time.perf_counter()
    upperVal = model(X, 
                     y,
                     {**params, 
                      **{var:upperArg}}, 
                     )
    end_time = time.perf_counter()
    run_time = end_time - start_time
    timesAndScores = timesAndScores + [{var: upperArg,
                                        "score": upperVal,
                                        "time": run_time}]
    
    if lowerVal==upperVal:
        return pd.DataFrame(timesAndScores)
    
    if lowerVal>upperVal:
        return compareVals(X, 
                           y, 
                           model,
                           params, 
                           var,
                           decimals,
                           upperArg - searchStep,
                           lowerArg,
                           lowerVal, 
                           timesAndScores)
    
    if upperVal>lowerVal:
        return compareVals(X, 
                           y,  
                           model,
                           params,  
                           var,
                           decimals,
                           lowerArg + searchStep,
                           lowerArg,
                           lowerVal, 
                           timesAndScores)
    

def line_search(x, y, w, prm_str, score,
                n_splits=3, lo=0., hi=0.5, decim=3,
                random_state=None):
    best_prms = []
    kf = KFold(n_splits=n_splits, random_state=random_state)
    for trn_idx, val_idx in kf.split(x):
        x_trn, y_trn, w_trn = x[trn_idx,:], y[trn_idx], w[trn_idx]
        x_val, y_val, w_val = x[val_idx,:], y[val_idx], w[val_idx]
        prms = {'w_trn':w_trn, 'w_val':w_val, 'x_val':x_val, 'y_val':y_val}
        tbl = compareValsBaseCase(x_trn, y_trn, score, prms, prm_str,
                                    decim, lo, hi) #hack to input seed
        best_prm_k = tbl[prm_str].iloc[-1]
        best_prms.append(best_prm_k)
    best_prm = np.mean(np.asarray(best_prms)) 
    return best_prm