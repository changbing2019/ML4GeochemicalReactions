#!/usr/bin/env python
# coding: utf-8

# ### this script is to test various regression models 

# This script is used to train ML models 


import pickle
import numpy as np
import pandas as pd
import os, glob
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

## The following are the ML models which can be used for trasinning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

from sklearn.preprocessing import MinMaxScaler,StandardScaler
import warnings
warnings.filterwarnings("ignore")


def outputModelTrainningScore(model,X,y,nCV=10):
    R2_score = cross_val_score(model, X, y,scoring='r2', cv=nCV)
    RMSE_score = cross_val_score(model, X, y,scoring='neg_root_mean_squared_error', cv=10)
    MAE_score = cross_val_score(model, X, y,scoring='neg_mean_absolute_error', cv=10) 
    R2_score_mean = R2_score.mean()
    RMSE_score_mean = RMSE_score.mean()
    MAE_score_mean = MAE_score.mean()
    return [R2_score_mean,RMSE_score_mean, MAE_score_mean] 

def GPyModel(xtrain,ytrain,n_dim,OutFP):
    length_scale = [1]*n_dim
    GPy_Model = GaussianProcessRegressor(kernel=Matern(length_scale=length_scale, nu=2.5), alpha = 1.0e-6,n_restarts_optimizer=9, normalize_y=True)
    #GPy_random = RandomizedSearchCV(estimator = GPy_Model, param_distributions = random_GPyModel, scoring = 'r2',n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
    GPy_Model.fit(xtrain, ytrain)        
    pickle.dump(GPy_Model, open(OutFP, 'wb'))             
    # Cross Validation
    scores = outputModelTrainningScore(GPy_Model,xtrain,ytrain,nCV=10)     
    return GPy_Model,scores


def main():

    ## Reading data prepared by Nick 

    ## Define the group datasets

    dataFolder = os.getcwd()
    file = '35_SCc_02_LHS_5000_54854_01_s1_G.csv' 
    InsFile = os.path.join(dataFolder, file)
    data = pd.read_csv(InsFile)
    data.columns =[col.strip() for col in data.columns]
    inputColNames =['CaO', 'SiO2', 'Al2O3', 'SO3', 'K2O', 'CO2', 'H2O']
    
    scaler =  StandardScaler().fit(data[inputColNames].values)
    trainData,testData = train_test_split(data,test_size=0.3, random_state=42,shuffle =True)
    ## The target variables you want to simulate
    # pH should be estiamted first because NSiO2 dpends on pH estimates for Group 2
    targetColumnNames = ['MassWater', 'Ca_aq', 'Si_aq', 'Al_aq', 'S_aq', 'K_aq', 'C_aq', 'Portlandite',
                          'AmorfSi', 'Gibbsite', 'Katoite', 'Monosulfate','Ettringite', 'Straetlingite',
                          'Chabazite', 'Calcite', 'Hemicarbonate', 'Monocarbonate', 'Thaumasite', 'mCSHQ']
    
    ## Save training infomation to a file
    modelSumaryFileName = 'ModelTrainSummary.csv'
    modelTestFileName = 'ModelTest.csv'

    #groupLst=[]
    varLst = []
    modelTypeLst = []
    trainScore = []
    dfTestResults = pd.DataFrame()
    outFolder = os.path.join(dataFolder,'SavedModel')
    if not os.path.exists(outFolder):
         os.makedirs(outFolder)
    
    for col in targetColumnNames: 
        # CHeck the col is in the dataFrame
        if col not in trainData.columns:
            sys.exit('The target column is not defined in the dataset, please check!')                
        
        print('Target==>',col)
        
        trainDataX=trainData[inputColNames];
        trainDataY = trainData[col]
        
        X = trainDataX.values
        Y = trainDataY.values
        fileName = os.path.normpath( os.path.join( outFolder,'GPyModel_'+col+'.sav' ) )             
        X_scaled = scaler.transform(X)  
        GPy,scores = GPyModel(X_scaled,Y,len(inputColNames),fileName)
        #groupLst.append(group)
        varLst.append(col)
        modelTypeLst.append('GPyModel')  
        trainScore.append(scores)
        
        testX = testData[inputColNames].values
        testX_scaled = scaler.transform(testX) 
        testY = testData[col].values
        predY,predYStd = GPy.predict(testX_scaled, return_std=True)
        CI_low = predY - 2*predYStd
        CI_upp = predY + 2*predYStd
        
        groupPredDF = pd.DataFrame({'testDataY':testY,
                                    'predDataY':predY,
                                    'predCI_low':CI_low,
                                    'predCI_upp':CI_upp,
                                   # 'group':[group]*len(testGroupData),
                                    'var':[col]*len(testData),
                                    'modelType':['GPY']*len(testData)}) 
        if len(dfTestResults)==0:
            dfTestResults = groupPredDF
        else:
            dfTestResults = pd.concat([dfTestResults,groupPredDF],axis=0)
        
        
    ModelTrainSummary = pd.DataFrame({'var':varLst,'modelType':modelTypeLst})
    ModelTrainMeasures = pd.DataFrame(trainScore,columns=['R2_mean','RMSE_mean','MAE_mean'])
    ModelTrainSummary = pd.concat([ModelTrainSummary,ModelTrainMeasures],axis=1)
    ModelTrainSummary.to_csv(modelSumaryFileName, index =False)
        
    dfTestResults.to_csv(modelTestFileName, index =False)
    
    
if __name__ == '__main__': main()






