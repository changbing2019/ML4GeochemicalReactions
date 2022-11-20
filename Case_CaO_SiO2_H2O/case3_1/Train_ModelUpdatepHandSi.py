#!/usr/bin/env python
# coding: utf-8

# ### this script is to test various regression models 

# This script is used to train ML models 


import pickle
import numpy as np
import pandas as pd
import os, glob
import seaborn as sns
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold, cross_validate
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

## The following are the ML models which can be used for trasinning
from sklearn.model_selection import RandomizedSearchCV,GridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import MinMaxScaler,StandardScaler
import scipy.stats as stats  #truncnorm, randint
from sklearn.utils.fixes import loguniform
import timeit
import warnings
warnings.filterwarnings("ignore")


def outputModelTrainningScore(model,X,y,nCV=10):

    R2_score = cross_val_score(model, X, y,scoring='r2', cv=nCV)
    RMSE_score = cross_val_score(model, X, y,scoring='neg_root_mean_squared_error', cv=10)
    R2_score_mean = R2_score.mean()
    R2_score_std = R2_score.std()
    RMSE_score_mean = RMSE_score.mean()
    RMSE_score_std = RMSE_score.std()
    return [R2_score_mean,R2_score_std,RMSE_score_mean, RMSE_score_std] 
    
    



def randomforestModel(xtrain,ytrain,OutFP):
    
    
    model_rf = RandomForestRegressor(n_estimators=100, max_depth=30, random_state=2,bootstrap =True)
    model_rf.fit(xtrain, ytrain)        
    final_model_rf = model_rf #rf_random.best_estimator_
    # Save the model to the local drive
    pickle.dump(final_model_rf, open(OutFP, 'wb'))             
    # Cross Validation
    scores = outputModelTrainningScore(final_model_rf,xtrain,ytrain,nCV=10)     
    return scores

def xgbModel(xtrain,ytrain,OutFP):
    
    
    xgb=XGBRegressor(max_depth=3,n_estimators=1020,min_child_weight=1,scale_pos_weight=1,
               gamma=1,learning_rate=0.05,objective='reg:squarederror',
               colsample_bytree=1.0,random_state=500,seed=100)
    xgb.fit(xtrain, ytrain)        
    # Select the best model
    final_model_xgb = xgb
    # Save the model to hard drive
    pickle.dump(final_model_xgb, open(OutFP, 'wb'))             
    # Cross Validation
    scores = outputModelTrainningScore(final_model_xgb,xtrain,ytrain,nCV=10)     
    return scores
    
def lgbModel(xtrain,ytrain,OutFP):
    
    lgbmodel = lgb.LGBMRegressor(objective='regression',num_leaves=50,
                          learning_rate=0.1, n_estimators=700)
    
    lgbmodel.fit(xtrain, ytrain)        
    # Save the model to hard drive
    pickle.dump(lgbmodel, open(OutFP, 'wb'))             
    # Cross Validation
    scores = outputModelTrainningScore(lgbmodel,xtrain,ytrain,nCV=10)     
    return scores



def gboostModel(xtrain,ytrain,OutFP):
    
    
    gboost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                               max_depth=8, max_features='sqrt',
                               min_samples_leaf=25, min_samples_split=10, 
                               loss='huber', random_state =5)
    
    gboost.fit(xtrain, ytrain)        
    # Select the best model
    final_model_gb = gboost
    # Save the model to hard drive
    pickle.dump(final_model_gb, open(OutFP, 'wb'))             
    # Cross Validation
    scores = outputModelTrainningScore(final_model_gb,xtrain,ytrain,nCV=10)     
    return scores



def GPyModel(xtrain,ytrain,OutFP):
    
    GPy_Model = GaussianProcessRegressor(kernel=Matern(length_scale=[1,1,1], nu=2.5),alpha=10e-8, n_restarts_optimizer=9, normalize_y=True)
    #GPy_random = RandomizedSearchCV(estimator = GPy_Model, param_distributions = random_GPyModel, scoring = 'r2',n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
    GPy_Model.fit(xtrain, ytrain)        
    pickle.dump(GPy_Model, open(OutFP, 'wb'))             
    # Cross Validation
    scores = outputModelTrainningScore(GPy_Model,xtrain,ytrain,nCV=10)     

    return scores


def MLPModel(xtrain,ytrain,OutFP):

    hidden_layer_sizes =(10,10,10)
    MLPModel = MLPRegressor( hidden_layer_sizes=hidden_layer_sizes,activation ='tanh',solver = 'adam', alpha = 0.0001,
                               tol = 1e-5,max_iter = 1000, learning_rate="adaptive",random_state =5)
    
    MLPModel.fit(xtrain, ytrain)        
    # Select the best model
    final_model_MLP = MLPModel
    # print(final_model_gb)
    # Save the model to hard drive
    pickle.dump(final_model_MLP, open(OutFP, 'wb'))             
    # Cross Validation
    scores = outputModelTrainningScore(final_model_MLP,xtrain,ytrain,nCV=10)     
    return scores




def main():

    ## Reading data prepared by Nick 

    ## Define the group datasets
    fileTrainLst ={'group1':'Group5K1_trainning_dataset.csv',
                   'group2':'Group5K2_trainning_dataset.csv', 
                   'group3':'Group5K3_trainning_dataset.csv',}
    dataFolder = os.getcwd()

    trainData=[]
    for group, file in fileTrainLst.items():
        ## Read the input dataset
        InsFile = os.path.join(dataFolder, file)
        data =pd.read_csv(InsFile)
        trainData.append(data)
    
    wholeDataset = pd.concat([trainData[0],trainData[1],trainData[2]],axis=0)

    InsFile = os.path.join(dataFolder, 'dataset5K_test.csv')
    data =pd.read_csv(InsFile)
    trainAndTest = pd.concat([wholeDataset,data],axis=0)
    X = trainAndTest.iloc[:,1:4]
    scaler = StandardScaler().fit(X)
    pickle.dump(scaler, open('scaler.pkl', 'wb'))

    groupLst=[]
    varLst = []
    modelTypeLst = []
    trainScore = []
    rfModel = False
    xgBoost = False #True
    lgBoost = False #True #True
    gBoost  =  False #True
    MLP = False #True
    GPy = True #True
    
    
    for i, data in enumerate(trainData):
    
    
        ## Read the input dataset
        #InsFile = os.path.join(dataFolder, file)
        #data =pd.read_csv(InsFile)
        group = 'group'+str(i+1)
        dataX=data.iloc[:,1:4]
        #Start to prepare the target data, here we need to loop the targets 
        # but firstly 
        outFolder = os.path.join(dataFolder,'SavedModel',group)
        if not os.path.exists(outFolder):
             os.makedirs(outFolder)
        
        for jcol,col in enumerate(['pH']):  # data.columns[4:]): 
            print('Training With ===>', group,'Target==>',col)
            icol = jcol + 4
            dataY = data[col]
            ## Check the max and min value of Y
            ## IF change of Y is less than 1%,
            ## then the target will be considered as constant, and 
            ## this value will be applied to the group
            
            X = dataX.values
            Y = dataY.values
            
            
            if '/' in col:
               fileCol = col.replace("/",'')
               #print(fileCol)
            else:
               fileCol = col            
            if Y.max()+Y.min() != 0:
                percnt = (Y.max()-Y.min())/(Y.max()+Y.min())*200.0
            else:
                percnt =0.0
            if percnt <1.0:
               fileName = os.path.join(outFolder,'const_'+fileCol+'.csv')
               tempdf = pd.DataFrame({'var':[col],'varID':[icol],'const':[(Y.max()+Y.min())*0.5]})
               tempdf.to_csv(fileName,index=False)
               modelTypeLst.append('CONST')
               trainScore.append([0,0,0,0])
               groupLst.append(group)
               varLst.append(col)
               
               continue
                
            # Check the correlation of each of dataX to DataY 
            corrLst = []
            for colX in dataX.columns:
                corr = dataX[colX].corr(dataY)
                corrLst.append(abs(corr))
            
            if max(corrLst)>=0.99:   # we use a linear model to simulate
                #give the X and Y for fiting a linear model                   
                regLinear = LinearRegression().fit(X, Y) 
                # conduct cross validation
                scores = outputModelTrainningScore(regLinear,X,Y,nCV=10)     
                trainScore.append(scores)
                groupLst.append(group)
                varLst.append(col)
                modelTypeLst.append('linear')  
                fileName = os.path.join(outFolder,'linear_'+fileCol+'.sav')
                pickle.dump(regLinear, open(fileName, 'wb'))             
                continue
                
            ## To  train with the 6 ML models    
            
            #if col =='pH' and group =='group2':
                ## Use all the train dataset for this case      
            #    wholeDataset = pd.concat([trainData[0],trainData[1],trainData[2]],axis=0)
            #    dataX=wholeDataset.iloc[:,1:4]
            #    dataY=wholeDataset['pH']
            #    X = dataX.values
            #    Y = dataY.values
            #    print(X.shape,Y.shape)
            #else:
            #    X = dataX.values
            #    Y = dataY.values
            
            
            
            #scaler = StandardScaler().fit(X)
            X_scaled = scaler.transform(X)  # This will be used for input of trainning dataset

            if rfModel:
                fileName = os.path.normpath( os.path.join( outFolder,'rfModel_'+str(icol)+'_'+fileCol+'.sav' ) )             
                scores = randomforestModel(X_scaled,Y,fileName)
                groupLst.append(group)
                varLst.append(col)
                modelTypeLst.append('rfModel')  
                trainScore.append(scores)

            if xgBoost:
                fileName = os.path.normpath( os.path.join( outFolder,'xgbModel_'+str(icol)+'_'+fileCol+'.sav' ) )             
                scores = xgbModel(X_scaled,Y,fileName)
                groupLst.append(group)
                varLst.append(col)
                modelTypeLst.append('xgbModel')  
                trainScore.append(scores)
               
            if lgBoost:
                fileName = os.path.normpath( os.path.join( outFolder,'lgbModel_'+str(icol)+'_'+fileCol+'.sav' ) )             
                scores = lgbModel(X_scaled,Y,fileName)
                groupLst.append(group)
                varLst.append(col)
                modelTypeLst.append('lgbModel')  
                trainScore.append(scores)

            if gBoost:
                fileName = os.path.normpath( os.path.join( outFolder,'gbModel_'+str(icol)+'_'+fileCol+'.sav' ) )             
                scores = gboostModel(X_scaled,Y,fileName)
                groupLst.append(group)
                varLst.append(col)
                modelTypeLst.append('gbModel')  
                trainScore.append(scores)

            if MLP:
                fileName = os.path.normpath( os.path.join( outFolder,'MLPModel_'+str(icol)+'_'+fileCol+'.sav' ) )             
                scores = MLPModel(X_scaled,Y,fileName)
                groupLst.append(group)
                varLst.append(col)
                modelTypeLst.append('MLPModel')  
                trainScore.append(scores)
            if GPy:
                fileName = os.path.normpath( os.path.join( outFolder,'GPyModel_'+fileCol+'.sav' ) )             
                scores = GPyModel(X_scaled,Y,fileName)
                groupLst.append(group)
                varLst.append(col)
                modelTypeLst.append('GPyModel')  
                trainScore.append(scores)
    ModelTrainSummary = pd.DataFrame({'group':groupLst,'var':varLst,'modelType':modelTypeLst})
    ModelTrainMeasures = pd.DataFrame(trainScore,columns=['R2_mean','R2_std','RMSE_mean','RMSE_std'])
    ModelTrainSummary = pd.concat([ModelTrainSummary,ModelTrainMeasures],axis=1)
    ModelTrainSummary.to_csv('ModelTrainSummary.csv')
    
    
    
    
if __name__ == '__main__': main()






