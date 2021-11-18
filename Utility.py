import numpy as np
import pandas as pd
from IPython.display import Image
from sklearn.model_selection import train_test_split
import os
import sys
import os.path
from os import path
from scipy import stats
import statistics

from causalml.dataset import make_uplift_classification
from causalml.inference.tree import UpliftTreeClassifier, UpliftRandomForestClassifier
from causalml.inference.tree import uplift_tree_string, uplift_tree_plot
from causalml.inference.meta import BaseXRegressor, BaseRRegressor, BaseSRegressor, BaseTRegressor
from causalml.metrics.visualize import *
from econml.metalearners import TLearner, SLearner, XLearner, DomainAdaptationLearner
from econml.dr import DRLearner

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from xgboost import XGBRegressor, XGBClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor
import warnings




separation = '_vv_'
    
def executeBaseline(inFileName, method, outcome, factors, excludeFactors):

    fold = 5
    BASE_DIR = os.getcwd()
    inputPath = os.path.join(BASE_DIR, 'input', inFileName)
    baseOutFolder = os.path.join(BASE_DIR, 'output', method, inFileName)
    if not os.path.exists(baseOutFolder):
        try:
            os.makedirs(baseOutFolder)
        except:
            print("An exception occurred making dir: " + baseOutFolder)  
                                   
    for i in range(1, fold + 1):
        
        trainFilePath = os.path.join(inputPath,'split', inFileName + '_train_'  + str(i) + '.csv')
        orTrainData = pd.read_csv(trainFilePath,  encoding = "ISO-8859-1", engine='python')
        testFilePath = os.path.join(inputPath,'split', inFileName + '_test_'  + str(i) + '.csv')
        orTestData = pd.read_csv(testFilePath,  encoding = "ISO-8859-1", engine='python')
        
        trainData = orTrainData.copy()
        testData = orTestData.copy()
        bestFTestData = orTestData.copy()
        tempImprve = pd.DataFrame({})

        for factor in factors:
            
            covariates = list(set(trainData.columns) - {outcome} - set(excludeFactors) - set(factors)) 
            X_train = trainData[covariates].to_numpy()
            treatment_train = trainData[factor]  
            y_train = trainData[outcome]
        
            X_test = testData[covariates].to_numpy()
            treatment_test = testData[factor]  
            y_test = testData[outcome]
           
            if(method  == 'Xlearner'):
                # Instantiate X learner
                models = GradientBoostingRegressor(n_estimators=500, max_depth=6, min_samples_leaf=int(50))
                propensity_model = RandomForestClassifier(n_estimators=500, max_depth=6)
                X_learner = XLearner(models=models, propensity_model=propensity_model)
                X_learner.fit(y_train, treatment_train, X=X_train)
                X_te = X_learner.effect(X_test)    
        
            if(method  == 'DRL'):
                # Instantiate DRL
                
                outcome_model = GradientBoostingRegressor(n_estimators=500, max_depth=6)
                pseudo_treatment_model = GradientBoostingRegressor(n_estimators=500, max_depth=6, min_samples_leaf=int(50))
                propensity_model = RandomForestClassifier(n_estimators=500, max_depth=6)
                
                DR_learner = DRLearner(model_regression=outcome_model, model_propensity=propensity_model,
                                       model_final=pseudo_treatment_model, cv=5)
                # Train DR_learner
                DR_learner.fit(y_train, treatment_train, X=X_train)
                # Estimate treatment effects on test data
                X_te = DR_learner.effect(X_test)
        
            testData['LIFT_SCORE'] = X_te
            testData['TREATMENT_NAME'] = factor
            testData['FOLLOW_REC'] = testData.apply(lambda x: x[x['TREATMENT_NAME']] , axis=1)
            testData = estimateUplift(testData, outcome)
            outFolder = os.path.join(baseOutFolder, factor)
            if not os.path.exists(outFolder):
                try:
                    os.makedirs(outFolder)
                except:
                    print("An exception occurred making dir: " + outFolder)  
                    
            outFilePath = os.path.join(outFolder, inFileName +'_' + method + '_' + str(i) + '.csv')
            testData.to_csv(outFilePath, index=False)     
             ### For the best factor
            tempImprve[factor] = pd.Series(X_te)
            bestFTestData['Lift' + '_' + factor] = tempImprve[factor]       

        bestFactor = tempImprve.idxmax(axis=1)
        bestFTestData['TREATMENT_NAME'] = pd.Series(bestFactor)
        bestFTestData['LIFT_SCORE'] = bestFTestData.apply(lambda x: x['Lift' + '_' + x['TREATMENT_NAME'] ], axis= 1)    
        bestFTestData['FOLLOW_REC'] = bestFTestData.apply(lambda x: x[x['TREATMENT_NAME']] , axis=1)
        bestFTestData = estimateUplift(bestFTestData, outcome)        
        
        outFolder = os.path.join(baseOutFolder, 'bestFactor')
        
        if not os.path.exists(outFolder):
            try:
                os.makedirs(outFolder)
            except:
                print("An exception occurred making dir: " + outFolder) 
                    
        outFilePath = os.path.join(outFolder, inFileName +'_' + method + '_' + str(i) + '.csv')
        bestFTestData.to_csv(outFilePath, index=False)
        
        
def executeBaselineOne(inFileName, method, outcome, factors, excludeFactors):

    BASE_DIR = os.getcwd()
    inputPath = os.path.join(BASE_DIR, 'input', inFileName)
    baseOutFolder = os.path.join(BASE_DIR, 'output', method, inFileName, 'One')
    if not os.path.exists(baseOutFolder):
        try:
            os.makedirs(baseOutFolder)
        except:
            print("An exception occurred making dir: " + baseOutFolder)  
                
                
                
    trainFilePath = os.path.join(inputPath, inFileName  + '.csv')
    orTrainData = pd.read_csv(trainFilePath,  encoding = "ISO-8859-1", engine='python')
    testFilePath = os.path.join(inputPath, inFileName + '_full' + '.csv')
    orTestData = pd.read_csv(testFilePath,  encoding = "ISO-8859-1", engine='python')

    trainData = orTrainData.copy()
    testData = orTestData.copy()
    bestFTestData = orTestData.copy()
    tempImprve = pd.DataFrame({})

    for factor in factors:
        
        covariates = list(set(trainData.columns) - {outcome} - set(excludeFactors) - set(factors)) 
        X_train = trainData[covariates].to_numpy()
        treatment_train = trainData[factor]  
        y_train = trainData[outcome]

        X_test = testData[covariates].to_numpy()
        treatment_test = testData[factor]  
        y_test = testData[outcome]
       
        if(method  == 'Xlearner'):
            # Instantiate X learner
            models = GradientBoostingRegressor(n_estimators=500, max_depth=6, min_samples_leaf=int(50))
            propensity_model = RandomForestClassifier(n_estimators=500, max_depth=6, 
                                                              min_samples_leaf=int(50))
            X_learner = XLearner(models=models, propensity_model=propensity_model)
            X_learner.fit(y_train, treatment_train, X=X_train)
            X_te = X_learner.effect(X_test)    

        if(method  == 'DRL'):
            # Instantiate DRL
            
            outcome_model = GradientBoostingRegressor(n_estimators=500, max_depth=6, min_samples_leaf=int(50))
            pseudo_treatment_model = GradientBoostingRegressor(n_estimators=500, max_depth=6, min_samples_leaf=int(50))
            propensity_model = RandomForestClassifier(n_estimators=500, max_depth=6, 
                                                              min_samples_leaf=int(50))
            
            DR_learner = DRLearner(model_regression=outcome_model, model_propensity=propensity_model,
                                   model_final=pseudo_treatment_model, cv=5)
            # Train DR_learner
            DR_learner.fit(y_train, treatment_train, X=X_train)
            # Estimate treatment effects on test data
            X_te = DR_learner.effect(X_test)

        testData['LIFT_SCORE'] = X_te
        testData['TREATMENT_NAME'] = factor
        testData['FOLLOW_REC'] = testData.apply(lambda x: x[x['TREATMENT_NAME']] , axis=1)           
        testData = estimateUplift(testData, outcome)
        outFolder = os.path.join(baseOutFolder, factor)
        if not os.path.exists(outFolder):
            try:
                os.makedirs(outFolder)
            except:
                print("An exception occurred making dir: " + outFolder)  
                
        outFilePath = os.path.join(outFolder, inFileName +'_' + method + '.csv')
        testData.to_csv(outFilePath, index=False)     
         ### For the best factor
        tempImprve[factor] = pd.Series(X_te)
        bestFTestData['Lift' + '_' + factor] = tempImprve[factor]       

    bestFactor = tempImprve.idxmax(axis=1)
    bestFTestData['TREATMENT_NAME'] = pd.Series(bestFactor)
    bestFTestData['LIFT_SCORE'] = bestFTestData.apply(lambda x: x['Lift' + '_' + x['TREATMENT_NAME'] ], axis= 1)
    
    bestFTestData['FOLLOW_REC'] = bestFTestData.apply(lambda x: x[x['TREATMENT_NAME']] , axis=1)
    bestFTestData = estimateUplift(bestFTestData, outcome)

    outFolder = os.path.join(baseOutFolder, 'bestFactor')

    if not os.path.exists(outFolder):
        try:
            os.makedirs(outFolder)
        except:
            print("An exception occurred making dir: " + outFolder) 
                
    outFilePath = os.path.join(outFolder, inFileName +'_' + method +'.csv')
    bestFTestData.to_csv(outFilePath, index=False)
    
    
def estimateUplift(estimatedImprovements, outcomeName, sortbyabs=False):
    
    if(sortbyabs):
        estimatedImprovements['ABS_Improvement'] = estimatedImprovements['LIFT_SCORE'].abs()
        estimatedImprovements.sort_values(by=['ABS_Improvement'], ascending = [False], inplace=True, axis=0)
    else:
        estimatedImprovements.sort_values(by=['LIFT_SCORE'], ascending = [False], inplace=True, axis=0)
    estimatedImprovements = estimatedImprovements.reset_index(drop=True) 
    
    Sum_Y_Follow_Rec    = np.array([])
    Sum_Nbr_Follow_Rec    = np.array([])
    Sum_Y_Not_Follow_Rec    = np.array([])
    Sum_Nbr_Not_Follow_Rec    = np.array([])
    Improvement    = np.array([])
    total_Y_Follow_Rec  = 0
    total_Nbr_Follow_Rec = 0
    total_Y_Not_Follow_Rec  = 0
    total_Nbr_Not_Follow_Rec = 0    
    for index, individual in estimatedImprovements.iterrows():
        improvementTemp = 0
        if(individual['FOLLOW_REC'] == 1):
            total_Nbr_Follow_Rec = total_Nbr_Follow_Rec + 1
            total_Y_Follow_Rec = total_Y_Follow_Rec + individual[outcomeName]
        else:
            total_Nbr_Not_Follow_Rec = total_Nbr_Not_Follow_Rec + 1
            total_Y_Not_Follow_Rec = total_Y_Not_Follow_Rec + individual[outcomeName]   
        Sum_Nbr_Follow_Rec = np.append (Sum_Nbr_Follow_Rec, total_Nbr_Follow_Rec)
        Sum_Y_Follow_Rec = np.append (Sum_Y_Follow_Rec, total_Y_Follow_Rec)
        Sum_Nbr_Not_Follow_Rec = np.append (Sum_Nbr_Not_Follow_Rec, total_Nbr_Not_Follow_Rec)
        Sum_Y_Not_Follow_Rec = np.append (Sum_Y_Not_Follow_Rec, total_Y_Not_Follow_Rec)
        if(total_Nbr_Follow_Rec == 0 or total_Nbr_Not_Follow_Rec == 0 ):
            if(total_Nbr_Follow_Rec > 0):
                improvementTemp = (total_Y_Follow_Rec/total_Nbr_Follow_Rec)
            else:
                improvementTemp = 0
        else:
            improvementTemp = (total_Y_Follow_Rec/total_Nbr_Follow_Rec) - (total_Y_Not_Follow_Rec/total_Nbr_Not_Follow_Rec)   
        Improvement = np.append (Improvement, improvementTemp)
    ser = pd.Series(Sum_Nbr_Follow_Rec)
    estimatedImprovements['N_TREATED'] = ser
    ser = pd.Series(Sum_Y_Follow_Rec)
    estimatedImprovements['Y_TREATED'] = ser
    ser = pd.Series(Sum_Nbr_Not_Follow_Rec)
    estimatedImprovements['N_UNTREATED'] = ser
    ser = pd.Series(Sum_Y_Not_Follow_Rec)
    estimatedImprovements['Y_UNTREATED'] = ser  
    ser = pd.Series(Improvement)
    estimatedImprovements['UPLIFT'] = ser
    return estimatedImprovements
    
def genQiniDataML(folderName, fold, prefileName, postfileName, modelName, outcomeName):
    improvementMtreeModels = []
    for fileCount in range (1, fold + 1):
        fPath = os.path.join(folderName, prefileName + str(fileCount) + postfileName + '.csv')   
        Sdataset0 = pd.read_csv(fPath,  encoding = "ISO-8859-1", engine='python')
        if (not('Improvement' in Sdataset0.columns)):
            Sdataset0 ['Improvement'] =  Sdataset0 ['LIFT_SCORE'] 
        if (not('FollowRec' in Sdataset0.columns)):
            Sdataset0 ['FollowRec'] = Sdataset0 ['FOLLOW_REC']           
        newImprovement = estimateQiniCurve(Sdataset0, outcomeName, modelName)
        improvementMtreeModels.append(newImprovement)   
    improvementMtreeCurves = pd.DataFrame({})
    improvementMtreeCurves['n'] = improvementMtreeModels[0]['n']
    improvementMtreeCurves['model'] = improvementMtreeModels[0]['model']   
    icount = 1
    modelNames = []
    groupModelNames = []   
    for eachM in improvementMtreeModels:
        improvementMtreeCurves['uplift' + str(icount)] = eachM['uplift']
        modelNames.append('uplift' + str(icount))
        improvementMtreeCurves['grUplift' + str(icount)] = eachM['grUplift']
        groupModelNames.append('grUplift' + str(icount))
        icount = icount + 1   
    improvementMtreeCurves['uplift'] = improvementMtreeCurves[modelNames].mean(axis=1)
    improvementMtreeCurves['grUplift'] = improvementMtreeCurves[groupModelNames].mean(axis=1)
    return improvementMtreeCurves
    
def plotBarImprovementTopV2(improvementModel, modelNames, iaxis, startCount, perPop = [0.2, 0.4, 0.6, 0.8, 1.0], tickLabel=None, minY=None, maxY=None, title=None, xlabel=None):
    averageImprovement = []
    inPerPop = []
    inModelNames = []
    transfModelNames = []
    for modelName in modelNames:
        tempModel = ''
        if(modelName == 'CausalTree'):
            tempModel = 'CT'   
        elif(modelName == 'TOTree'):
            tempModel = 'TOT'   
        elif(modelName == 'TStatisticTree'):
            tempModel = 'ST'  
        elif(modelName == 'FitBasedTree'):
            tempModel = 'FT'   
        elif(modelName == 'MCTTree'):
            tempModel = 'MCT'   
        elif(modelName == 'CausalTree_En'):
            tempModel = 'CTEn'   
        elif(modelName == 'TOTree_En'):
            tempModel = 'TOTEn'   
        elif(modelName == 'TStatisticTree_En'):
            tempModel = 'STEn'  
        elif(modelName == 'FitBasedTree_En'):
            tempModel = 'FTEn'           
        else:
            tempModel = modelName

            
        transfModelNames.append(tempModel)
        for per in perPop:   
            tempVal = improvementModel[(improvementModel['n'] < per)&(improvementModel['model'] ==  modelName)].copy()
            tempVal.reset_index(drop=True, inplace=True)
            averageImprovement.append(tempVal['grUplift'].iloc[-1] )
            inPerPop.append(per)
            inModelNames.append(tempModel)
    topImModel = pd.DataFrame({'AverageImp': averageImprovement, 'perPop': inPerPop, 'model': inModelNames})
            
    print('--------topImModel------')
    print(topImModel)
    for fileCount in range(0, len(transfModelNames)):
        tempImModel =  topImModel[topImModel.model == transfModelNames[fileCount] ]          
        flatui = None   
        if(transfModelNames[fileCount] == 'CF'):
            flatui = 'deeppink' 
        if(transfModelNames[fileCount] == 'IPCW'):
            flatui = 'steelblue'   
        if(transfModelNames[fileCount] == 'Cox'):
            flatui = 'olive'  
        if(transfModelNames[fileCount] == 'Survf'):
            flatui = 'blue'    
        if((transfModelNames[fileCount] == 'PCGF') or (transfModelNames[fileCount] == 'PIR')):
            flatui = 'tab:orange' 
        if(transfModelNames[fileCount] == 'DSURV'):
            flatui = 'darkred'    
        if(transfModelNames[fileCount] == 'DMTLR'):
            flatui = 'saddlebrown' 
        if(transfModelNames[fileCount] == 'KSVM'):
            flatui = 'yellowgreen'            

        if(transfModelNames[fileCount] == 'LRIR'):
            flatui = 'tab:orange'  
            
        if(transfModelNames[fileCount] == 'LRIR-CF'):
            flatui = 'yellow'               
            
        g = sns.barplot(x='perPop', y='AverageImp',  data= tempImModel, ax = iaxis[startCount + fileCount], color = flatui )
        patch = g.patches[0]
        tempX = np.arange(0,len(tempImModel)) + patch.get_width()/2
        sns.lineplot(data = tempImModel, x=tempX, y='AverageImp', color = 'blue',  marker='o', ax = iaxis[startCount + fileCount])
        iaxis[startCount + fileCount].grid(False)
        
        iaxis[startCount + fileCount].set_xlabel('')
        iaxis[startCount + fileCount].set_ylabel('')
        iaxis[startCount + fileCount].spines['top'].set_visible(False)
        iaxis[startCount + fileCount].spines['right'].set_visible(False)
        iaxis[startCount + fileCount].spines['left'].set_color('black')
        iaxis[startCount + fileCount].spines['bottom'].set_color('black')
        iaxis[startCount + fileCount].tick_params(axis='both', which='major', labelsize=12)
        #iaxis[startCount + fileCount].set_xticklabels(iaxis[startCount + fileCount].get_xticklabels(), rotation=45)
        iaxis[startCount + fileCount].set_xticklabels([])
        #iaxis[startCount + fileCount].legend(fontsize=16, ncol=1)
        iaxis[startCount + fileCount].set(ylim=(minY, maxY))
        iaxis[startCount + fileCount].spines['left'].set_linewidth(1)
        iaxis[startCount + fileCount].spines['bottom'].set_linewidth(1) 
        iaxis[startCount + fileCount].patch.set_facecolor('w')  
        if(title !=None):
            iaxis[startCount + fileCount].set_title(title, fontsize=12)
            
        if(xlabel != None):
            iaxis[startCount + fileCount].set_xlabel(xlabel,fontsize=12, rotation=45) 
        #iaxis[startCount + fileCount].yaxis.set_major_locator(ticker.MultipleLocator(3))
        #iaxis[startCount + fileCount].yaxis.set_major_formatter(ticker.ScalarFormatter())        
            
        if(tickLabel != None):
            iaxis[startCount + fileCount].set_xticklabels(tickLabel, rotation=45)
            
def getAUUCTopGroup(FolderLocation, fold, prefileName, postfileName, outcomeName, plotFig=True):
    improvementMtreeModels = []
    for fileCount in range (1, fold + 1):
        improveFilePath = os.path.join(FolderLocation, prefileName + str(fileCount) + postfileName + '.csv')
        if(not (path.exists(improveFilePath) )):
            continue
        results = pd.read_csv(improveFilePath,  encoding = "ISO-8859-1", engine='python')
        if (not('Improvement' in results.columns)):
            results ['Improvement'] =  results ['LIFT_SCORE'] 
        if (not('FollowRec' in results.columns)):
            results ['FollowRec'] = results ['FOLLOW_REC']
                        
        newImprovement = estimateQiniCurve(results, outcomeName, 'Tree')
        improvementMtreeModels.append(newImprovement)   
    improvementMtreeCurves = pd.DataFrame({})
    improvementMtreeCurves['n'] = improvementMtreeModels[0]['n']
    improvementMtreeCurves['model'] = improvementMtreeModels[0]['model']
    icount = 1
    modelNames = []
    groupModelNames = []
    for eachM in improvementMtreeModels:
        improvementMtreeCurves['uplift' + str(icount)] = eachM['uplift']
        modelNames.append('uplift' + str(icount))
        improvementMtreeCurves['grUplift' + str(icount)] = eachM['grUplift']
        groupModelNames.append('grUplift' + str(icount))
        icount = icount + 1  
    improvementMtreeCurves['uplift'] = improvementMtreeCurves[modelNames].mean(axis=1)
    improvementMtreeCurves['grUplift'] = improvementMtreeCurves[groupModelNames].mean(axis=1)
    improvementModels = pd.DataFrame({})
    improvementModels = improvementModels.append(improvementMtreeCurves)
    ## convert to percent
    improvementModels['uplift'] = improvementModels['uplift']* 100
    improvementModels['grUplift'] = improvementModels['grUplift']* 100
    if(plotFig):
        plotQini(improvementModels)
    curveNames = ['Tree']   
    improvementModels['uplift'] = improvementModels['uplift'].fillna(0)
    estimateAres = areaUnderCurve(improvementModels, curveNames)
    return  estimateAres[0]/100

def getAuucScore(FolderLocation, fold, prefileName, postfileName, outcomeName, factor):
    icount = 0
    totalScoreg = 0
    totalScoregi = 0
    for fileCount in range (1, fold + 1):
        improveFilePath = os.path.join(FolderLocation, prefileName + str(fileCount) + postfileName + '.csv')
        if(not (path.exists(improveFilePath) )):
            continue
        results = pd.read_csv(improveFilePath,  encoding = "ISO-8859-1", engine='python')
        df = pd.DataFrame({})
        df['Y'] = results[outcomeName]
        df['T'] = results[factor]
        df['CUR'] = results['LIFT_SCORE'] 
        icount = icount + 1
        scoreg = auuc_score(df,  outcome_col='Y', treatment_col='T')
        scoregi = qini_score(df,  outcome_col='Y', treatment_col='T')
        totalScoreg = totalScoreg + scoreg['CUR']
        totalScoregi = totalScoregi + scoregi['CUR']
        
    return totalScoreg/icount, totalScoregi/icount
    
def getDiversification(FolderLocation, fold, prefileName, postfileName, outcomeName, perPop):
    globalKendal = 0
    globalVariation = 0
    globalVariance = 0
    improvementModel = pd.DataFrame({})  
    newMolde = genQiniDataML(FolderLocation,fold, prefileName, postfileName, 'NoName', outcomeName)
    
    improvementModel = improvementModel.append(newMolde)
    improvementModel['uplift'] = improvementModel['uplift']* 100
    improvementModel['grUplift'] = improvementModel['grUplift']* 100   
    averageImprovement = []
    inPerPop = []
    averageVal = 0
    for per in perPop:
        tempVal = improvementModel[(improvementModel['n'] < per)].copy()
        tempVal.reset_index(drop=True, inplace=True)          
        tempVal = tempVal['grUplift'].iloc[-1]
        averageImprovement.append( tempVal)
        inPerPop.append(per)
        if(per == 1):
            averageVal = tempVal       
    topImModel = pd.DataFrame({'AverageImprove': averageImprovement, 'perPop': inPerPop})
    topImModel.fillna(averageVal)
    topImModel['perPop'] = sorted(perPop, reverse=True)
    res = stats.kendalltau(topImModel['AverageImprove'], topImModel['perPop'])
    globalKendal =  res[0]
    res = stats.spearmanr(topImModel['AverageImprove'], topImModel['perPop'])
    globalSpear = res[0]           
    return (globalKendal, globalSpear)
    
def estimateQiniCurve(estimatedImprovements, outcomeName, modelName):

    ranked = pd.DataFrame({})
    ranked['uplift_score'] = estimatedImprovements['Improvement']
    ranked['NUPLIFT'] = estimatedImprovements['UPLIFT']
    ranked['FollowRec'] = estimatedImprovements['FollowRec']
    ranked[outcomeName] = estimatedImprovements[outcomeName]
    ranked['countnbr'] = 1
    ranked['n'] = ranked['countnbr'].cumsum() / ranked.shape[0]
    uplift_model, random_model = ranked.copy(), ranked.copy()
    C, T = sum(ranked['FollowRec'] == 0), sum(ranked['FollowRec'] == 1)
    ranked['CR'] = 0
    ranked['TR'] = 0
    ranked.loc[(ranked['FollowRec'] == 0)
                            &(ranked[outcomeName]  == 1),'CR'] = ranked[outcomeName]
    ranked.loc[(ranked['FollowRec'] == 1)
                            &(ranked[outcomeName]  == 1),'TR'] = ranked[outcomeName]
    ranked['NotFollowRec'] = 1
    ranked['NotFollowRec'] = ranked['NotFollowRec']  - ranked['FollowRec'] 
    ranked['NotFollowRecCum'] = ranked['NotFollowRec'].cumsum() 
    ranked['FollowRecCum'] = ranked['FollowRec'].cumsum() 
    ranked['CR/C'] = ranked['CR'].cumsum() / ranked['NotFollowRec'].cumsum()    
    ranked['TR/T'] = ranked['TR'].cumsum() / ranked['FollowRec'] .cumsum()
    # Calculate and put the uplift into dataframe
    uplift_model['uplift'] = round((ranked['TR/T'] - ranked['CR/C'])*ranked['n'] ,5)
    uplift_model['uplift'] = round((ranked['NUPLIFT'])*ranked['n'] ,5)
    uplift_model['grUplift'] =ranked['NUPLIFT']
    random_model['uplift'] = round(ranked['n'] * uplift_model['uplift'].iloc[-1],5)
    ranked['uplift']  = ranked['TR/T'] - ranked['CR/C']
    # Add q0
    q0 = pd.DataFrame({'n':0, 'uplift':0}, index =[0])
    uplift_model = pd.concat([q0, uplift_model]).reset_index(drop = True)
    random_model = pd.concat([q0, random_model]).reset_index(drop = True)  
    # Add model name & concat
    uplift_model['model'] = modelName
    random_model['model'] = 'Random model'
    return uplift_model
    
def areaUnderCurve(models, modelNames):
    modelAreas = []
    for modelName in modelNames:   
        area = 0
        tempModel = models[models['model'] == modelName].copy()
        tempModel.reset_index(drop=True, inplace=True)       
        for i in range(1, len(tempModel)):  # df['A'].iloc[2]
            delta = tempModel['n'].iloc[i] - tempModel['n'].iloc[i-1]
            y = (tempModel['uplift'].iloc[i] + tempModel['uplift'].iloc[i-1]  )/2
            area += y*delta
        modelAreas.append(area)  
    return modelAreas
    
    
def findRec(currentRow):
    
    bestFactor = currentRow['TREATMENT_NAME']    
    res = bestFactor.split(separation)    
    facName = res[0]
    facVal = res[1]    
    
    return 1 if str(currentRow[facName]) == str(facVal) else 0
        
def executeBaselineML(inFileName, method, outcome, factors, factorVals, excludeFactors):

    fold = 5
    BASE_DIR = os.getcwd()
    inputPath = os.path.join(BASE_DIR, 'input', inFileName)
    baseOutFolder = os.path.join(BASE_DIR, 'output', method, inFileName)
    separation = '_vv_'

    if not os.path.exists(baseOutFolder):
        try:
            os.makedirs(baseOutFolder)
        except:
            print("An exception occurred making dir: " + baseOutFolder)  
            
            
    for i in range(1, fold + 1):
        
        trainFilePath = os.path.join(inputPath,'split', inFileName + '_train_'  + str(i) + '.csv')
        orTrainData = pd.read_csv(trainFilePath,  encoding = "ISO-8859-1", engine='python')
        testFilePath = os.path.join(inputPath,'split', inFileName + '_test_'  + str(i) + '.csv')
        orTestData = pd.read_csv(testFilePath,  encoding = "ISO-8859-1", engine='python')
        
        trainData = orTrainData.copy()
        bestFTestData = orTestData.copy()
        tempImprve = pd.DataFrame({})
        
        globalLiftScores = pd.DataFrame({})
        for factor in factors:
            covariates = list(set(trainData.columns) - {outcome} - set(excludeFactors) - set(factors)) 
            trainData[factor] = trainData[factor].apply(lambda x: str(x))
            
            
            if(method == 'Xlearner'):
                causaVals = factorVals[factor]   
                liftScores = pd.DataFrame({})
                for ival in range(1, len(causaVals)):
                    
                    f_train_data = trainData.copy()
                    f_train_data = trainData[trainData[factor].isin ([str(causaVals[0]), str(causaVals[ival])]) ]
                    f_train_data[factor] = f_train_data[factor].apply(lambda x: 1 if x == str(causaVals[ival])  else 0)
                    
                    
                    models = GradientBoostingRegressor(n_estimators=500, max_depth=6, min_samples_leaf=int(50))
                    propensity_model = RandomForestClassifier(n_estimators=500, max_depth=6)
                    X_learner = XLearner(models=models, propensity_model=propensity_model)
                    X_learner.fit(f_train_data[outcome], f_train_data[factor], X=f_train_data[covariates].to_numpy())    
                          
                    testData = orTestData.copy()
                    X_te = X_learner.effect(testData[covariates].to_numpy()) 
                    tempScore = pd.DataFrame(X_te, columns =[factor + separation + str(causaVals[ival])] )
                     
                    liftScores = pd.concat([liftScores, tempScore], axis=1)
                    globalLiftScores = pd.concat([globalLiftScores, tempScore], axis=1)
                

            elif(method == 'DRL'):            
                causaVals = factorVals[factor]   
                liftScores = pd.DataFrame({})
                for ival in range(1, len(causaVals)):
                    
                    f_train_data = trainData.copy()
                    f_train_data = trainData[trainData[factor].isin ([str(causaVals[0]), str(causaVals[ival])]) ]
                    f_train_data[factor] = f_train_data[factor].apply(lambda x: 1 if x == str(causaVals[ival])  else 0)
                    
                    outcome_model = GradientBoostingRegressor(n_estimators=500, max_depth=6)
                    pseudo_treatment_model = GradientBoostingRegressor(n_estimators=500, max_depth=6, min_samples_leaf=int(50))
                    propensity_model = RandomForestClassifier(n_estimators=500, max_depth=6)
                
                    
                    DR_learner = DRLearner(model_regression=outcome_model, model_propensity=propensity_model,
                                       model_final=pseudo_treatment_model)
                    DR_learner.fit(f_train_data[outcome], f_train_data[factor], X=f_train_data[covariates].to_numpy())
                    
                    testData = orTestData.copy()
                    X_te = DR_learner.effect(testData[covariates].to_numpy()) 
                    tempScore = pd.DataFrame(X_te, columns =[factor + separation + str(causaVals[ival])] )
                     
                    liftScores = pd.concat([liftScores, tempScore], axis=1)
                    globalLiftScores = pd.concat([globalLiftScores, tempScore], axis=1)

                            
            testData['LIFT_SCORE'] = liftScores.max(axis=1)
            
            testData['TREATMENT_NAME'] = liftScores.idxmax(axis=1)
            testData['FOLLOW_REC'] = testData.apply(lambda x: 1 if x['TREATMENT_NAME'] ==  factor + separation + str(x[factor])  else 0 , axis=1) 
           
            
            testData = estimateUplift(testData, outcome)
            outFolder = os.path.join(baseOutFolder, factor)
            if not os.path.exists(outFolder):
                try:
                    os.makedirs(outFolder)
                except:
                    print("An exception occurred making dir: " + outFolder) 
                    
            outFilePath = os.path.join(outFolder, inFileName +'_' + method + '_' + str(i) + '.csv')
            testData.to_csv(outFilePath, index=False)
            
                    
        bestFTestData ['LIFT_SCORE'] = globalLiftScores.max(axis=1)   
        bestFTestData['TREATMENT_NAME'] = globalLiftScores.idxmax(axis=1)
        bestFTestData['FOLLOW_REC'] = bestFTestData.apply(findRec, axis=1)
        bestFTestData = estimateUplift(bestFTestData, outcome)

        outFolder = os.path.join(baseOutFolder, 'bestFactor')
        
        if not os.path.exists(outFolder):
            try:
                os.makedirs(outFolder)
            except:
                print("An exception occurred making dir: " + outFolder) 
                    
        outFilePath = os.path.join(outFolder, inFileName +'_' + method + '_'  + str(i) + '.csv')
        bestFTestData.to_csv(outFilePath, index=False)
        
        
def executeBaselineOneML(inFileName, method, outcome, factors, factorVals, excludeFactors):

    BASE_DIR = os.getcwd()
    inputPath = os.path.join(BASE_DIR, 'input', inFileName)
    baseOutFolder = os.path.join(BASE_DIR, 'output', method, inFileName, 'One')
    separation = '_vv_'
                    
    if not os.path.exists(baseOutFolder):
        try:
            os.makedirs(baseOutFolder)
        except:
            print("An exception occurred making dir: " + baseOutFolder)  
            
            
    trainFilePath = os.path.join(inputPath, inFileName  + '.csv')
    orTrainData = pd.read_csv(trainFilePath,  encoding = "ISO-8859-1", engine='python')

    testFilePath = os.path.join(inputPath, inFileName + '_full' + '.csv')
    orTestData = pd.read_csv(testFilePath,  encoding = "ISO-8859-1", engine='python')

    trainData = orTrainData.copy()
    bestFTestData = orTestData.copy()
    tempImprve = pd.DataFrame({})

    globalLiftScores = pd.DataFrame({})
    for factor in factors:
        covariates = list(set(trainData.columns) - {outcome} - set(excludeFactors) - set(factors)) 
        trainData[factor] = trainData[factor].apply(lambda x: str(x))
        
        
        if(method == 'Xlearner'):
        
            causaVals = factorVals[factor]   
            liftScores = pd.DataFrame({})
            for ival in range(1, len(causaVals)):
                
                f_train_data = trainData.copy()
                f_train_data = trainData[trainData[factor].isin ([str(causaVals[0]), str(causaVals[ival])]) ]
                f_train_data[factor] = f_train_data[factor].apply(lambda x: 1 if x == str(causaVals[ival])  else 0)
                
                
                models = GradientBoostingRegressor(n_estimators=500, max_depth=6, min_samples_leaf=int(50))
                propensity_model = RandomForestClassifier(n_estimators=500, max_depth=6)
                X_learner = XLearner(models=models, propensity_model=propensity_model)
                X_learner.fit(f_train_data[outcome], f_train_data[factor], X=f_train_data[covariates].to_numpy())    
                      
                testData = orTestData.copy()
                X_te = X_learner.effect(testData[covariates].to_numpy()) 
                tempScore = pd.DataFrame(X_te, columns =[factor + separation + str(causaVals[ival])] )
                 
                liftScores = pd.concat([liftScores, tempScore], axis=1)
                globalLiftScores = pd.concat([globalLiftScores, tempScore], axis=1)            

        elif(method == 'DRL'):            
            causaVals = factorVals[factor]   
            liftScores = pd.DataFrame({})
            for ival in range(1, len(causaVals)):
                
                f_train_data = trainData.copy()
                f_train_data = trainData[trainData[factor].isin ([str(causaVals[0]), str(causaVals[ival])]) ]
                f_train_data[factor] = f_train_data[factor].apply(lambda x: 1 if x == str(causaVals[ival])  else 0)
                
                outcome_model = GradientBoostingRegressor(n_estimators=500, max_depth=6)
                pseudo_treatment_model = GradientBoostingRegressor(n_estimators=500, max_depth=6, min_samples_leaf=int(50))
                propensity_model = RandomForestClassifier(n_estimators=500, max_depth=6)
                
                DR_learner = DRLearner(model_regression=outcome_model, model_propensity=propensity_model,
                                   model_final=pseudo_treatment_model)
                DR_learner.fit(f_train_data[outcome], f_train_data[factor], X=f_train_data[covariates].to_numpy())
                
                testData = orTestData.copy()
                X_te = DR_learner.effect(testData[covariates].to_numpy()) 
                tempScore = pd.DataFrame(X_te, columns =[factor + separation + str(causaVals[ival])] )
                 
                liftScores = pd.concat([liftScores, tempScore], axis=1)
                globalLiftScores = pd.concat([globalLiftScores, tempScore], axis=1)

                        
        testData['LIFT_SCORE'] = liftScores.max(axis=1)
        
        testData['TREATMENT_NAME'] = liftScores.idxmax(axis=1)
        testData['FOLLOW_REC'] = testData.apply(lambda x: 1 if x['TREATMENT_NAME'] ==  factor + separation + str(x[factor])  else 0 , axis=1) 
                   
    bestFTestData ['LIFT_SCORE'] = globalLiftScores.max(axis=1)   
    bestFTestData['TREATMENT_NAME'] = globalLiftScores.idxmax(axis=1)
    bestFTestData['FOLLOW_REC'] = bestFTestData.apply(findRec, axis=1)
    bestFTestData = estimateUplift(bestFTestData, outcome)

    outFolder = os.path.join(baseOutFolder, 'bestFactor')

    if not os.path.exists(outFolder):
        try:
            os.makedirs(outFolder)
        except:
            print("An exception occurred making dir: " + outFolder) 
                

    outFilePath = os.path.join(outFolder, inFileName +'_' + method  +'.csv')
    bestFTestData.to_csv(outFilePath, index=False)
    
    
def plotQini(model):
    plt.clf()
    
    # plot the data
    ax = sns.lineplot(x='n', y='uplift', hue='model', data=model,
                      style='model', dashes=False)
    # Plot settings
    sns.set_style('whitegrid')
    handles, labels = ax.get_legend_handles_labels()
    plt.xlabel('Proportion targeted',fontsize=15)
    plt.ylabel('CIMP (%)',fontsize=15)
    plt.subplots_adjust(right=1)
    plt.subplots_adjust(top=1)
    plt.legend(fontsize=12)
    ax.tick_params(labelsize=15)
    ax.legend(handles=handles[1:], labels=labels[1:], loc='upper left')

    
    

    
