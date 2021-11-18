
import os
import sys
import statistics 
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl, matplotlib.pyplot as plt
import os.path
from os import path
import random

from Utility import *

plotGBar = True
plotTrend = True
Rank = True

BASE_DIR = os.getcwd()
baseInFolder = os.path.join(BASE_DIR, 'output')


def plotTrendLine(inputName, factors, outcomeName, allAxis, datasetName, gcount, minY=None, maxY=None, label=False, xlabel=False, includeFac=True):

    startCount = gcount
    for cause in factors:
    
        factor = 'FOLLOW_REC'
        methods = []
        gainScores = []
        qiniScores = []
        auucScores = []
        
        #startCount = gcount
        perPop = [0.25, 0.5,  0.75, 1.0]        
        tickLabel  = ['0.25','0.5', '0.75', '1.0']     
        tickLabel = None
        

        method = 'Cox'
        method_name = method
        prefileName = inputName + '_' + method + '_' 
        postfileName = ''
        fullResultFolder = baseInFolder + '/'+ method +'/' + inputName +'/' + cause          
        newMolde = genQiniDataML(fullResultFolder, 5, prefileName, postfileName, method, outcomeName)
        improvementModels = pd.DataFrame({})
        improvementModels = improvementModels.append(newMolde)
        improvementModels['uplift'] = improvementModels['uplift']
        improvementModels['grUplift'] = improvementModels['grUplift'] 
        if(label):
            if(includeFac):
                title = datasetName +' (' + cause + ') '
            else:
                title = datasetName 
        else:
            title = None
        ixlabel = method_name if (xlabel) else None
        plotBarImprovementTopV2(improvementModels, [method], allAxis, startCount, perPop, tickLabel, minY, maxY,title=title, xlabel=ixlabel)
        startCount = startCount + 1

                        
        method = 'IPCW'
        method_name = method
        prefileName = inputName + '_' + method + '_' 
        postfileName = ''
        fullResultFolder = baseInFolder + '/'+ method +'/' + inputName +'/' + cause          
        newMolde = genQiniDataML(fullResultFolder, 5, prefileName, postfileName, method, outcomeName)
        improvementModels = pd.DataFrame({})
        improvementModels = improvementModels.append(newMolde)
        improvementModels['uplift'] = improvementModels['uplift']
        improvementModels['grUplift'] = improvementModels['grUplift']   
        if(label):
            if(includeFac):
                title = datasetName +' (' + cause + ') '
            else:
                title = datasetName 
        else:
            title = None
        ixlabel = method_name if (xlabel) else None
        plotBarImprovementTopV2(improvementModels, [method], allAxis, startCount, perPop, tickLabel, minY, maxY,title=title, xlabel=ixlabel)
        startCount = startCount + 1
        
        
        method = 'CF'
        method_name = 'SCF'
        prefileName = inputName + '_' + method + '_' 
        postfileName = ''
        fullResultFolder = baseInFolder + '/'+ method +'/' + inputName +'/' + cause          
        newMolde = genQiniDataML(fullResultFolder, 5, prefileName, postfileName, method, outcomeName)
        improvementModels = pd.DataFrame({})
        improvementModels = improvementModels.append(newMolde)
        improvementModels['uplift'] = improvementModels['uplift']
        improvementModels['grUplift'] = improvementModels['grUplift']   
        if(label):
            if(includeFac):
                title = datasetName +' (' + cause + ') '
            else:
                title = datasetName 
                
        else:
            title = None
        ixlabel = method_name if (xlabel) else None
        plotBarImprovementTopV2(improvementModels, [method], allAxis, startCount, perPop, tickLabel, minY, maxY,title=title, xlabel=ixlabel)
        startCount = startCount + 1
        
                
        
        method = 'DSURV'
        method_name = method
        prefileName = inputName + '_' + method + '_' 
        postfileName = ''
        fullResultFolder = baseInFolder + '/'+ method +'/' + inputName +'/' + cause          
        newMolde = genQiniDataML(fullResultFolder, 5, prefileName, postfileName, method, outcomeName)
        improvementModels = pd.DataFrame({})
        improvementModels = improvementModels.append(newMolde)
        improvementModels['uplift'] = improvementModels['uplift']
        improvementModels['grUplift'] = improvementModels['grUplift']   
        if(label):
            if(includeFac):
                title = datasetName +' (' + cause + ') '
            else:
                title = datasetName 
        else:
            title = None
        ixlabel = method_name if (xlabel) else None
        plotBarImprovementTopV2(improvementModels, [method], allAxis, startCount, perPop, tickLabel, minY, maxY,title=title, xlabel=ixlabel)
        startCount = startCount + 1       
             
        
        method = 'DMTLR'
        method_name = method
        prefileName = inputName + '_' + method + '_' 
        postfileName = ''
        fullResultFolder = baseInFolder + '/'+ method +'/' + inputName +'/' + cause          
        newMolde = genQiniDataML(fullResultFolder, 5, prefileName, postfileName, method, outcomeName)
        improvementModels = pd.DataFrame({})
        improvementModels = improvementModels.append(newMolde)
        improvementModels['uplift'] = improvementModels['uplift']
        improvementModels['grUplift'] = improvementModels['grUplift']  

        if(includeFac):
            title = datasetName +' (' + cause + ') '
        else:
            title = datasetName 
            
        ixlabel = method_name if (xlabel) else None
        plotBarImprovementTopV2(improvementModels, [method], allAxis, startCount, perPop, tickLabel, minY, maxY,title=title, xlabel=ixlabel)
        startCount = startCount + 1    


        method = 'Survf'
        method_name = 'SURVF'
        prefileName = inputName + '_' + method + '_' 
        postfileName = ''
        fullResultFolder = baseInFolder + '/'+ method +'/' + inputName +'/' + cause          
        newMolde = genQiniDataML(fullResultFolder, 5, prefileName, postfileName, method, outcomeName)
        improvementModels = pd.DataFrame({})
        improvementModels = improvementModels.append(newMolde)
        improvementModels['uplift'] = improvementModels['uplift']
        improvementModels['grUplift'] = improvementModels['grUplift']

        if(label):
            if(includeFac):
                title = datasetName +' (' + cause + ') '
            else:
                title = datasetName 
        else:
            title = None
            
        
        ixlabel = method_name if (xlabel) else None
        plotBarImprovementTopV2(improvementModels, [method], allAxis, startCount, 
                                perPop, tickLabel, minY, maxY,title=title, xlabel=ixlabel)
        startCount = startCount + 1
        
        
        method = 'KSVM'
        method_name = 'SVM'
        prefileName = inputName + '_' + method + '_' 
        postfileName = ''
        fullResultFolder = baseInFolder + '/'+ method +'/' + inputName +'/' + cause          
        newMolde = genQiniDataML(fullResultFolder, 5, prefileName, postfileName, method, outcomeName)
        improvementModels = pd.DataFrame({})
        improvementModels = improvementModels.append(newMolde)
        improvementModels['uplift'] = improvementModels['uplift']
        improvementModels['grUplift'] = improvementModels['grUplift']   
        if(label):
            if(includeFac):
                title = datasetName +' (' + cause + ') '
            else:
                title = datasetName 
        else:
            title = None
        ixlabel = method_name if (xlabel) else None
        plotBarImprovementTopV2(improvementModels, [method], allAxis, startCount, perPop, tickLabel, minY, maxY,title=title, xlabel=ixlabel)
        startCount = startCount + 1
        
        method = 'LRIR-CF'
        method_name = method
        prefileName = inputName + '_' + method + '_' 
        postfileName = ''
        fullResultFolder = baseInFolder + '/'+ method +'/' + inputName +'/' + cause          
        newMolde = genQiniDataML(fullResultFolder, 5, prefileName, postfileName, method, outcomeName)
        improvementModels = pd.DataFrame({})
        improvementModels = improvementModels.append(newMolde)
        improvementModels['uplift'] = improvementModels['uplift']
        improvementModels['grUplift'] = improvementModels['grUplift']  
        if(label):
            if(includeFac):
                title = datasetName +' (' + cause + ') '
            else:
                title = datasetName 
        else:
            title = None
        ixlabel = method_name if (xlabel) else None
        plotBarImprovementTopV2(improvementModels, [method], allAxis, startCount, perPop, tickLabel, minY, maxY,title=title, xlabel=ixlabel)
        startCount = startCount + 1    
        
        
        method = 'LRIR'
        method_name = method
        prefileName = inputName + '_' + method + '_' 
        postfileName = ''
        fullResultFolder = baseInFolder + '/'+ method +'/' + inputName +'/' + cause          
        newMolde = genQiniDataML(fullResultFolder, 5, prefileName, postfileName, method, outcomeName)
        improvementModels = pd.DataFrame({})
        improvementModels = improvementModels.append(newMolde)
        improvementModels['uplift'] = improvementModels['uplift']
        improvementModels['grUplift'] = improvementModels['grUplift']  
        if(label):
            if(includeFac):
                title = datasetName +' (' + cause + ') '
            else:
                title = datasetName 
        else:
            title = None
        ixlabel = method_name if (xlabel) else None
        plotBarImprovementTopV2(improvementModels, [method], allAxis, startCount, perPop, tickLabel, minY, maxY,title=title, xlabel=ixlabel)
        startCount = startCount + 1   
        
        
def plotAUUCBar(inputName, factors, outcomeName, iaxis, datasetName, gcount, minY=None, includeFac=True):


    for factor in factors:  
        print(factor)
        
        methods = []
        gainScores = []
        qiniScores = []
        auucScores = []
    
        method = 'CF'
        methods.append('SCF')
        prefileName = inputName + '_' + method + '_' 
        postfileName = ''
        fullResultFolder = baseInFolder + '/'+ method +'/' + inputName +'/' + factor
        opiAuc = getAUUCTopGroup(fullResultFolder, 5, prefileName, postfileName, outcomeName, False)
        auucScores.append(opiAuc) 
        print(method + ": " + str(opiAuc))  
                
        method = 'IPCW'
        methods.append('IPCW')
        prefileName = inputName + '_' + method + '_' 
        postfileName = ''
        fullResultFolder = baseInFolder + '/'+ method +'/' + inputName +'/' + factor
        opiAuc = getAUUCTopGroup(fullResultFolder, 5, prefileName, postfileName, outcomeName, False)
        auucScores.append(opiAuc) 
        print(method + ": " + str(opiAuc)) 
        
        method = 'Cox'
        methods.append(method)
        prefileName = inputName + '_' + method + '_' 
        postfileName = ''
        fullResultFolder = baseInFolder + '/'+ method +'/' + inputName +'/' + factor        
        opiAuc = getAUUCTopGroup(fullResultFolder, 5, prefileName, postfileName, outcomeName, False)
        auucScores.append(opiAuc) 
        print(method + ": " + str(opiAuc)) 
        
        method = 'Survf'
        methods.append('SURVF')
        prefileName = inputName + '_' + method + '_' 
        postfileName = ''
        fullResultFolder = baseInFolder + '/'+ method +'/' + inputName +'/' + factor        
        opiAuc = getAUUCTopGroup(fullResultFolder, 5, prefileName, postfileName, outcomeName, False)
        auucScores.append(opiAuc) 
        print(method + ": " + str(opiAuc)) 

        method = 'DSURV'
        methods.append(method)
        prefileName = inputName + '_' + method + '_' 
        postfileName = ''
        fullResultFolder = baseInFolder + '/'+ method +'/' + inputName +'/' + factor 
        opiAuc = getAUUCTopGroup(fullResultFolder, 5, prefileName, postfileName, outcomeName, False)
        auucScores.append(opiAuc) 
        print(method + ": " + str(opiAuc))  
        
        method = 'DMTLR'
        methods.append(method)        
        prefileName = inputName + '_' + method + '_' 
        postfileName = ''
        fullResultFolder = baseInFolder + '/'+ method +'/' + inputName +'/' + factor      
        opiAuc = getAUUCTopGroup(fullResultFolder, 5, prefileName, postfileName, outcomeName, False)
        auucScores.append(opiAuc) 
        print(method + ": " + str(opiAuc)) 
        
        method = 'LRIR'
        methods.append(method)
        prefileName = inputName + '_' + method + '_' 
        postfileName = ''
        fullResultFolder = baseInFolder + '/'+ method +'/' + inputName +'/' + factor
        opiAuc = getAUUCTopGroup(fullResultFolder, 5, prefileName, postfileName, outcomeName, False)
        auucScores.append(opiAuc) 
        print(method + ": " + str(opiAuc))   

        method = 'LRIR-CF'
        methods.append(method)
        prefileName = inputName + '_' + method + '_' 
        postfileName = ''
        fullResultFolder = baseInFolder + '/'+ method +'/' + inputName +'/' + factor
        opiAuc = getAUUCTopGroup(fullResultFolder, 5, prefileName, postfileName, outcomeName, False)
        auucScores.append(opiAuc) 
        print(method + ": " + str(opiAuc)) 
        
        
        method = 'KSVM'
        methods.append('SVM')
        prefileName = inputName + '_' + method + '_' 
        postfileName = ''
        fullResultFolder = baseInFolder + '/'+ method +'/' + inputName +'/' + factor      
        opiAuc = getAUUCTopGroup(fullResultFolder, 5, prefileName, postfileName, outcomeName, False)
        auucScores.append(opiAuc) 
        print(method + ": " + str(opiAuc))         
        
        
        grhData = pd.DataFrame({'CIMP': auucScores, 'Method': methods})
        gcount += 1
        
        colors = [ 'olive', 'steelblue', 'deeppink', 'darkred', 'saddlebrown', 'blue', 'yellowgreen', 
                  'yellow', 'tab:orange', 'red', 'green']
        g = sns.barplot(x='Method', y='CIMP', data= grhData, 
                        order = ['Cox','IPCW', 'SCF',  'DSURV', 'DMTLR', 'SURVF','SVM', 'LRIR-CF', 
                                 'LRIR'],  
        palette = colors,
        ax = iaxis[gcount])
        
        g.patch.set_facecolor('w')
        '''
        if(factor == 'bestFactor'):
            g.patch.set_facecolor('#f0e2a8')
        else:
            g.patch.set_facecolor('w')
        '''
        g.set_xlabel('')
        if((gcount % 2) == 0):
            g.set_ylabel('AUUC', fontsize=12)
        else:
            g.set_ylabel('')
        ## rename to a friendly ones
        if(factor == 'Grey_wage'):
            factor = 'Grey wage'
            
        if(factor == 'bestFactor'):
            factor = 'Personalized factor'
         
        if(factor == 'satisfaction_level'):
            factor = 'Satisfaction'
         
        if(factor == 'average_montly_hours'):
            factor = 'Average monthly hours'
            
        if(includeFac):
            g.set_title(factor +' (' + datasetName + ') ', fontsize=12)
        else: 
            g.set_title(datasetName, fontsize=12)
        g.tick_params(axis='both', which='major', labelsize=12)
        g.set_xticklabels(g.get_xticklabels(), rotation=45)
        g.spines['top'].set_visible(False)
        g.spines['right'].set_visible(False) 
        g.spines['left'].set_color('black')
        g.spines['bottom'].set_color('black')
        g.spines['left'].set_linewidth(1)
        g.spines['bottom'].set_linewidth(1)
        g.grid(False)
        if(minY != None):
            g.set(ylim=(minY, None))
        
            
        
def generateRank(inputName, factors, outcomeName, datasetName):

    ncauses  = factors.copy()
    perPop = [0.25, 0.5,  0.75, 1.0]
    methods = []
    cfactors = []
    kendals = []
    spears = []
    
    column_names = ['Dataset', 'Cox', 'IPCW', 'SCF', 'DSURV',  'DMTLR', 'SURVF', 'SVM', 'LRIR-CF', 'LRIR']
    res = pd.DataFrame(columns = column_names)

    
    for factor in ncauses:
    
        dictVal = {}
        
        dictVal['Dataset'] = datasetName
        

            
        mtag = ''
        
        if(factor == 'bestFactor'):
            mtag = '_pFactor'
            
        method = 'CF'
        method_name = 'SCF'
        prefileName = inputName + '_' + method + '_' 
        postfileName = ''
        fullResultFolder = baseInFolder + '/'+ method +'/' + inputName  + '/' +factor
        kendal, spear = getDiversification(fullResultFolder, 5, prefileName, postfileName, outcomeName, perPop)
        kendals.append(kendal)
        spears.append(spear)
        methods.append(method + mtag)
        cfactors.append(factor)
        dictVal[method_name] = spear

        method = 'IPCW'
        method_name = method
        prefileName = inputName + '_' + method + '_' 
        postfileName = ''
        fullResultFolder = baseInFolder + '/'+ method +'/' + inputName  + '/' +factor
        kendal, spear = getDiversification(fullResultFolder, 5, prefileName, postfileName, outcomeName, perPop)
        kendals.append(kendal)
        spears.append(spear)
        methods.append(method + mtag)
        cfactors.append(factor)
        dictVal[method_name] = spear

        method = 'Cox'
        method_name = method
        prefileName = inputName + '_' + method + '_' 
        postfileName = ''
        fullResultFolder = baseInFolder + '/'+ method +'/' + inputName  + '/' +factor
        kendal, spear = getDiversification(fullResultFolder, 5, prefileName, postfileName, outcomeName, perPop)
        kendals.append(kendal)
        spears.append(spear)
        methods.append(method + mtag)
        cfactors.append(factor)
        dictVal[method_name] = spear

        method = 'Survf'
        method_name = 'SURVF'
        prefileName = inputName + '_' + method + '_' 
        postfileName = ''
        fullResultFolder = baseInFolder + '/'+ method +'/' + inputName  + '/' +factor
        kendal, spear = getDiversification(fullResultFolder, 5, prefileName, postfileName, outcomeName, perPop)
        kendals.append(kendal)
        spears.append(spear)
        methods.append(method + mtag)
        cfactors.append(factor)
        dictVal[method_name] = spear

        
        method = 'DSURV'
        method_name = method
        prefileName = inputName + '_' + method + '_' 
        postfileName = ''
        fullResultFolder = baseInFolder + '/'+ method +'/' + inputName  + '/' +factor
        kendal, spear = getDiversification(fullResultFolder, 5, prefileName, postfileName, outcomeName, perPop)
        kendals.append(kendal)
        spears.append(spear)
        methods.append(method + mtag)
        cfactors.append(factor)
        dictVal[method_name] = spear


        method = 'DMTLR'
        method_name = method
        prefileName = inputName + '_' + method + '_' 
        postfileName = ''
        fullResultFolder = baseInFolder + '/'+ method +'/' + inputName  + '/' +factor
        kendal, spear = getDiversification(fullResultFolder, 5, prefileName, postfileName, outcomeName, perPop)
        kendals.append(kendal)
        spears.append(spear)
        methods.append(method + mtag)
        cfactors.append(factor)
        dictVal[method_name] = spear


        method = 'KSVM'
        method_name = 'SVM'
        prefileName = inputName + '_' + method + '_' 
        postfileName = ''
        fullResultFolder = baseInFolder + '/'+ method +'/' + inputName  + '/' +factor
        kendal, spear = getDiversification(fullResultFolder, 5, prefileName, postfileName, outcomeName, perPop)
        kendals.append(kendal)
        spears.append(spear)
        methods.append(method + mtag)
        cfactors.append(factor)
        dictVal[method_name] = spear
        
        method = 'LRIR-CF'
        method_name = 'LRIR-CF'
        prefileName = inputName + '_' + method + '_' 
        postfileName = ''
        fullResultFolder = baseInFolder + '/'+ method +'/' + inputName  + '/' +factor
        kendal, spear = getDiversification(fullResultFolder, 5, prefileName, postfileName, outcomeName, perPop)
        kendals.append(kendal)
        spears.append(spear)
        methods.append(method + mtag)
        cfactors.append(factor)
        dictVal[method_name] = spear


        method = 'LRIR'
        method_name = 'LRIR'
        prefileName = inputName + '_' + method + '_' 
        postfileName = ''
        fullResultFolder = baseInFolder + '/'+ method +'/' + inputName  + '/' +factor
        kendal, spear = getDiversification(fullResultFolder, 5, prefileName, postfileName, outcomeName, perPop)
        kendals.append(kendal)
        spears.append(spear)
        methods.append(method + mtag)
        cfactors.append(factor)
        dictVal[method_name] = spear
        
        
        q0 = pd.DataFrame(dictVal, index =[0]) 
        
        res = pd.concat([q0, res]).reset_index(drop = True)        
                
    return res


random.seed(20)
np.random.seed(20)

if(plotGBar == True):
    print('plotGBarML')
    plt.clf()    
  
    figure, allAxis = plt.subplots(2, 2, figsize=(8,6),  sharey=False, dpi=300)
    allAxis = allAxis.flatten()
    gcount = -1
    
    
    
    outcomeName = 'time_spend_company'
    inputName = 'hr'
    #factors = ['satisfaction_level', 'average_montly_hours', 'bestFactor']
    factors = ['bestFactor']
    datasetName = 'HR'
    minY = None
    plotAUUCBar(inputName, factors, outcomeName, allAxis, datasetName, gcount, minY, includeFac=False)
    appResults = os.path.join(baseInFolder, 'PerformanceEval')
    gcount += len(factors)
    
    
    
    
    outcomeName = 'Long_emp_retention'
    inputName = 'turnover'
    #factors = [ 'Extraversion', 'Grey_wage', 'bestFactor']
    
    factors = [ 'bestFactor']
    
    datasetName = 'TO'
    minY = None
    plotAUUCBar(inputName, factors, outcomeName, allAxis, datasetName, gcount, minY, includeFac=False)
    appResults = os.path.join(baseInFolder, 'PerformanceEval')
    gcount += len(factors)

    
    outcomeName = 'days'
    inputName = 'ACTG175'
    factors = [ 'bestFactor']
    datasetName = 'ACTG175'
    minY = None
    plotAUUCBar(inputName, factors, outcomeName, allAxis, datasetName, gcount, minY, includeFac=False)
    appResults = os.path.join(baseInFolder, 'PerformanceEval')
    gcount += len(factors)
    
    
    
    outcomeName = 'outcome'
    inputName = 'gbsg'
    factors = [ 'bestFactor']
    datasetName = 'GBSG'
    minY = None# 2
    plotAUUCBar(inputName, factors, outcomeName, allAxis, datasetName, gcount, minY, includeFac=False)
    appResults = os.path.join(baseInFolder, 'PerformanceEval')
    gcount += len(factors) 
        
    
    figure.tight_layout(pad=1.0)
    figure.patch.set_facecolor('w')
    plt.savefig(appResults + '/'  + 'PublicAUUC.png', dpi=300, facecolor = 'w')
    
    
if(plotTrend == True):

    print('plotTrendML')

    plt.clf()    

    figure, allAxis = plt.subplots(4, 9, figsize=(8,6),  sharey='row', dpi=300)   
    allAxis = allAxis.flatten()
    gcount = 0
    
    outcomeName = 'Long_emp_retention'
    inputName = 'turnover'
    factors = ['bestFactor']
    datasetName = 'TO'
    minY = -20
    maxY = 35
    plotTrendLine(inputName, factors, outcomeName, allAxis, datasetName, gcount, minY, maxY, includeFac=False)
    allAxis[gcount].set_ylabel('OCI', fontsize=12)
    gcount += 9
    
    
    outcomeName = 'time_spend_company'
    inputName = 'hr'
    factors = ['bestFactor']
    datasetName = 'HR'
    minY = -15
    maxY = 30
    plotTrendLine(inputName, factors, outcomeName, allAxis, datasetName, gcount, minY, maxY, includeFac=False)
    allAxis[gcount].set_ylabel('OCI', fontsize=12)
    gcount += 9     
    
    
    outcomeName = 'outcome'
    inputName = 'gbsg'
    factors = ['bestFactor']
    datasetName = 'GBSG'
    minY = -6
    maxY = 20
    plotTrendLine(inputName, factors, outcomeName, allAxis, datasetName, gcount, minY, maxY,includeFac=False)
    allAxis[gcount].set_ylabel('OCI', fontsize=12)
    gcount += 9      
    
    outcomeName = 'days'
    inputName = 'ACTG175'
    factors = ['bestFactor']
    datasetName = 'ACTG175'
    minY = -55
    maxY = 200
    plotTrendLine(inputName, factors, outcomeName, allAxis, datasetName, gcount, minY, maxY, xlabel=True, includeFac=False)
    allAxis[gcount].set_ylabel('OCI', fontsize=12)
    gcount += 9  
    
    
    figure.tight_layout(w_pad=0.2, h_pad=1.5)
    figure.subplots_adjust(top=0.9)
    appResults = os.path.join(baseInFolder, 'PerformanceEval')
    figure.patch.set_facecolor('w')
    plt.savefig(appResults + '/'  + 'PublicTrend.png', dpi=300, facecolor = 'w')    
    
    
if(Rank):

    results = []
    outcomeName = 'Long_emp_retention'
    inputName = 'turnover'
    factors = ['bestFactor']
    datasetName = 'Turnover'
    res = generateRank(inputName, factors, outcomeName, datasetName)
    results.append(res)

    outcomeName = 'outcome'
    inputName = 'gbsg'
    factors = ['treatment']
    datasetName = 'GBSG'
    res = generateRank(inputName, factors, outcomeName, datasetName)
    results.append(res)
    
    
    outcomeName = 'time_spend_company'
    inputName = 'hr'
    factors = ['bestFactor']
    datasetName = 'Hr'
    res = generateRank(inputName, factors, outcomeName, datasetName)
    results.append(res)
    
    
    outcomeName = 'days'
    inputName = 'ACTG175'
    factors = ['bestFactor']
    datasetName = 'ACTG175'
    res = generateRank(inputName, factors, outcomeName, datasetName)
    results.append(res)    
    
    results = pd.concat(results).reset_index(drop = True)
    results.loc['Average'] = results.mean()
    results = results[['Dataset', 'Cox', 'IPCW', 'SCF', 'DSURV',  'DMTLR', 'SURVF', 'SVM', 'LRIR-CF', 'LRIR']]
    appResults = os.path.join(baseInFolder, 'PerformanceEval')
    results.to_csv(appResults + '/'  + 'Spearman.csv', index=False) 