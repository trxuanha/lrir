source("LiftUtility.R")
library(foreach)
library(doParallel)



inputName<- 'turnover'
target <- 'Long_emp_retention'
manipulableFactorNames<- c('Extraversion', 'Grey_wage')
manipulableFactorValues<- list(c(0, 1, 2, 3),
                               c(0,1))

excludeVars <- c()

eventName <- 'event'
baseFolder = getwd()
methods <- c('SRC')
methods <- c('VT')
methods <- c('CF', 'Cox', 'IPCW')


fold <- 5
splitRule = ''
cvRule = ''
inputBase<- paste (baseFolder, '/input/', inputName, '/split', sep='') 


foreach(icount=1:length(methods)) %dopar% {
  
  method <- methods[icount]
  outputBase <- paste (baseFolder, '/output/', method,'/', inputName, sep='')
  dir.create(file.path(outputBase), showWarnings = FALSE) 

  doCrossValSurvNoDAG(inputName, target, fold, inputBase, outputBase, manipulableFactorNames, manipulableFactorValues, method, eventName, excludeVars = excludeVars )   
  
  
}

