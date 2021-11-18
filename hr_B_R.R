source("LiftUtility.R")
library(foreach)
library(doParallel)



inputName<- 'hr'
target <- 'time_spend_company'
manipulableFactorNames<- c('satisfaction_level', 'average_montly_hours')
manipulableFactorValues<- list(c(0, 1, 2),
                               c(0, 1, 2, 3, 4))

excludeVars <- c()

eventName <- 'left'
baseFolder = getwd()
methods <- c('CF')
methods <- c('VT')
methods <- c('SRC')
methods <- c('IPCW')
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

