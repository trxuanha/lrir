source("LiftUtility.R")
library(foreach)
library(doParallel)



inputName<- 'ACTG175'
target <- 'days'
manipulableFactorNames<- c('arms')
manipulableFactorValues<- list(c(0, 1, 2, 3))

excludeVars <- c()

eventName <- 'cens'
baseFolder = getwd()
methods <- c('CF', 'Cox', 'IPCW')

fold <- 5
splitRule = ''
cvRule = ''
inputBase<- paste (baseFolder, '/input/', inputName, '/split', sep='') 



foreach(icount=1:length(methods))  %dopar% {
  
  method <- methods[icount]
  outputBase <- paste (baseFolder, '/output/', method,'/', inputName, sep='')
  dir.create(file.path(outputBase), showWarnings = FALSE) 
  doCrossValSurvNoDAG(inputName, target, fold, inputBase, outputBase, manipulableFactorNames, manipulableFactorValues, method, eventName, excludeVars = excludeVars ) 
  
}

