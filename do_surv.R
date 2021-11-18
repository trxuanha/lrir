library(ini)
library(survival)
library(survminer)
library(patchwork)

args = commandArgs(trailingOnly=TRUE)

if (length(args)==0) {
  #stop("At least one argument must be supplied (input file).n", call.=FALSE)
}

configName = args[1]

baseFolder = getwd()
iniFile <-  paste (baseFolder, '/config/', configName, '.ini', sep='')

config <- read.ini(iniFile)

inputName<- config$dataset$inputName
target <- config$dataset$target
fold <- config$parameter$fold

manipulableFactorNames <- config$dataset$manipulableFactorNames
manipulableFactorNames <- strsplit(manipulableFactorNames, ",")[[1]]

temp_manipulableFactorValues <- config$dataset$manipulableFactorValues
temp_manipulableFactorValues <- strsplit(temp_manipulableFactorValues, ";")[[1]]

manipulableFactorValues<- list()

for (item in temp_manipulableFactorValues){
  
  values <- strsplit(item, ",")[[1]]
  manipulableFactorValues <- c(manipulableFactorValues, list(as.numeric(values)))
}

excludeVars <- config$dataset$excludeVars

if(is.null (excludeVars)){
  excludeVars <- c()
}else{
  excludeVars <- strsplit(excludeVars, ",")[[1]]
}

eventName <- config$dataset$event
fold <- as.numeric(config$parameter$fold)

factor = 'bestFactor'


dataforPlot = NULL

getSurvPlot <-function(inData, targetVar, eventName, method, topPercent){
  
  liftSurvivalData <-data.frame(inData)
  liftSurvivalData <- liftSurvivalData[order(-liftSurvivalData$LIFT_SCORE),]
  topRow = as.integer(nrow(liftSurvivalData)*topPercent)
  liftSurvivalData <- liftSurvivalData[1:topRow, ]
  dataforPlot <<- liftSurvivalData
  
  
  dataforPlot <<- dataforPlot[c(targetVar, eventName, 'FOLLOW_REC')]
  colnames(dataforPlot) <- c('time','status', 'FOLLOW_REC')
  
  fit2 <- survfit(Surv(time = dataforPlot$time, event = dataforPlot$status)
                  ~ FOLLOW_REC , data = dataforPlot)
  
  
  
  #surv_diff <- survdiff(Surv(time = dataforPlot$time, event = dataforPlot$status)
  #                      ~ FOLLOW_REC, data = dataforPlot, rho = 0)
  
  
  #print(inData)
  
  median0 = NA
  median1 = NA
  tryCatch(
    expr = {
      median0 = summary(fit2)$table[, "median"][1]
      median1 = summary(fit2)$table[, "median"][2]
    },
    error = function(e){
      message('Caught an error!')
      print(e)
      
      return (NA)
    }
  )  
  
  
    result = list()
    result$median0 = median0
    result$median1 = median1
	  
  return (result)
}

methods <- c('Cox','IPCW','CF', 'DSURV', 'DMTLR', 'Survf', 'KSVM', 'LRIR-CF', 'LRIR')

allGraph <- NULL

outputF <- 'output'



top25 = c()



method_names = c()
for (topPercent in c(0.25)){
  for (method in methods){
    
    outputBase <- paste(baseFolder, '/',outputF,'/', method,'/', inputName ,sep='')
    
    combinedData <- NULL
    
    
    totalMean <- 0
	
	curFold = fold
    for(icount in 1:fold ){
      
      inFilePath <- paste (outputBase, '/', factor , '/', inputName,'_', method,  '_', icount,'.csv' , sep='')
      
      inData <- read.csv(file = inFilePath)
	  
	     
      dataforPlot <- NULL
      mean_res <- getSurvPlot(inData, target, eventName, method, topPercent)
      
	  mean_surv <- mean_res$median1 - mean_res$median0
	  
	  if(is.na(mean_surv)){
		curFold = curFold -1
		next
		
	  }
      totalMean <- totalMean + mean_surv
	  
	  totalMean0 <- mean_res$median0
	  
	  totalMean1 <- mean_res$median1
      
    }
	

    
	
    outstr <- paste('means survival time for method ', method , ' is:' , totalMean/curFold ,sep='')
    
    print(outstr)
    
    method_name = method
	  if(method == 'KSVM'){
		method_name = 'SVM'
	  } 	  

	  if(method == 'CF'){
		method_name = 'SCF'
	  } 

	  if(method == 'Survf'){
		method_name = 'SURVF'
	  } 	
	  method_names = c(method_names, method_name)
      top25 = c(top25, totalMean/curFold)
	  
  }
  
  
}
  

df <- data.frame(Method = method_names,
                 Surv_Increase = top25
)



outputTrueBase <- paste(baseFolder, '/',outputF,'/PerformanceEval/' ,sep='')
outFileName <- paste(inputName, '_MeanSurv','.csv' ,sep='')
print(paste0(outputTrueBase, outFileName))
write.csv(df, paste0(outputTrueBase, outFileName), row.names = FALSE)

