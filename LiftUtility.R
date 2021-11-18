library(randomForestSRC)
library(survival)
library(causalTree)
library(grf)
library(rpart.plot)
library(graph)



#' Compute E[T | X]
#'
#' @param S.hat The estimated survival curve.
#' @param Y.grid The time values corresponding to S.hat.
#' @return A vector of expected values.
expected_survival <- function(S.hat, Y.grid) {
  grid.diff <- diff(c(0, Y.grid, max(Y.grid)))
  
  c(cbind(1, S.hat) %*% grid.diff)
}


doCrossValSurv<- function(inputName, targetVar, matrixPath, fold, inputBase, outputBase, manipulableFactorNames, manipulableFactorValues, testType, eventName, excludeVars=c(), SuvrTest=F){
  
  set.seed(20)
  
  if(length(excludeVars > 0)){
    print('excludeVars')
    print(excludeVars)
  }
  
  
  separation<- '_vv_'
  
  causalMax <- read.csv(file = matrixPath)
  causalMax <- as(causalMax, "matrix")
  mygraph <- as(causalMax,"graphNEL")
  nattrs <- list(node=list(shape="ellipse", fixedsize=FALSE, fontsize=20))
  plot(mygraph, attrs=nattrs)
  
  for(timecn in 1: fold){
    trainFile <- paste (inputBase,'/', inputName, '_train_',timecn,'.csv', sep='') 
    oriTrainDat <- read.csv(file = trainFile)
    trainDat<- dplyr::select(oriTrainDat, -excludeVars)
    testFile <- paste (inputBase,'/', inputName, '_test_',timecn,'.csv', sep='')
    orginalTestDat<- read.csv(file = testFile)
    if(!SuvrTest){
      orginalTestDat <- dplyr::filter(orginalTestDat, !!as.name(eventName ) == 1 )
    }
    
    
    potentialCauses <- inEdges(targetVar, mygraph)
    
    causes <- c()
    for(cause in potentialCauses[[targetVar]]){
      if( cause %in%  manipulableFactorNames){
        causes <- c(causes, cause)
      }
    }
    
    covariates = unique(setdiff(colnames(trainDat), c(targetVar, causes, eventName)))
    globalPredictions <- NULL
    globalTempColNames <- c()
    
    for (factor in causes) {
      parentsOfCause <- inEdges(factor, mygraph)
      parentsOfTarget <- setdiff (potentialCauses[[targetVar]], factor)
      allParents = unique(c(parentsOfCause[[factor]], parentsOfTarget))
      
      vIndex <- -1
      for(icount in 1: length(manipulableFactorNames)){
        if(manipulableFactorNames[icount] == factor){
          vIndex <- icount
          break
        }
      }
      
      predictions <- NULL
      causaVals <- unlist(manipulableFactorValues[vIndex])
      tempColNames <- c()
      
      if(testType == 'SRC'){
        
        tempDat = data.frame(trainDat)
        tempDat = dplyr::select(tempDat, c(targetVar, eventName, covariates, factor) )
        
        colnames(tempDat)[which(names(tempDat) == targetVar)] <- "Y"
        colnames(tempDat)[which(names(tempDat) == eventName)] <- "D"
        df <- tempDat
        
        ntree <- 500   
        fit <- rfsrc(Surv(Y, D) ~ ., ntree = ntree, nsplit = 0, data = df)
        
        testDat <- data.frame(orginalTestDat)
        n.test <- nrow(testDat)
        
        baseline <- rep(causaVals[1], n.test)
        dfTest <- data.frame(cbind(testDat[covariates], tempNi999 = baseline))
        colnames(dfTest)[which(names(dfTest) == 'tempNi999')] <- factor
        
        baseline_Y <- predict(fit, newdata=dfTest)
        
        for (ival in 2: length(causaVals)){
          
          curLevel <- rep(causaVals[ival], n.test)
          dfTest <- data.frame(cbind(testDat[covariates], tempNi999 = curLevel))
          colnames(dfTest)[which(names(dfTest) == 'tempNi999')] <- factor
          curLevel_Y <- predict(fit, newdata=dfTest)
          
          res <- expected_survival(curLevel_Y$survival, curLevel_Y$time.interest) - expected_survival(baseline_Y$survival, baseline_Y$time.interest)
          assign(paste0(factor, separation, causaVals[ival]) , res)      
          

          if (is.null(predictions)){
            predictions <- data.frame(eval( as.name (paste0(factor, separation, causaVals[ival]) ) )   )
          }else{
            predictions <- cbind (predictions, eval( as.name (paste0(factor, separation, causaVals[ival]) ) ) )
          }
          
          if (is.null(globalPredictions)){
            globalPredictions <- data.frame(eval( as.name (paste0(factor, separation, causaVals[ival]) ) )   )
          }else{
            globalPredictions <- cbind (globalPredictions, eval( as.name (paste0(factor, separation, causaVals[ival]) ) ) )
          }
          
          
          tempColNames <- c(tempColNames, paste0(factor, separation, causaVals[ival]))
          globalTempColNames <- c(globalTempColNames, paste0(factor, separation, causaVals[ival]))
          
        }## end of causaVals

        
      }
      
      
      
      
      if(testType == 'Cox'){
        
        tempDat = data.frame(trainDat)
        tempDat = dplyr::select(tempDat, c(targetVar, eventName, covariates, factor) )
        
        colnames(tempDat)[which(names(tempDat) == targetVar)] <- "Y"
        colnames(tempDat)[which(names(tempDat) == eventName)] <- "D"
        df <- tempDat
        
        fit <- coxph(Surv(time=Y, event=D) ~ .  , data=df)
        testDat <- data.frame(orginalTestDat)
        n.test <- nrow(testDat)
        
        baseline <- rep(causaVals[1], n.test)
        dfTest <- data.frame(cbind(testDat[covariates], tempNi999 = baseline))
        colnames(dfTest)[which(names(dfTest) == 'tempNi999')] <- factor
        
        baseline_Y <- predict(fit, type="risk",se.fit=FALSE, newdata=dfTest)
        
        for (ival in 2: length(causaVals)){
          
          curLevel <- rep(causaVals[ival], n.test)
          dfTest <- data.frame(cbind(testDat[covariates], tempNi999 = curLevel))
          colnames(dfTest)[which(names(dfTest) == 'tempNi999')] <- factor
          curLevel_Y <- predict(fit, type="risk",se.fit=FALSE, newdata=dfTest)
          
          res <- curLevel_Y - baseline_Y
          
          assign(paste0(factor, separation, causaVals[ival]) , -res)  
          
          if (is.null(predictions)){
            predictions <- data.frame(eval( as.name (paste0(factor, separation, causaVals[ival]) ) )   )
          }else{
            predictions <- cbind (predictions, eval( as.name (paste0(factor, separation, causaVals[ival]) ) ) )
          }
          
          if (is.null(globalPredictions)){
            globalPredictions <- data.frame(eval( as.name (paste0(factor, separation, causaVals[ival]) ) )   )
          }else{
            globalPredictions <- cbind (globalPredictions, eval( as.name (paste0(factor, separation, causaVals[ival]) ) ) )
          }
          
          
          tempColNames <- c(tempColNames, paste0(factor, separation, causaVals[ival]))
          globalTempColNames <- c(globalTempColNames, paste0(factor, separation, causaVals[ival]))
          
        }## end of causaVals
        
        
      }# Cox      
      
      if((testType == 'CF') || (testType == 'VT') || (testType == 'IPCW') ){
        
        for (ival in 2: length(causaVals)){
          f_train_data <- dplyr::filter(trainDat, !!as.name(factor ) %in% c( causaVals[1], causaVals[ival] ) )  
          f_train_data[factor] <- ifelse(f_train_data[factor] == causaVals[1],0, 1)
          testDat <- data.frame(orginalTestDat)
          
          if(testType == 'CF'){
            
            forest <- causal_survival_forest(X = f_train_data[covariates], Y = f_train_data[[targetVar]], 
                                             W = f_train_data[[factor]],
                                             D = f_train_data[[eventName]]
            )
            
            assign(paste0(factor, separation, causaVals[ival]) , predict(forest, newdata = testDat[covariates])) 
            
          }
          
          if(testType == 'IPCW'){
            
            sf.censor = survival_forest(f_train_data[covariates], f_train_data[[targetVar]], 1 - f_train_data[[eventName]], prediction.type = "Nelson-Aalen",
                                        num.trees = 500)
            
            C.hat = predict(sf.censor)$predictions
            
            Y.relabeled = sf.censor$Y.relabeled
            
            C.Y.hat = rep(1, nrow(f_train_data[covariates])) # (for events before the first failure, C.Y.hat is one)
            
            
            C.Y.hat[Y.relabeled != 0] = C.hat[cbind(1:nrow(f_train_data[covariates]), Y.relabeled)]
            sample.weights = 1 / C.Y.hat
            subset = f_train_data[[eventName]] == 1
            cf = causal_forest(f_train_data[subset, covariates], f_train_data[subset, targetVar], f_train_data[subset, factor], sample.weights = sample.weights[subset])
            p = predict(cf, testDat[covariates])
            assign(paste0(factor, separation, causaVals[ival]) , p)
            
            
            
          }
          
          if(testType == 'VT'){
            
            ntree = 500
            df0 = dplyr::filter(f_train_data,!!as.name(factor )== 0 )
            df0 = dplyr::select(df0, c(targetVar, eventName, covariates) )
            colnames(df0)[which(names(df0) == targetVar)] <- "Y"
            colnames(df0)[which(names(df0) == eventName)] <- "D"  
            fit0 = rfsrc(Surv(Y, D) ~ ., ntree = ntree, nsplit = 0, data = df0)
            
            
            df1 = dplyr::filter(f_train_data,!!as.name(factor )== 1 )
            df1 = dplyr::select(df1, c(targetVar, eventName, covariates) )
            colnames(df1)[which(names(df1) == targetVar)] <- "Y"
            colnames(df1)[which(names(df1) == eventName)] <- "D"  
            fit1 = rfsrc(Surv(Y, D) ~ ., ntree = ntree, nsplit = 0, data = df1)            
            
            
            baseline_Y <- predict(fit0, newdata=testDat[covariates])
            curLevel_Y <- predict(fit1, newdata=testDat[covariates])
            
            res <- expected_survival(curLevel_Y$survival, curLevel_Y$time.interest) -
              expected_survival(baseline_Y$survival, baseline_Y$time.interest)
            
            assign(paste0(factor, separation, causaVals[ival]) , res)    
            
          }
  
          
          
          if (is.null(predictions)){
            predictions <- data.frame(eval( as.name (paste0(factor, separation, causaVals[ival]) ) )   )
          }else{
            predictions <- cbind (predictions, eval( as.name (paste0(factor, separation, causaVals[ival]) ) ) )
          }
          
          if (is.null(globalPredictions)){
            globalPredictions <- data.frame(eval( as.name (paste0(factor, separation, causaVals[ival]) ) )   )
          }else{
            globalPredictions <- cbind (globalPredictions, eval( as.name (paste0(factor, separation, causaVals[ival]) ) ) )
          }
          
          
          tempColNames <- c(tempColNames, paste0(factor, separation, causaVals[ival]))
          globalTempColNames <- c(globalTempColNames, paste0(factor, separation, causaVals[ival]))
          
          
        }## end of causaVals
        
        
      }
      
      
      
      
      colnames(predictions) <- tempColNames
      predictions['LIFT_SCORE'] <- apply(predictions, 1, max)
      predictions['TREATMENT_NAME'] <- colnames(predictions[tempColNames])[apply(predictions[tempColNames], 1, which.max)]
      
      testDat <- cbind(testDat, predictions)
      testDat <- estimateUplift(testDat, targetVar, causaVals[1])
      
      factOutputBase <- paste (outputBase, '/', factor ,sep='')
      dir.create(file.path(outputBase, factor), showWarnings = FALSE)
      outFileName <- paste (inputName, '_',testType,'_',timecn , '.csv', sep='') 
      fullPath <- paste(c(factOutputBase,'/', outFileName ), collapse = "")
      write.csv(testDat, fullPath, row.names = FALSE)
    }## End for causes
    
    colnames(globalPredictions) <- globalTempColNames
    globalPredictions['LIFT_SCORE'] <- apply(globalPredictions, 1, max)
    
    globalPredictions['TREATMENT_NAME'] <- colnames(globalPredictions[globalTempColNames])[apply(globalPredictions[globalTempColNames], 1, which.max)]
    testBestDat <- data.frame(orginalTestDat)
    testBestDat <- cbind(testBestDat, globalPredictions)
    testBestDat <- estimateBestUplift(testBestDat, targetVar)
    
    factOutputBase <- paste (outputBase, '/', 'bestFactor' ,sep='')
    dir.create(file.path(outputBase, 'bestFactor'), showWarnings = FALSE)
    outFileName <- paste (inputName, '_',testType,'_',timecn , '.csv', sep='') 
    fullPath <- paste(c(factOutputBase,'/', outFileName ), collapse = "")
    write.csv(testBestDat, fullPath, row.names = FALSE)
    
    
    
    ##############
    
  }
}

#################################


doCrossValSurvNoDAG<- function(inputName, targetVar,  fold, inputBase, outputBase, manipulableFactorNames, manipulableFactorValues, testType, eventName, excludeVars=c(), SuvrTest=F){
  
  set.seed(20)
  
  if(length(excludeVars > 0)){
    print('excludeVars')
    print(excludeVars)
  }
  
  
  separation<- '_vv_'
  
  
  for(timecn in 1: fold){
    trainFile <- paste (inputBase,'/', inputName, '_train_',timecn,'.csv', sep='') 
    oriTrainDat <- read.csv(file = trainFile)
    trainDat<- dplyr::select(oriTrainDat, -excludeVars)
    testFile <- paste (inputBase,'/', inputName, '_test_',timecn,'.csv', sep='')
    orginalTestDat<- read.csv(file = testFile)
    if(!SuvrTest){
		print('Include complete data')
      orginalTestDat <- dplyr::filter(orginalTestDat, !!as.name(eventName ) == 1 )
    }else{
	
		print('Include censoring data')
	
	}
        
    causes <- manipulableFactorNames
    
    covariates = unique(setdiff(colnames(trainDat), c(targetVar, causes, eventName)))
    globalPredictions <- NULL
    globalTempColNames <- c()
    
    for (factor in causes) {
      
      vIndex <- -1
      for(icount in 1: length(manipulableFactorNames)){
        if(manipulableFactorNames[icount] == factor){
          vIndex <- icount
          break
        }
      }
      
      predictions <- NULL
      causaVals <- unlist(manipulableFactorValues[vIndex])
      tempColNames <- c()
      
      if(testType == 'SRC'){
        
        tempDat = data.frame(trainDat)
        tempDat = dplyr::select(tempDat, c(targetVar, eventName, covariates, factor) )
        
        colnames(tempDat)[which(names(tempDat) == targetVar)] <- "Y"
        colnames(tempDat)[which(names(tempDat) == eventName)] <- "D"
        df <- tempDat
        
        ntree <- 500   
        fit <- rfsrc(Surv(Y, D) ~ ., ntree = ntree, nsplit = 0, data = df)
        
        testDat <- data.frame(orginalTestDat)
        n.test <- nrow(testDat)
        
        baseline <- rep(causaVals[1], n.test)
        dfTest <- data.frame(cbind(testDat[covariates], tempNi999 = baseline))
        colnames(dfTest)[which(names(dfTest) == 'tempNi999')] <- factor
        
        baseline_Y <- predict(fit, newdata=dfTest)
        
        for (ival in 2: length(causaVals)){
          
          curLevel <- rep(causaVals[ival], n.test)
          dfTest <- data.frame(cbind(testDat[covariates], tempNi999 = curLevel))
          colnames(dfTest)[which(names(dfTest) == 'tempNi999')] <- factor
          curLevel_Y <- predict(fit, newdata=dfTest)
          
          res <- expected_survival(curLevel_Y$survival, curLevel_Y$time.interest) - expected_survival(baseline_Y$survival, baseline_Y$time.interest)
          assign(paste0(factor, separation, causaVals[ival]) , res)      
          

          if (is.null(predictions)){
            predictions <- data.frame(eval( as.name (paste0(factor, separation, causaVals[ival]) ) )   )
          }else{
            predictions <- cbind (predictions, eval( as.name (paste0(factor, separation, causaVals[ival]) ) ) )
          }
          
          if (is.null(globalPredictions)){
            globalPredictions <- data.frame(eval( as.name (paste0(factor, separation, causaVals[ival]) ) )   )
          }else{
            globalPredictions <- cbind (globalPredictions, eval( as.name (paste0(factor, separation, causaVals[ival]) ) ) )
          }
          
          
          tempColNames <- c(tempColNames, paste0(factor, separation, causaVals[ival]))
          globalTempColNames <- c(globalTempColNames, paste0(factor, separation, causaVals[ival]))
          
        }## end of causaVals

        
      }
      
      
      
      
      if(testType == 'Cox'){
        
        tempDat = data.frame(trainDat)
        tempDat = dplyr::select(tempDat, c(targetVar, eventName, covariates, factor) )
        
        colnames(tempDat)[which(names(tempDat) == targetVar)] <- "Y"
        colnames(tempDat)[which(names(tempDat) == eventName)] <- "D"
        df <- tempDat
        
        fit <- coxph(Surv(time=Y, event=D) ~ .  , data=df)
        testDat <- data.frame(orginalTestDat)
        n.test <- nrow(testDat)
        
        baseline <- rep(causaVals[1], n.test)
        dfTest <- data.frame(cbind(testDat[covariates], tempNi999 = baseline))
        colnames(dfTest)[which(names(dfTest) == 'tempNi999')] <- factor
        
        baseline_Y <- predict(fit, type="risk",se.fit=FALSE, newdata=dfTest)
        
        for (ival in 2: length(causaVals)){
          
          curLevel <- rep(causaVals[ival], n.test)
          dfTest <- data.frame(cbind(testDat[covariates], tempNi999 = curLevel))
          colnames(dfTest)[which(names(dfTest) == 'tempNi999')] <- factor
          curLevel_Y <- predict(fit, type="risk",se.fit=FALSE, newdata=dfTest)
          
          res <- curLevel_Y - baseline_Y
          
          assign(paste0(factor, separation, causaVals[ival]) , -res)  
          
          if (is.null(predictions)){
            predictions <- data.frame(eval( as.name (paste0(factor, separation, causaVals[ival]) ) )   )
          }else{
            predictions <- cbind (predictions, eval( as.name (paste0(factor, separation, causaVals[ival]) ) ) )
          }
          
          if (is.null(globalPredictions)){
            globalPredictions <- data.frame(eval( as.name (paste0(factor, separation, causaVals[ival]) ) )   )
          }else{
            globalPredictions <- cbind (globalPredictions, eval( as.name (paste0(factor, separation, causaVals[ival]) ) ) )
          }
          
          
          tempColNames <- c(tempColNames, paste0(factor, separation, causaVals[ival]))
          globalTempColNames <- c(globalTempColNames, paste0(factor, separation, causaVals[ival]))
          
        }## end of causaVals
        
        
      }# Cox      
      
      if((testType == 'CF') || (testType == 'VT') || (testType == 'IPCW') ){
        
        for (ival in 2: length(causaVals)){
          f_train_data <- dplyr::filter(trainDat, !!as.name(factor ) %in% c( causaVals[1], causaVals[ival] ) )  
          f_train_data[factor] <- ifelse(f_train_data[factor] == causaVals[1],0, 1)
          testDat <- data.frame(orginalTestDat)
          
          if(testType == 'CF'){
            
            forest <- causal_survival_forest(X = f_train_data[covariates], Y = f_train_data[[targetVar]], min.node.size = 50,
                                             W = f_train_data[[factor]],
                                             D = f_train_data[[eventName]]
            )
            
            assign(paste0(factor, separation, causaVals[ival]) , predict(forest, newdata = testDat[covariates])) 
            
          }
          
          if(testType == 'IPCW'){
            
            sf.censor = survival_forest(f_train_data[covariates], f_train_data[[targetVar]], 1 - f_train_data[[eventName]], prediction.type = "Nelson-Aalen",
                                        num.trees = 500)
            
            C.hat = predict(sf.censor)$predictions
            
            Y.relabeled = sf.censor$Y.relabeled
            
            C.Y.hat = rep(1, nrow(f_train_data[covariates])) # (for events before the first failure, C.Y.hat is one)
            
            
            C.Y.hat[Y.relabeled != 0] = C.hat[cbind(1:nrow(f_train_data[covariates]), Y.relabeled)]
            sample.weights = 1 / C.Y.hat
            subset = f_train_data[[eventName]] == 1
            cf = causal_forest(f_train_data[subset, covariates], f_train_data[subset, targetVar], f_train_data[subset, factor], sample.weights = sample.weights[subset])
            p = predict(cf, testDat[covariates])
            assign(paste0(factor, separation, causaVals[ival]) , p)
            
            
            
          }
          
          if(testType == 'VT'){
            
            ntree = 500
            df0 = dplyr::filter(f_train_data,!!as.name(factor )== 0 )
            df0 = dplyr::select(df0, c(targetVar, eventName, covariates) )
            colnames(df0)[which(names(df0) == targetVar)] <- "Y"
            colnames(df0)[which(names(df0) == eventName)] <- "D"  
            fit0 = rfsrc(Surv(Y, D) ~ ., ntree = ntree, nsplit = 0, data = df0)
            
            
            df1 = dplyr::filter(f_train_data,!!as.name(factor )== 1 )
            df1 = dplyr::select(df1, c(targetVar, eventName, covariates) )
            colnames(df1)[which(names(df1) == targetVar)] <- "Y"
            colnames(df1)[which(names(df1) == eventName)] <- "D"  
            fit1 = rfsrc(Surv(Y, D) ~ ., ntree = ntree, nsplit = 0, data = df1)            
            
            
            baseline_Y <- predict(fit0, newdata=testDat[covariates])
            curLevel_Y <- predict(fit1, newdata=testDat[covariates])
            
            res <- expected_survival(curLevel_Y$survival, curLevel_Y$time.interest) -
              expected_survival(baseline_Y$survival, baseline_Y$time.interest)
            
            assign(paste0(factor, separation, causaVals[ival]) , res)    
            
          }
  
          
          
          if (is.null(predictions)){
            predictions <- data.frame(eval( as.name (paste0(factor, separation, causaVals[ival]) ) )   )
          }else{
            predictions <- cbind (predictions, eval( as.name (paste0(factor, separation, causaVals[ival]) ) ) )
          }
          
          if (is.null(globalPredictions)){
            globalPredictions <- data.frame(eval( as.name (paste0(factor, separation, causaVals[ival]) ) )   )
          }else{
            globalPredictions <- cbind (globalPredictions, eval( as.name (paste0(factor, separation, causaVals[ival]) ) ) )
          }
          
          
          tempColNames <- c(tempColNames, paste0(factor, separation, causaVals[ival]))
          globalTempColNames <- c(globalTempColNames, paste0(factor, separation, causaVals[ival]))
          
          
        }## end of causaVals
        
        
      }
      
      
      
      
      colnames(predictions) <- tempColNames
      predictions['LIFT_SCORE'] <- apply(predictions, 1, max)
      predictions['TREATMENT_NAME'] <- colnames(predictions[tempColNames])[apply(predictions[tempColNames], 1, which.max)]
      
      testDat <- cbind(testDat, predictions)
      testDat <- estimateUplift(testDat, targetVar, causaVals[1])
      
      factOutputBase <- paste (outputBase, '/', factor ,sep='')
      dir.create(file.path(outputBase, factor), showWarnings = FALSE)
	  
	  if(SuvrTest){
		outFileName <- paste (inputName, '_',testType,'_Cen_',timecn , '.csv', sep='') 
	  }else{
		outFileName <- paste (inputName, '_',testType,'_',timecn , '.csv', sep='') 
	  }
      
	  
	  
	  
      fullPath <- paste(c(factOutputBase,'/', outFileName ), collapse = "")
      write.csv(testDat, fullPath, row.names = FALSE)
    }## End for causes
    
    colnames(globalPredictions) <- globalTempColNames
    globalPredictions['LIFT_SCORE'] <- apply(globalPredictions, 1, max)
    
    globalPredictions['TREATMENT_NAME'] <- colnames(globalPredictions[globalTempColNames])[apply(globalPredictions[globalTempColNames], 1, which.max)]
    testBestDat <- data.frame(orginalTestDat)
    testBestDat <- cbind(testBestDat, globalPredictions)
    testBestDat <- estimateBestUplift(testBestDat, targetVar)
    
    factOutputBase <- paste (outputBase, '/', 'bestFactor' ,sep='')
    dir.create(file.path(outputBase, 'bestFactor'), showWarnings = FALSE)
	if(SuvrTest){
		outFileName <- paste (inputName, '_',testType,'_Cen_',timecn , '.csv', sep='') 
	}else{
		outFileName <- paste (inputName, '_',testType,'_',timecn , '.csv', sep='') 
	}
	  
	
    fullPath <- paste(c(factOutputBase,'/', outFileName ), collapse = "")
    write.csv(testBestDat, fullPath, row.names = FALSE)
    
    
    
    ##############
    
  }
}

#################################

doCrossValTreesML<- function(inputName, targetVar, matrixPath, fold, inputBase, outputBase, manipulableFactorNames, manipulableFactorValues, testType, excludeVars=c()){
  
  if(length(excludeVars > 0)){
    print('excludeVars')
    print(excludeVars)
  }
  if(testType == 'CT'){
    splitRule = 'CT'
    cvRule = 'CT'
  }
  if(testType == 'TOT'){
    splitRule = 'TOT'
    cvRule = 'TOT'
  }
  if(testType == 'TST'){
    splitRule = 'tstats'
    cvRule = 'fit'
  }
  if(testType == 'FT'){
    splitRule = 'fit'
    cvRule = 'fit'
  }    
  
  separation<- '_vv_'
  
  causalMax <- read.csv(file = matrixPath)
  causalMax <- as(causalMax, "matrix")
  mygraph <- as(causalMax,"graphNEL")
  nattrs <- list(node=list(shape="ellipse", fixedsize=FALSE, fontsize=20))
  plot(mygraph, attrs=nattrs)
  
  for(timecn in 1: fold){
    trainFile <- paste (inputBase,'/', inputName, '_train_',timecn,'.csv', sep='') 
    oriTrainDat <- read.csv(file = trainFile)
    trainDat<- dplyr::select(oriTrainDat, -excludeVars)
    testFile <- paste (inputBase,'/', inputName, '_test_',timecn,'.csv', sep='')
    orginalTestDat<- read.csv(file = testFile)
    
    potentialCauses <- inEdges(targetVar, mygraph)
    
    causes <- c()
    for(cause in potentialCauses[[targetVar]]){
      if( cause %in%  manipulableFactorNames){
        causes <- c(causes, cause)
      }
    }
    
    covariates = unique(setdiff(colnames(trainDat), c(targetVar, causes)))
    globalPredictions <- NULL
    globalTempColNames <- c()
    
    for (factor in causes) {
      parentsOfCause <- inEdges(factor, mygraph)
      parentsOfTarget <- setdiff (potentialCauses[[targetVar]], factor)
      allParents = unique(c(parentsOfCause[[factor]], parentsOfTarget))
      
      vIndex <- -1
      for(icount in 1: length(manipulableFactorNames)){
        if(manipulableFactorNames[icount] == factor){
          vIndex <- icount
          break
        }
      }
      
      predictions <- NULL
      causaVals <- unlist(manipulableFactorValues[vIndex])
      tempColNames <- c()
      
      for (ival in 2: length(causaVals)){
        f_train_data <- dplyr::filter(trainDat, !!as.name(factor ) %in% c( causaVals[1], causaVals[ival] ) )  
        f_train_data[factor] <- ifelse(f_train_data[factor] == causaVals[1],0, 1)
        testDat <- data.frame(orginalTestDat)
        
        if(testType == 'CF'){
          
          forest <- causal_forest(X = f_train_data[covariates], Y = f_train_data[[targetVar]], 
                                  W = f_train_data[[factor]],
          )
          
          
          assign(paste0(factor, separation, causaVals[ival]) , predict(forest, newdata = testDat[covariates]))          
        } else{
          
          form.ps = paste(factor,'~ ' ,paste(allParents, collapse=' + '),  collapse='' )
          
          reg<-glm(as.formula(form.ps)
                   , family=binomial
                   , data=f_train_data)  
          propensity_scores = reg$fitted
          
          form.ps = paste(targetVar, paste(covariates, collapse = "+"), sep = "~")
          
          tree <- causalTree(as.formula(form.ps)
                             , data = f_train_data, treatment = f_train_data[[factor]],
                             split.Rule = splitRule, cv.option = cvRule, split.Honest = T, cv.Honest = T, split.Bucket = F, 
                             xval = 5, cp = 0,  propensity = propensity_scores
          )
          assign(paste0(factor, separation, causaVals[ival]) , predict(tree, newdata = testDat[covariates]))    
        } 
        
        if (is.null(predictions)){
          predictions <- data.frame(eval( as.name (paste0(factor, separation, causaVals[ival]) ) )   )
        }else{
          predictions <- cbind (predictions, eval( as.name (paste0(factor, separation, causaVals[ival]) ) ) )
        }
        
        if (is.null(globalPredictions)){
          globalPredictions <- data.frame(eval( as.name (paste0(factor, separation, causaVals[ival]) ) )   )
        }else{
          globalPredictions <- cbind (globalPredictions, eval( as.name (paste0(factor, separation, causaVals[ival]) ) ) )
        }
        
        
        tempColNames <- c(tempColNames, paste0(factor, separation, causaVals[ival]))
        globalTempColNames <- c(globalTempColNames, paste0(factor, separation, causaVals[ival]))
        
        
      }## end of causaVals
      
      colnames(predictions) <- tempColNames
      predictions['LIFT_SCORE'] <- apply(predictions, 1, max)
      predictions['TREATMENT_NAME'] <- colnames(predictions[tempColNames])[apply(predictions[tempColNames], 1, which.max)]
      
      testDat <- cbind(testDat, predictions)
      testDat <- estimateUplift(testDat, targetVar, causaVals[1])
      
      factOutputBase <- paste (outputBase, '/', factor ,sep='')
      dir.create(file.path(outputBase, factor), showWarnings = FALSE)
      outFileName <- paste (inputName, '_',testType,'_',timecn , '.csv', sep='') 
      fullPath <- paste(c(factOutputBase,'/', outFileName ), collapse = "")
      write.csv(testDat, fullPath, row.names = FALSE)
    }## End for causes
    
    colnames(globalPredictions) <- globalTempColNames
    globalPredictions['LIFT_SCORE'] <- apply(globalPredictions, 1, max)
    
    globalPredictions['TREATMENT_NAME'] <- colnames(globalPredictions[globalTempColNames])[apply(globalPredictions[globalTempColNames], 1, which.max)]
    testBestDat <- data.frame(orginalTestDat)
    testBestDat <- cbind(testBestDat, globalPredictions)
    testBestDat <- estimateBestUplift(testBestDat, targetVar)
    
    factOutputBase <- paste (outputBase, '/', 'bestFactor' ,sep='')
    dir.create(file.path(outputBase, 'bestFactor'), showWarnings = FALSE)
    outFileName <- paste (inputName, '_',testType,'_',timecn , '.csv', sep='') 
    fullPath <- paste(c(factOutputBase,'/', outFileName ), collapse = "")
    write.csv(testBestDat, fullPath, row.names = FALSE)
    
    ##############
    
  }
}

######
estimateUplift<- function(estimatedImprovements, outComeColName, controlVal){
  data <- data.frame(estimatedImprovements)
  data['UPLIFT'] = 0
  data ['Y_TREATED'] = 0
  data ['N_TREATED'] = 0
  data ['Y_UNTREATED'] = 0
  data ['N_UNTREATED'] = 0
  data ['FOLLOW_REC'] = 0
  separation <- '_vv_'
  
  data <- data[order(-data$LIFT_SCORE),]
  
  y_treated <- 0
  n_treated <- 0
  y_untreated <- 0
  n_untreated <- 0
  for(row in 1: nrow(data)){
    TREATMENT_TEMPVAL = data[row,'TREATMENT_NAME']
    TREATMENT_TEMPVAL <- toString(TREATMENT_TEMPVAL)
    TREATMENT_TEMPVAL <- strsplit(TREATMENT_TEMPVAL, separation)
    TREATMENT_TEMPVAL <- unlist(TREATMENT_TEMPVAL)
    TREATMENT_NAME <- TREATMENT_TEMPVAL[1]
    TREATMENT_VAL <- TREATMENT_TEMPVAL[2]        
    data[row,'N_TREATED']<- n_treated
    data[row,'Y_TREATED']<- y_treated
    data[row,'N_UNTREATED']<- n_untreated
    data[row,'Y_UNTREATED']<- y_untreated
    
    TREATMENT_CRT <- controlVal
    if((TREATMENT_NAME != 'NA')&&(
      (data[row,TREATMENT_NAME] == TREATMENT_VAL)
      
    )
    
    ){
      data [row, 'FOLLOW_REC'] = 1
      n_treated <- n_treated + 1
      data[row,'N_TREATED']<- n_treated
      y_treated <- y_treated + data[row,outComeColName]
      data[row,'Y_TREATED']<- y_treated
    }else{
      n_untreated <- n_untreated + 1
      data[row,'N_UNTREATED']<- n_untreated
      y_untreated <- y_untreated + data[row,outComeColName]
      data[row,'Y_UNTREATED']<- y_untreated
    }
    if(n_treated == 0) {
      data[row,'UPLIFT'] = 0
    }else if(n_untreated == 0){
      data[row,'UPLIFT'] = 0
    }else{
      liftestimate = ((y_treated/n_treated) - (y_untreated/n_untreated) )*(n_treated + n_untreated)
      qiniestimate = ((y_treated) - (y_untreated*(n_treated/n_untreated) ))
      data[row,'UPLIFT'] <- liftestimate
    }
  }
  
  totalIncrease <- ((y_treated/n_treated) - (y_untreated/n_untreated) )
  for(row in 1: nrow(data)){
    n_treated <- data[row,'N_TREATED']
    y_treated <- data[row,'Y_TREATED']
    n_untreated <- data[row,'N_UNTREATED']
    y_untreated <- data[row,'Y_UNTREATED']
    liftestimate <- (((y_treated/n_treated) - (y_untreated/n_untreated) ))
    liftestimateWithBase <- (((y_treated/n_treated) - (y_untreated/n_untreated) ))/totalIncrease
    data[row,'UPLIFT'] <- liftestimate
  }
  return(data)
}

estimateBestUplift<- function(estimatedImprovements, outComeColName){
  data <- data.frame(estimatedImprovements)
  data['UPLIFT'] = 0
  data ['Y_TREATED'] = 0
  data ['N_TREATED'] = 0
  data ['Y_UNTREATED'] = 0
  data ['N_UNTREATED'] = 0
  data ['FOLLOW_REC'] = 0
  separation <- '_vv_'
  
  data <- data[order(-data$LIFT_SCORE),]
  
  y_treated <- 0
  n_treated <- 0
  y_untreated <- 0
  n_untreated <- 0
  for(row in 1: nrow(data)){
    TREATMENT_TEMPVAL = data[row,'TREATMENT_NAME']
    TREATMENT_TEMPVAL <- toString(TREATMENT_TEMPVAL)
    TREATMENT_TEMPVAL <- strsplit(TREATMENT_TEMPVAL, separation)
    TREATMENT_TEMPVAL <- unlist(TREATMENT_TEMPVAL)
    TREATMENT_NAME <- TREATMENT_TEMPVAL[1]
    TREATMENT_VAL <- TREATMENT_TEMPVAL[2]        
    data[row,'N_TREATED']<- n_treated
    data[row,'Y_TREATED']<- y_treated
    data[row,'N_UNTREATED']<- n_untreated
    data[row,'Y_UNTREATED']<- y_untreated
    
    if((TREATMENT_NAME != 'NA')&&(
      (data[row,TREATMENT_NAME] == TREATMENT_VAL)
    )
    ){
      data [row, 'FOLLOW_REC'] = 1
      n_treated <- n_treated + 1
      data[row,'N_TREATED']<- n_treated
      y_treated <- y_treated + data[row,outComeColName]
      data[row,'Y_TREATED']<- y_treated
    }else{
      n_untreated <- n_untreated + 1
      data[row,'N_UNTREATED']<- n_untreated
      y_untreated <- y_untreated + data[row,outComeColName]
      data[row,'Y_UNTREATED']<- y_untreated
    }
    if(n_treated == 0) {
      data[row,'UPLIFT'] = 0
    }else if(n_untreated == 0){
      data[row,'UPLIFT'] = 0
    }else{
      liftestimate = ((y_treated/n_treated) - (y_untreated/n_untreated) )*(n_treated + n_untreated)
      qiniestimate = ((y_treated) - (y_untreated*(n_treated/n_untreated) ))
      data[row,'UPLIFT'] <- liftestimate
    }
  }
  
  totalIncrease <- ((y_treated/n_treated) - (y_untreated/n_untreated) )
  for(row in 1: nrow(data)){
    n_treated <- data[row,'N_TREATED']
    y_treated <- data[row,'Y_TREATED']
    n_untreated <- data[row,'N_UNTREATED']
    y_untreated <- data[row,'Y_UNTREATED']
    liftestimate <- (((y_treated/n_treated) - (y_untreated/n_untreated) ))
    liftestimateWithBase <- (((y_treated/n_treated) - (y_untreated/n_untreated) ))/totalIncrease
    data[row,'UPLIFT'] <- liftestimate
  }
  return(data)
}