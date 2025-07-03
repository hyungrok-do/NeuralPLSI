
# get inital value from parametric function
initial.beta <- function(outcome=NULL,par,exposure){
  th0 <- par$coefficients[exposure]/par$coefficients[exposure][1];th0 <- th0[-1]
  return(th0)
}
# PLSI model function, a function used to obtain single index estimates by nlm:
# required arguments:
# 1. outcome: specify type of outcome (mean (normal response), logistic, quantile, survival)
# 2. fun: objective function to minimize to get estimation of unknown single index parameters
# 3. th0: initial estimates of single index parameters, with length p-1, where p is the number of single index parameters. Note that the first estimate is omitted here due to unit length contrain.
# 4. x: a matrix of exposure variables
# 5. z: a matrix of confounders. Note that factors need to be converted into several dummy variables
# optional arguments:
# 1. y is required if outcome is 'mean', 'logistic', 'quantile', 'longitudinal'
# 2. t (survial response) is required if outcome is 'survival'
# 3. d is required if outcome is 'survival'
# 4. weights: a vector of weights
# 5. family is required if outcome is 'mean', 'logistic' or 'survival'
# 6. tau: specifying percentiles of interest. Need to be specified if quantile outcome is used.
plsi <- function(outcome,fun,th0,y=NULL,t=NULL,d=NULL,x,z,weights=NULL,family=NULL,tau=NULL){
  if (outcome=="mean" | outcome=="logistic"){
    # get updated beta using no penalization
    f0 <- nlm(fun,th0,y=y,x=x,z=z,weights=weights,family=family,fx=TRUE,k=5);print("f0 finished") 
    # get final beta with smoothing parameter selection
    f1 <- nlm(fun,f0$estimate,y=y,x=x,z=z,weights=weights,family=family,k=10);print("f1 finished") 
    # get link function and gamma
    b <- fun(f1$estimate,y=y,x=x,z=z,weights=weights,family=family,opt=FALSE) 
  } else if(outcome=="quantile"){
    f0 <- nlm(fun,th0,y=y,x=x,z=z,weights=weights,tau=tau,fx=TRUE,k=5);print("f0 finished") 
    f1 <- nlm(fun,f0$estimate,y=y,x=x,z=z,weights=weights,tau=tau,k=10);print("f1 finished") 
    b <- fun(f1$estimate,y=y,x=x,z=z,weights=weights,tau=tau,opt=FALSE)
  } else if(outcome=="survival"){
    f0 <- nlm(fun,th0,t=t,d=d,x=x,z=z,fx=TRUE,k=5);print("f0 finished")
    f1 <- nlm(fun,f0$estimate,t=t,d=d,x=x,z=z,k=10);print("f1 finished")
    b <- fun(f1$estimate,t=t,d=d,x=x,z=z,opt=FALSE)
  }
  return(b)
}
# PLSI: a function to report parameter estimates from partial-linear single-index models
# required arguments:
# 1. outcome: specify type of outcome (mean (normal response), logistic, quantile, survival)
# 2. b: PLSI model object
# 3. exposure: a vector of exposure variables to form single index
# 4. covariate: a character vector of covariates (partially linear terms)
# optional arguments:
# 1. out22 is required in sensitivity analysis for 22 exposures
plsi_output <- function(outcome,b,out22=NULL,exposure,covariate){
  if (outcome=="longitudinal"){
    results <- data.frame(Name=c(exposure,c("Intercept",covariate)),Estimates=c(b$beta.f,b$model$gam$coefficients[c('(Intercept)',covariate)]))
  } else if (outcome=="survival"){
    results <- data.frame(Name=c(exposure,covariate),Estimates=c(b$beta,b$coefficients[covariate]))
  } else {
    results <- data.frame(Name=c(exposure,c("Intercept",covariate)),Estimates=c(b$beta,b$coefficients[c('(Intercept)',covariate)]))
  }
  return(results)
}
# new dataset for prediction to plot PLSI model link function 
# required arguments:
# 1. outcome: specify type of outcome (mean (normal response), logistic, quantile, survival)
# 2. b: PLSI model object
# 3. othervalue: specify values of covariates
# 4. covariate: a character vector of covariates (partially linear terms)
sindex_plot_preddata <- function(outcome,b,othervalue,covariate){
  if (outcome=="longitudinal"){
    sindex_pred <- as.data.frame(cbind(seq(min(b$model$gam$model$u),max(b$model$gam$model$u),length.out=200),matrix(othervalue,200,length(covariate))))
    colnames(sindex_pred) <- c("u",covariate)
  } else{
    sindex_pred <- as.data.frame(cbind(seq(min(b$model$a),max(b$model$a),length.out=200),matrix(othervalue,200,length(covariate))))
    colnames(sindex_pred) <- c("a",covariate)
  }
  return(sindex_pred)
}
# predicted value used to plot PLSI model link function (default is 200 single index points)
# required arguments:
# 1. b: PLSI model object
# 2. outcome: specify type of outcome (mean (normal response), logistic, quantile, survival)
# 3. covariate: a character vector of covariates (partially linear terms)
pred_sindex <- function(outcome=NULL,b,covariate){
  if (outcome=="longitudinal"){
    index <- as.data.frame(cbind(seq(min(b$model$gam$model$u),max(b$model$gam$model$u),length.out=200),matrix(100,200,length(covariate))))
    colnames(index) <- c("u",covariate)
    gindex <- predict(b$model$gam,index,type="terms",exclude=covariate)
  } else {
    index <- as.data.frame(cbind(seq(min(b$model$a),max(b$model$a),length.out=200),matrix(0,200,length(covariate))))
    colnames(index) <- c("a",covariate)
    gindex <- predict(b,index,type="terms",exclude=covariate)
  }
  return(gindex)
}
pred_sindex2 <- function(outcome=NULL,b,covariate){
  if (outcome=="longitudinal"){
    index <- as.data.frame(cbind(seq(min(b$model$gam$model$u),max(b$model$gam$model$u),length.out=200),matrix(100,200,length(covariate))))
    colnames(index) <- c("u",covariate)
    gindex <- predict(b$model$gam,index,type="terms",exclude=covariate)
  } else {
    index <- as.data.frame(cbind(seq(min(b$model$a),max(b$model$a),length.out=200),matrix(0,200,length(covariate))))
    colnames(index) <- c("a",covariate)
    gindex <- predict(b,index,type="terms",exclude=covariate)
  }
  
  return(data.frame(index=index, gindex=gindex))
}
# bootstrap for CIs
# required arguments:
# 1. outcome: specify type of outcome (mean (normal response), logistic, quantile, survival)
# 2. fun: objective function to minimize to get estiamates of single index parameters
# 3. B: the number of bootstrap to perform
# 4. exposure: a character vector of exposures
# 5. covariate: a character vector of covariates
# 6. formula: formulas for parametric models
# 7. othervalue: specify values of covariates for prediction
# 8. data: original dataset
# 9. b: PLSI model object
# optional arguments:
# 1. y_name is required if outcome is "logistic"
# 2. tau is required if outcome is "quantile", specifying percentiles of interest
# 3. family is required if outcome is 'mean', 'logistic' or 'survival'
# 4. weights: a vector of weights
# 5. random is required if outcome is 'longitudinal': random component, refer to 'gamm' function
# 6. correlation is required if outcome is 'longitudinal': specify correlation matrix, refer to 'lme' function
bootstrap <- function(outcome,fun,B,exposure,covariate,formula,othervalue,data,b,y_name=NULL,tau=NULL,family=NULL,weights=NULL,random=NULL,correlation=NULL){
  P <- length((as.character(formula)[[3]] %>% strsplit(.,split='+',fixed=TRUE))[[1]])+1 # total number of unknown parameters
  if ((length(exposure)+length(covariate)+1)!=P) {stop('exposure and covariate do not agree with formula')}
  
  plsi_est <- matrix(NA,B,P)
  par_quantile_est <- matrix(NA,B,P)
  par_est <- matrix(NA,B,length(exposure))
  score_est_se <- matrix(NA,B,200)
  sindex_pred <- sindex_plot_preddata(outcome,b,othervalue,covariate)
  gindex <- as.vector(pred_sindex(outcome,b,covariate))
  for (i in 1:B){
    set.seed(i)
    boot.sample <- data[sample(nrow(data),nrow(data),replace=TRUE),]
    if (outcome=="mean" | outcome=="logistic"){
      par_boot <- glm(formula=formula,family=family,weights=NULL,data=boot.sample)
      par_est[i,] <- par_boot$coefficients[exposure]/sqrt(sum(par_boot$coefficients[exposure]^2))
      th0_boot <- par_boot$coefficients[exposure]/(par_boot$coefficients[exposure][1]);th0_boot <- th0_boot[-1]
      y_boot <- boot.sample[,y_name];x_boot <- as.matrix(boot.sample[,exposure]);z_boot <- as.matrix(boot.sample[,covariate])
      f0_boot <- nlm(fun,th0_boot,y=y_boot,x=x_boot,z=z_boot,weights=NULL,family=family,fx=TRUE,k=5);print("f0 finished")
      f1_boot <- nlm(fun,f0_boot$estimate,y=y_boot,x=x_boot,z=z_boot,weights=NULL,family=family,k=10);print("f1 finished")
      b_boot <- fun(f1_boot$estimate,y=y_boot,x=x_boot,z=z_boot,weights=NULL,family=family,opt=FALSE)
      plsi_est[i,1:length(exposure)] <- b_boot$beta;plsi_est[i,(length(exposure)+1):P] <- b_boot$coefficients[c('(Intercept)',covariate)]
      score_est_se[i,] <- predict(b_boot,sindex_pred,type="terms",exclude=covariate)
    } else if (outcome=="quantile"){
      par_boot <- rq(formula=formula,weights=weights,tau=0.25,data=boot.sample)
      par_quantile_est[i,] <- c(par_boot$coefficients[exposure],par_boot$coefficients[c('(Intercept)',covariate)])
      par_est[i,] <- par_boot$coefficients[exposure]/sqrt(sum(par_boot$coefficients[exposure]^2))
      th0_boot <- par_boot$coefficients[exposure]/(par_boot$coefficients[exposure][1]);th0_boot <- th0_boot[-1]
      y_boot <- boot.sample[,y_name];x_boot <- as.matrix(boot.sample[,exposure]);z_boot <- as.matrix(boot.sample[,covariate])
      f0_boot <- nlm(fun,th0_boot,y=y_boot,x=x_boot,z=z_boot,weights=weights,tau=tau,fx=TRUE,k=5);print("f0 finished")
      f1_boot <- nlm(fun,f0_boot$estimate,y=y_boot,x=x_boot,z=z_boot,weights=weights,tau=tau,k=10);print("f1 finished")
      b_boot <- fun(f1_boot$estimate,y=y_boot,x=x_boot,z=z_boot,weights=weights,tau=tau,opt=FALSE)
      plsi_est[i,1:length(exposure)] <- b_boot$beta;plsi_est[i,(length(exposure)+1):ncol(plsi_est)] <- b_boot$coefficients[c('(Intercept)',covariate)]
      score_est_se[i,] <- predict(b_boot,sindex_pred,type="terms",exclude=covariate)
    } else if (outcome=="survival"){
      par_boot <- coxph(formula=formula,data=boot.sample)
      par_est[i,] <- par_boot$coefficients[exposure]/sqrt(sum(par_boot$coefficients[exposure]^2))
      th0_boot <- par_boot$coefficients[exposure]/(par_boot$coefficients[exposure][1]);th0_boot <- th0_boot[-1]
      t_boot <- boot.sample$time;d_boot <- boot.sample$status;x_boot <- as.matrix(boot.sample[,exposure]);z_boot <- as.matrix(boot.sample[,covariate])
      f0_boot <- nlm(fun,th0_boot,t=t_boot,d=d_boot,x=x_boot,z=z_boot,fx=TRUE,k=5);print("f0 finished")
      f1_boot <- nlm(fun,f0_boot$estimate,t=t_boot,d=d_boot,x=x_boot,z=z_boot,k=10);print("f1 finished")
      b_boot <- fun(f1_boot$estimate,t=t_boot,d=d_boot,x=x_boot,z=z_boot,opt=FALSE)
      plsi_est[i,1:length(exposure)] <- b_boot$beta;plsi_est[i,(length(exposure)+1):ncol(plsi_est)] <- c(NA,b_boot$coefficients[covariate])
      score_est_se[i,] <- predict(b_boot,sindex_pred,type="terms",exclude=covariate)
    } else if (outcome=="longitudinal"){
      par_boot <- lme(formula,random=random,correlation=correlation,data=boot.sample)
      par_est[i,] <- par_boot$coefficients$fixed[exposure]/sqrt(sum(par_boot$coefficients$fixed[exposure]^2))
      b_boot <- try(lsim(x=exposure,z=covariate,y='yij',id='id',random=random,correlation=correlation,data=boot.sample))
      if(isTRUE(class(b_boot)=="try-error")){
        plsi_est[i,1:length(exposure)] <- rep(NA,length(exposure));plsi_est[i,(length(exposure)+1):ncol(plsi_est)] <- rep(NA,(length(covariate)+1));score_est_se[i,] <- rep(NA,200)
      }else{
        plsi_est[i,1:length(exposure)] <- b_boot$beta.f
        plsi_est[i,(length(exposure)+1):ncol(plsi_est)] <- b_boot$model$gam$coefficients[c('(Intercept)',covariate)]
        colnames(sindex_pred) <- c("u",covariate)
        score_est_se[i,] <- predict(b_boot$model$gam,sindex_pred,type="terms",exclude=covariate)
      }
    }
    print(i)
  }
  for (i in 1:B){
    if (score_est_se[i,200] > gindex[200]+5*sd(score_est_se[,200],na.rm=TRUE)) {score_est_se[i,] <- rep(NA,200)}
  }
  if (outcome=="quantile"){
    return(list(plsi_est=plsi_est,par_quantile_est=par_quantile_est,par_est=par_est,score_est_se=score_est_se,index=sindex_pred[,1]))
  } else {
    return(list(plsi_est=plsi_est,par_est=par_est,score_est_se=score_est_se,index=sindex_pred[,1]))
  }
}
# PLSI link function plot
# required arguments:
# 1. outcome: specify type of outcome (mean (normal response), logistic, quantile, survival)
# 2. b: PLSI model object
# 3. boot: bootstrap object
# 4. covariate: a character vector of covariates
# optional arguments:
# 1. z is required if outcome is "mean"
# 2. quad is required if simulated link function is quadratic
plsi_plot <- function(outcome,b,boot,covariate,exposure,z=NULL,quad=NULL){
  if (outcome=="longitudinal"){
    linkplot <- data.frame(index=boot$index,gindex=as.vector(pred_sindex(outcome="longitudinal",b,covariate))+b$model$gam$coefficients['(Intercept)'],lower=apply(boot$score_est_se+boot$plsi_est[,length(exposure)+1],2,quantile,probs=c(0.025),na.rm=TRUE),upper=apply(boot$score_est_se+boot$plsi_est[,length(exposure)+1],2,quantile,probs=c(0.975),na.rm=TRUE),se=apply(boot$score_est_se+boot$plsi_est[,length(exposure)+1],2,sd,na.rm=TRUE))
    plot(linkplot$index,linkplot$gindex,type="l",lwd=1,col=2,xlab="Index",ylab="g(index)",xaxt='n')
    axis(side=1,at=b$model$gam$model$u,labels=FALSE)
    if (is.null(quad)){
      x0=c(-3,0,5);y0=c(-3,0,5);lines(x0,y0)
    } else {
      x0=runif(300,-3,5);y0=x0^2;smoothingSpline=smooth.spline(x0,y0,spar=0.01);lines(smoothingSpline)
    }
  } else if (outcome=="survival"){
    linkplot <- data.frame(index=boot$index,gindex=as.vector(pred_sindex(outcome,b,covariate)),lower=apply(boot$score_est_se,2,quantile,probs=c(0.025),na.rm=TRUE),upper=apply(boot$score_est_se,2,quantile,probs=c(0.975),na.rm=TRUE),se=apply(boot$score_est_se,2,sd,na.rm=TRUE))
    linkplot <- data.frame(rbind(linkplot,c(0,0,0,0,0)));linkplot <- linkplot[order(linkplot$index),]
    plot(linkplot$index,linkplot$gindex,type="l",lwd=1,col=2,xlab="Index",ylab="g(index)",xaxt='n')
    axis(side=1,at=b$model$a,labels=FALSE)
    abline(h=0,lty=3);abline(v=0,lty=3)
    if (is.null(quad)){
      x0=c(-3,0,5);y0=c(-3,0,5);lines(x0,y0)
    } else {
      x0=runif(300,-3,5);y0=x0^2;smoothingSpline=smooth.spline(x0,y0,spar=0.01);lines(smoothingSpline)
    }
  } else {
    linkplot <- data.frame(index=boot$index,gindex=as.vector(pred_sindex(outcome,b,covariate))+b$coefficients['(Intercept)'],lower=apply(boot$score_est_se+boot$plsi_est[,length(exposure)+1],2,quantile,probs=c(0.025),na.rm=TRUE),upper=apply(boot$score_est_se+boot$plsi_est[,length(exposure)+1],2,quantile,probs=c(0.975),na.rm=TRUE),se=apply(boot$score_est_se+boot$plsi_est[,length(exposure)+1],2,sd,na.rm=TRUE))
    if (outcome=="mean"){
      plot(b$model$a,(as.matrix(b$y)-z%*%as.matrix(b$coefficients[covariate])),xlab="Index",ylab="g(index)") # plot the observed values
      lines(linkplot$index,linkplot$gindex,lwd=1,col=2) # plot predicted link function
    } else if (outcome=="quantile"){
      plot(linkplot$index,linkplot$gindex,type="l",lwd=1,col=2,xlab="Index",ylab="g(index)",xlim=c(-2,4),ylim=c(-2,3),xaxt='n')
      axis(side=1,at=b$model$a,labels=FALSE)
    } else if (outcome=="logistic"){
      plot(linkplot$index,linkplot$gindex,type="l",lwd=1,col=2,xlab="Index",ylab="g(index)",xaxt='n')
      axis(side=1,at=b$model$a,labels=FALSE)
    }
  }
  lines(linkplot$index,linkplot$gindex+qnorm(0.025)*linkplot$se,lwd=1,lty=2,col=2) # lower bound of 95% CI
  lines(linkplot$index,linkplot$gindex+qnorm(0.975)*linkplot$se,lwd=1,lty=2,col=2) # upper bound of 95% CI
  # lines(linkplot$index,linkplot$lower,lwd=1,lty=2,col=2) # lower bound of 95% CI
  # lines(linkplot$index,linkplot$upper,lwd=1,lty=2,col=2) # upper bound of 95% CI
}

si <- function(theta,y,x,z,weights,family,opt=TRUE,k=10,fx=FALSE) {
  beta <- c(1,theta)
  kk <- sqrt(sum(beta^2))
  beta <- beta/kk # normed coefficients
  a <- x%*%beta # calculate single index
  if (is.null(z)) { # get the fomula and dataset for gam call
    dat <- as.data.frame(cbind(y,a));colnames(dat) <- c("y","a")
    formula.t <- as.formula(paste('y~s(a,fx=',fx,',k=',k,')'))
  } else {
    dat <- as.data.frame(cbind(y,a,z));colnames(dat) <- c("y","a",colnames(z))
    formula.t <- as.formula(paste('y~s(a,fx=',fx,',k=',k,')+',paste(colnames(z),collapse = '+')))
  }
  b <- gam(formula.t,data=dat,family=family,weights=weights,method="ML")
  if (opt) return(b$gcv.ubre) else {
    b$beta <- beta
    return(b)
  }
}