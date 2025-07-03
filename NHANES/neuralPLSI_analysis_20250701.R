################################################
###  BMKR code for CU mixtures workshop      ###
###  developed by Katrina Devick             ### 
###  last updated: 5/17/19                  ###
###  Changed POP groups                      ###
################################################

#path
wd <- "/Users/ml5977/Library/CloudStorage/OneDrive-NYULangoneHealth/Research/Fair_joint_single_index_model/data/NHANES1/gibson"
setwd(wd)

## load required libraries 
library(bkmr)
library(corrplot)
library(ggplot2)


################################################
###         Data Manipulation                ###
################################################


## read in data and only consider complete data 
## this drops 327 individuals, but BKMR does not handle missing data
nhanes <- na.omit(read.csv("studypop.csv"))

## center/scale continous covariates and create indicators for categorical covariates
nhanes$age_z         <- scale(nhanes$age_cent)         ## center and scale age
nhanes$agez_sq       <- nhanes$age_z^2                 ## square this age variable
nhanes$bmicat2       <- as.numeric(nhanes$bmi_cat3==2) ## 25 <= BMI < 30
nhanes$bmicat3       <- as.numeric(nhanes$bmi_cat3==3) ## BMI >= 30 (BMI < 25 is the reference)
nhanes$educat1       <- as.numeric(nhanes$edu_cat==1)  ## no high school diploma
nhanes$educat3       <- as.numeric(nhanes$edu_cat==3)  ## some college or AA degree
nhanes$educat4       <- as.numeric(nhanes$edu_cat==4)  ## college grad or above (reference is high schol grad/GED or equivalent)
nhanes$otherhispanic <- as.numeric(nhanes$race_cat==1) ## other Hispanic or other race - including multi-racial
nhanes$mexamerican   <- as.numeric(nhanes$race_cat==2) ## Mexican American 
nhanes$black         <- as.numeric(nhanes$race_cat==3) ## non-Hispanic Black (non-Hispanic White as reference group)
nhanes$wbcc_z        <- scale(nhanes$LBXWBCSI)
nhanes$lymphocytes_z <- scale(nhanes$LBXLYPCT)
nhanes$monocytes_z   <- scale(nhanes$LBXMOPCT)
nhanes$neutrophils_z <- scale(nhanes$LBXNEPCT)
nhanes$eosinophils_z <- scale(nhanes$LBXEOPCT)
nhanes$basophils_z   <- scale(nhanes$LBXBAPCT)
nhanes$lncotinine_z  <- scale(nhanes$ln_lbxcot)         ## to access smoking status, scaled ln cotinine levels


## our y variable - ln transformed and scaled mean telomere length
lnLTL_z <- scale(log(nhanes$TELOMEAN)) 
nhanes$lnLTL_z <- lnLTL_z

## our Z matrix
mixture <- with(nhanes, cbind(LBX074LA, LBX099LA, LBX118LA, LBX138LA, LBX153LA, LBX170LA, LBX180LA, LBX187LA, 
                              LBX194LA, LBXHXCLA, LBXPCBLA,
                              LBXD03LA, LBXD05LA, LBXD07LA,
                              LBXF03LA, LBXF04LA, LBXF05LA, LBXF08LA)) 
lnmixture   <- apply(mixture, 2, log)
lnmixture_z <- scale(lnmixture)
colnames(lnmixture_z) <- c(paste0("PCB",c(74, 99, 118, 138, 153, 170, 180, 187, 194, 169, 126)), 
                           paste0("Dioxin",1:3), paste0("Furan",1:4)) 

## our X matrix
covariates <- with(nhanes, cbind(age_z, agez_sq, male, bmicat2, bmicat3, educat1, educat3, educat4, 
                                 otherhispanic, mexamerican, black, wbcc_z, lymphocytes_z, monocytes_z, 
                                 neutrophils_z, eosinophils_z, basophils_z, lncotinine_z)) 
colnames(covariates) <- c("age_z", "agez_sq", "male", "bmicat2", "bmicat3", "educat1", "educat3", "educat4", 
                          "otherhispanic", "mexamerican", "black", "wbcc_z", "lymphocytes_z", "monocytes_z", 
                          "neutrophils_z", "eosinophils_z", "basophils_z", "lncotinine_z")

#based on previous stuides with variable importance
#select.z <- lnmixture_z[,c("Furan1","PCB169","PCB126",
#                           "PCB74","PCB153","PCB170","PCB180","PCB194")]
select.z <- lnmixture_z[,c("Furan1","PCB169","PCB126",
                           "PCB74","PCB153","PCB170","PCB180","PCB187")]
                           
##PLSI
source("PLSI_linear_functions.R")
library(mgcv);library(dplyr)
data <- data.frame(cbind(lnLTL_z, select.z, covariates)); colnames(data)[1] <-"lnLTL_z"
select.mix.name <- colnames(select.z)
covariate.name <- c("age_z", "agez_sq", "male", "bmicat2", "bmicat3", "educat1", "educat3", "educat4", 
                    "otherhispanic", "mexamerican", "black", "wbcc_z", "lymphocytes_z", "monocytes_z", 
                    "neutrophils_z", "eosinophils_z", "basophils_z", "lncotinine_z")
formula_continuous <- as.formula(paste('lnLTL_z~',paste(c(select.mix.name, covariate.name),collapse="+"),sep=""))
formula_continuous

linear.base <- lm(formula_continuous, data=data)
th0.base <- initial.beta(outcome="mean", linear.base, exposure=select.mix.name) #get initial beta from linear regression
plsi.fit <- plsi(outcome="mean",fun=si,th0=th0.base,
                 y=data$lnLTL_z,x=select.z,z=covariates,family="gaussian")
plsi.fit.coef <- plsi_output(outcome="mean",plsi.fit,exposure=select.mix.name, covariate=covariate.name)
plsi.boot <- bootstrap(outcome="mean",y_name="lnLTL_z",exposure=select.mix.name,weights=NULL,
                       covariate=covariate.name, fun=si,B=1000,formula=formula_continuous,
                       family="gaussian",b=plsi.fit,othervalue=0,data=data)
plsi_plot(outcome="mean",b=plsi.fit,boot=plsi.boot,
          exposure=select.mix.name,covariate=covariate.name,z=covariates)
plsi.fit.coef[c(1:length(select.mix.name)),]
round(apply(plsi.boot$par_est, 2, sd),3)
round(2*(1-pnorm(abs(plsi.fit.coef[c(1:length(select.mix.name)),2]/apply(plsi.boot$par_est, 2, sd)))),3)


###BKMR
## create knots matrix for Gaussian predictive process (to speed up BKMR with large datasets)
set.seed(10)
knots100     <- fields::cover.design(select.z, nd = 100)$design

set.seed(1000)
bkmr.fit <-  kmbayes(y=lnLTL_z, Z=select.z, X=covariates, iter=10000, verbose=TRUE, varsel=TRUE, knots=knots100)
summary(bkmr.fit)

pred.resp.univar <- PredictorResponseUnivar(fit = bkmr.fit)
ggplot(pred.resp.univar, aes(z, est, ymin = est - 1.96*se, ymax = est + 1.96*se)) + 
  geom_smooth(stat = "identity") + 
  facet_wrap(~ variable) +
  ylab("h(z)")

risks.overall <- OverallRiskSummaries(fit = bkmr.fit, y = lnLTL_z, Z = select.z, X = covariates, 
                                      qs = seq(0.25, 0.75, by = 0.05), 
                                      q.fixed = 0.5, method = "exact")
risks.overall
ggplot(risks.overall, aes(quantile, est, ymin = est - 1.96*sd, ymax = est + 1.96*sd)) + 
  geom_pointrange()


analysis.fit <- list(linear.base=linear.base, th0.base=th0.base, plsi.fit=plsi.fit,
                     plsi.fit.coef=plsi.fit.coef, plsi.boot=plsi.boot, 
                     bkmr.fit=bkmr.fit, knots100=knots100,
                     
                     nhanes=nhanes,
                     data=data, formula_continuous=formula_continuous, lnLTL_z=lnLTL_z, covariates=covariates, select.z=select.z,
                     covariate.name=covariate.name, select.mix.name=select.mix.name)


#save(analysis.fit, file="nhanes_analysis_pre_20250701.RData")

