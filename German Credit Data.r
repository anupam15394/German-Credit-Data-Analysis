####################################################################################################################
> A1572 <- read_excel("UIC/Fall-2018/Study Material/IDS 572/Assignments/A1572.xls") # Import Excel
> View(A1572) # View Excel
> A1572[is.na(A1572)] <- 0
> A1572$AGE[A1572$AGE ==0] <- NA
> A1572$AGE[is.na(A1572$AGE)] <- 35.5
>rpModel1=rpart(RESPONSE ~ ., data=A1572, method="class")
> library(rpart.plot)
> rpart.plot::prp(rpModel1, type=2, extra=1)
> rpart.plot::prp(rpModel1, type=2, extra=1, main = 'Decision Tree for German Credit Data')
> plot(rpModel1, uniform=TRUE,  main="Decision Tree for German Credit Data")
> text(rpModel1, use.n=TRUE, all=TRUE, cex=.7)
> rpModel2=rpart(RESPONSE ~ ., data=A1572, method="class")
> predTrn=predict(rpModel2, mdTrn, type='class')
> table(pred = predTrn, true=A1572$RESPONSE)
    true
pred   0   1
   0 170  72
   1 130 628
> mean(predTrn==A1572$RESPONSE)
[1] 0.798
> table(pred=predict(rpModel2,A1572, type="class"), true=A1572$RESPONSE)
    true
pred   0   1
   0 170  72
   1 130 628

R version 3.5.1 (2018-07-02) -- "Feather Spray"
Copyright (C) 2018 The R Foundation for Statistical Computing
Platform: x86_64-w64-mingw32/x64 (64-bit)

R is free software and comes with ABSOLUTELY NO WARRANTY.
You are welcome to redistribute it under certain conditions.
Type 'license()' or 'licence()' for distribution details.

R is a collaborative project with many contributors.
Type 'contributors()' for more information and
'citation()' on how to cite R or R packages in publications.

Type 'demo()' for some demos, 'help()' for on-line help, or
'help.start()' for an HTML browser interface to help.
Type 'q()' to quit R.

> library(readxl)
> A1572 <- read_excel("UIC/Fall-2018/Study Material/IDS 572/Assignments/A1572.xls")
> View(A1572)
> A1572$AGE[is.na(A1572$AGE)] <- 35.5
> A1572$AGE[is.na(A1572$AGE)] <- 35
> A1572$AGE[(A1572$AGE)==35.5] <- 35
> A1572[is.na(A1572)] <- 0
> install.packages("rpart")
Installing package into ‘C:/Users/family/Documents/R/win-library/3.5’
(as ‘lib’ is unspecified)
trying URL 'https://cran.rstudio.com/bin/windows/contrib/3.5/rpart_4.1-13.zip'
Content type 'application/zip' length 950625 bytes (928 KB)
downloaded 928 KB

package ‘rpart’ successfully unpacked and MD5 sums checked

The downloaded binary packages are in
	C:\Users\family\AppData\Local\Temp\RtmpELA0uz\downloaded_packages
> library('rpart')
> install.packages("rpart.plot")
Installing package into ‘C:/Users/family/Documents/R/win-library/3.5’
(as ‘lib’ is unspecified)
trying URL 'https://cran.rstudio.com/bin/windows/contrib/3.5/rpart.plot_3.0.4.zip'
Content type 'application/zip' length 1058964 bytes (1.0 MB)
downloaded 1.0 MB

package ‘rpart.plot’ successfully unpacked and MD5 sums checked

The downloaded binary packages are in
	C:\Users\family\AppData\Local\Temp\RtmpELA0uz\downloaded_packages
> rpModel1=rpart(RESPONSE ~ ., data=A1572, method="class")
> library(rpart.plot)
> rpart.plot::prp(rpModel1, type=2, extra=1)
> rpart.plot::prp(rpModel1, type=2, extra=1, main = 'Decision Tree for German Credit Data')
> rpModel2=rpart(RESPONSE ~ ., data=A1572, method="class")
> predTrn=predict(rpModel2, A1572, type='class')
> table(pred = predTrn, true=A1572$RESPONSE)
    true
pred   0   1
   0 170  72
   1 130 628
> mean(predTrn==A1572$RESPONSE)
[1] 0.798
> table(pred=predict(rpModel2,A1572, type="class"), true=A1572$RESPONSE)
    true
pred   0   1
   0 170  72
   1 130 628
> predTrnProb=predict(rpModel2, A1572, type='prob')
> head(predTrnProb)
           0         1
1 0.05405405 0.9459459
2 0.86111111 0.1388889
3 0.13129103 0.8687090
4 0.62043796 0.3795620
5 0.62043796 0.3795620
6 0.13129103 0.8687090
> trnSc <- subset(A1572, select=c("RESPONSE")) 
> trnSc$score<-predTrnProb[, 1]
> trnSc<-trnSc[order(trnSc$score, decreasing=TRUE),]
> str(trnSc)
Classes ‘tbl_df’, ‘tbl’ and 'data.frame':	1000 obs. of  2 variables:
 $ RESPONSE: num  0 0 0 0 0 0 1 0 0 0 ...
 $ score   : num  0.861 0.861 0.861 0.861 0.861 ...
> trnSc$OUTCOME<-as.numeric(as.character(trnSc$RESPONSE))
> 
> str(trnSc)
Classes ‘tbl_df’, ‘tbl’ and 'data.frame':	1000 obs. of  3 variables:
 $ RESPONSE: num  0 0 0 0 0 0 1 0 0 0 ...
 $ score   : num  0.861 0.861 0.861 0.861 0.861 ...
 $ OUTCOME : num  0 0 0 0 0 0 1 0 0 0 ...
> trnSc$cumDefault<-cumsum(trnSc$RESPONSE)
> head(trnSc)
# A tibble: 6 x 4
  RESPONSE score OUTCOME cumDefault
     <dbl> <dbl>   <dbl>      <dbl>
1        0 0.861       0          0
2        0 0.861       0          0
3        0 0.861       0          0
4        0 0.861       0          0
5        0 0.861       0          0
6        0 0.861       0          0
> plot(seq(nrow(trnSc)), trnSc$cumDefault,type = "l", xlab='#cases', ylab='#default')
> plot(seq(nrow(trnSc)), trnSc$cumDefault,type = "l", xlab='False Positive Rate', ylab='True Positive Rate')
> predTrnProb=predict(rpModel2, A1572 , type='prob')
> head(predTrnProb)
           0         1
1 0.05405405 0.9459459
2 0.86111111 0.1388889
3 0.13129103 0.8687090
4 0.62043796 0.3795620
5 0.62043796 0.3795620
6 0.13129103 0.8687090
> trnSc <- subset(A1572, select=c("RESPONSE"))
> trnSc$score<-predTrnProb[, 1]
> trnSc<-trnSc[order(trnSc$score, decreasing=TRUE),]
> str(trnSc)
Classes ‘tbl_df’, ‘tbl’ and 'data.frame':	1000 obs. of  2 variables:
 $ RESPONSE: num  0 0 0 0 0 0 1 0 0 0 ...
 $ score   : num  0.861 0.861 0.861 0.861 0.861 ...
> levels(trnSc$RESPONSE)[1]<-1
> levels(trnSc$RESPONSE)[2]<-0
> trnSc$RESPONSE<-as.numeric(as.character(trnSc$RESPONSE))
> str(trnSc)
Classes ‘tbl_df’, ‘tbl’ and 'data.frame':	1000 obs. of  2 variables:
 $ RESPONSE: num  0 0 0 0 0 0 1 0 0 0 ...
 $ score   : num  0.861 0.861 0.861 0.861 0.861 ...
> trnSc$cumDefault<-cumsum(trnSc$RESPONSE)
> head(trnSc)
# A tibble: 6 x 3
  RESPONSE score cumDefault
     <dbl> <dbl>      <dbl>
1        0 0.861          0
2        0 0.861          0
3        0 0.861          0
4        0 0.861          0
5        0 0.861          0
6        0 0.861          0
> plot(seq(nrow(trnSc)), trnSc$cumDefault,type = "l", xlab='#cases', ylab='#default')
> install.packages(dplyr)
Error in install.packages : object 'dplyr' not found
> install.packages("lift")
> library('lift')
> plotLift(trnSc$score, trnSc$RESPONSE)
################################################################################################################################

install.packages("C50")
install.packages("rpart.plot")
install.packages("ROCR")
install.packages("lift")
library('rpart')
library('readxl')
install.packages("caret")
library(caret)
install.packages("e1071")
library(e1071)
library(C50)
cols <- c("RESPONSE", "FOREIGN", "TELEPHONE", "OWN_RES","NEW_CAR", "USED_CAR", "FURNITURE","RADIO/TV","EDUCATION","RETRAINING", "MALE_DIV", "MALE_SINGLE","MALE_MAR_or_WID", "CO-APPLICANT", "GUARANTOR","REAL_ESTATE", "PROP_UNKN_NONE", "OTHER_INSTALL", "RENT")
mdata[cols] <- lapply(mdata[cols], factor)
sapply(mdata, class)
mdata$X <- NULL
str(mdata)

# Developing the decision tree
nr=nrow(mdata)
trnIndex = sample(1:nr, size = round(0.5*nr), replace=FALSE)
mdTrn=mdata[trnIndex,]
mdTst = mdata[-trnIndex,]

dim(mdTrn) 
dim(mdTst)

rpmodel1 <- rpart(RESPONSE ~ ., data = mdTrn, method = "class")
preTrn <- predict(rpmodel1, mdTrn, type = "class")
table(pred = preTrn, true = mdTrn$RESPONSE)
confusionMatrix(preTrn,mdTrn$RESPONSE)
mean(preTrn==mdTrn$RESPONSE)
printcp(rpmodel1)
plotcp(rpmodel1)
summary(rpmodel1)
cm <- table(pred=predict(rpmodel1,mdTrn, type="class"), true=mdTrn$RESPONSE)
n = sum(cm) # number of instances
diag = diag(cm) # number of correctly classified instances per class 
rowsums = apply(cm, 2, sum) # number of instances per class
colsums = apply(cm, 1, sum) # number of predictions per class
p = rowsums / n # distribution of instances over the actual classes
q = colsums / n # distribution of instances over the predicted classes
accuracy = sum(diag) / n 
accuracy
precision = diag / colsums 
precision
recall = diag / rowsums
recall
f1 = 2 * precision * recall / (precision + recall) 
f1

library(ROCR)
Tsscore1 <-predict(rpmodel1,mdTst)
ROCR1_pred <- prediction(Tsscore1[,2],mdTst$RESPONSE)
ROCR1_pref <- performance(ROCR1_pred,"tpr","fpr")
plot(ROCR1_pref, main = "ROC Curve")

auc.perf = performance(ROCR1_pred, measure = "auc")
auc.perf@y.values

rpmodel2 <- rpart(RESPONSE ~ ., data = mdTst, method = "class")
preTst <- predict(rpmodel1, mdTrn, type = "class")
table(pred = preTst, true = mdTst$RESPONSE)
confusionMatrix(preTst,mdTst$RESPONSE)
mean(preTst==mdTst$RESPONSE)
printcp(rpmodel2)
plotcp(rpmodel2)
summary(rpmodel2)
cm2 <- table(pred=predict(rpmodel2,mdTst, type="class"), true=mdTst$RESPONSE)
n2 = sum(cm2) # number of instances
diag = diag(cm2) # number of correctly classified instances per class 
rowsums2 = apply(cm2, 2, sum) # number of instances per class
colsums2 = apply(cm2, 1, sum) # number of predictions per class
p2 = rowsums2 / n # distribution of instances over the actual classes
q2 = colsums2 / n # distribution of instances over the predicted classes
accuracy2 = sum(diag) / n
accuracy2
precision2 = diag / colsums2 
precision2
recall2 = diag / rowsums2
recall2
f2 = 2 * precision2 * recall2 / (precision2 + recall2) 
f2

# Experimenting with other splits
rpmodel1.2 <- rpart(RESPONSE ~ ., data = mdTrn, parms = list(split='gini'))
rpmodel1.3 <- rpart(RESPONSE ~ ., data = mdTrn, parms = list(split='information'))

#with minsplits and maxdepths
rpmodel1.4 <- rpart(RESPONSE ~ ., data=mdTrn, parms = list(split ='gini'), control= rpart.control(minsplit=20, maxdepth=15))

#with cp
rpmodel1.5 <- rpart(RESPONSE ~ ., data=mdTrn, parms = list(split ='Gini'), control= rpart.control(cp=0.001))

#c5.0 model

cModel1.fit <- C5.0(RESPONSE ~ ., data=mdTrn,  method="class")
summary(c5.fit)
cpred1 <- predict(cModel1.fit, mdTst, type = "class")
table(pred = cpred1, true=mdTst$RESPONSE)
mean(cpred1==mdTst$RESPONSE)

cModel1.fit_MC <- C5.0(RESPONSE ~ ., data=mdTrn, method="class", control=C5.0Control(minCases=5))
cpred2 <- predict(cModel1.fit_MC, mdTst, type = "class")
table(pred = cpred2, true=mdTst$RESPONSE)
mean(cpred2==mdTst$RESPONSE)

preTst <- predict(rpmodel1, mdTst, type = "class")

table(pred = preTst, true = mdTst$RESPONSE)

confusionMatrix(preTst,mdTst$RESPONSE)

mean(preTst==mdTst$RESPONSE)

printcp(rpmodel1)
plotcp(rpmodel1)

summary(rpmodel1)

#Best model with one the accuracy and precision and less FPrate with 70-30 split

rpmodel1.4 <- rpart(RESPONSE ~ ., data=mdTrn, parms = list(split ='gini'), control= rpart.control(cp=0.001, minsplit=20, maxdepth=15))

######################################################################################################################################################


library(caTools)
library(ROCR)
library(rpart)
library(rpart.plot)

set.seed(123)
split = sample.split(dataset$RESPONSE, SplitRatio = 0.50)
training_set1 = subset(dataset, split == TRUE)
test_set1 = subset(dataset, split == FALSE)

r.ctrl1 <- rpart.control(minsplit = 30, minbucket = 10 , cp = 0, parms = list(split = 'infomation'))
dt_50split = rpart(formula = RESPONSE ~ .,
                   data = training_set1,
                   method = "class",
                   control = r.ctrl1
)

set.seed(123)
split2 = sample.split(dataset$RESPONSE, SplitRatio = 0.7)
training_set2 = subset(dataset, split2 == TRUE)
test_set2 = subset(dataset, split2 == FALSE)

dt_70split = rpart(formula = RESPONSE ~ .,
                   data = training_set2,
                   method = "class",
                   control = r.ctrl1
)

test_set2$score <-predict(dt_70split,test_set2, type = 'prob')
ROCRpred2<-prediction(test_set2$score[,2],test_set2$RESPONSE)
ROCRperf2 <- performance(ROCRpred2,"tpr","fpr")
plot(ROCRperf2, main = "ROC Curve")

cost.perf = performance(ROCRpred2, "cost")
optcutoff = ROCRpred2@cutoffs[[1]][which.min(cost.perf@y.values[[1]])]
optcutoff

cost.perf = performance(ROCRpred2, "cost", cost.fp = 2, cost.fn = 1)
ROCRpred2@cutoffs[[1]][which.min(cost.perf@y.values[[1]])]

acc.perf = performance(ROCRpred2, measure = "acc")
plot(acc.perf)

auc.perf = performance(ROCRpred2, measure = "auc")
auc.perf@y.values

cutoff = (acc.perf@x.values[[1]])
cutoff

PROFITVAL=3
COSTVAL=0.1363636

test_set2$score=predict(dt_70split,test_set2, type="prob")['default']
prLifts=data.frame(test_set2$score)
prLifts=cbind(prLifts, test_set2$RESPONSE)

prLifts=prLifts[order(-test_set2$score) ,]  

install.packages("dplyr")
library(dplyr)

prLifts<-prLifts %>% mutate(profits=ifelse(prLifts$`test_set2$RESPONSE`=='default', PROFITVAL, COSTVAL), cumProfits=cumsum(profits))


maxProfit= max(prLifts$cumProfits)
maxProfit_Ind = which.max(prLifts$cumProfits)
maxProfit_score = prLifts$scoreTst[maxProfit_Ind]
print(c(maxProfit = maxProfit, scoreTst = maxProfit_score))