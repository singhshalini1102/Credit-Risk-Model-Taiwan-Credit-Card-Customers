library(dplyr)
library(plyr)
library(caret)
library(randomForest)
library(e1071)
library(ROCR)
library(rpart)
library(adabag)
library(foreach)
library(pROC)
library(AUC)
library(MLmetrics)

getwd()

dir_path <- "/Users/shalinisingh/Desktop/Projects/CreditRisk-Taiwan/data"
setwd(dir_path)
rm(list=ls())
infile<-"default_of_credit_card_clients.csv"
default_credit =read.csv(infile, header = TRUE, sep = ",")
nrow(default_credit)

# target variable
default_credit$default.payment.next.month <- as.factor(default_credit$default.payment.next.month)
class(default_credit$default.payment.next.month)

default_credit <- default_credit[,c(-1)]

# total defaulters
defaulters <- count(default_credit, "default.payment.next.month")


# count according to gender. 1= Male, 2= Female
sex <- count(default_credit, "SEX")
barplot(sex$freq,main = "applicants count as per gender",names= sex$SEX, xlab = "SEX", ylab = "count", col="light green", ylim = c(0,20000))

default_payment_1 <- subset(default_credit, default_credit$default.payment.next.month ==1)
class(default_payment_1)
View(default_payment_1)
nrow(default_payment_1)

# defaulters according to gender. 1= Male, 2= Female
default_sex <- count(default_payment_1, "SEX")
barplot(default_sex$freq,main = "defaulters as per gender",names= default_sex$SEX, xlab = "SEX", ylab = "defaulters", col="powderblue", ylim = c(0,4500))

# count according to education. 1= Graduate School, 2= University, 3= High school, 4= others
education <- count(default_credit, "EDUCATION")
barplot(education$freq,main = "applicants count as per education level",names= education$EDUCATION, xlab = "EDUCATION", ylab = "count", col="#FFCCCC", ylim = c(0,20000))

# defaulters according to education. 1= Graduate School, 2= University, 3= High school, 4= others
default_education <- count(default_payment_1, "EDUCATION")
barplot(default_education$freq,main = "defaulters as per EDUCATION",names= default_education$EDUCATION, xlab = "EDUCATION", ylab = "defaulters", col="#FFCCCC", ylim = c(0,4000))
text(default_education$freq, labels = default_education$freq, pos = 3)

# count according to marriage. 1= married, 2= single, 3= others
marriage <- count(default_credit, "MARRIAGE")
barplot(marriage$freq,main = "applicants count as per marriage status",names= marriage$MARRIAGE, xlab = "MARRIAGE", ylab = "count", col="#CCFFFF", ylim = c(0,20000))

# defaulters according to marriage. 1= married, 2= single, 3= others
default_marriage <- count(default_payment_1, "MARRIAGE")
barplot(default_marriage$freq,main = "defaulters as per MARRIAGE",names= default_marriage$MARRIAGE, xlab = "MARRIAGE", ylab = "defaulters", col="#CCFFFF", ylim = c(0,4000))
text(default_marriage$freq, labels = default_marriage$freq, pos = 3)

default_credit$PAY_AMT1_p<-(PAY_AMT1 /LIMIT_BAL)
default_credit$PAY_AMT2_p<-(PAY_AMT2 /LIMIT_BAL)
default_credit$PAY_AMT3_p<-(PAY_AMT3 /LIMIT_BAL)
default_credit$PAY_AMT4_p<-(PAY_AMT4 /LIMIT_BAL)
default_credit$PAY_AMT5_p<-(PAY_AMT5 /LIMIT_BAL)
default_credit$PAY_AMT6_p<-(PAY_AMT6 /LIMIT_BAL)

#Correlaion Matrix
library(corrplot)

default_credit_num <- subset(default_credit,select = c(PAY_0,PAY_2, PAY_3, PAY_4, PAY_5, PAY_6,
                                                       PAY_AMT1_p,PAY_AMT2_p,PAY_AMT3_p,PAY_AMT4_p,
                                                       PAY_AMT5_p,PAY_AMT6_p,
                                                       PAY_AMT3,PAY_AMT4,PAY_AMT5,
                                                       PAY_AMT6,BILL_AMT1,BILL_AMT2,BILL_AMT3,BILL_AMT4,
                                                       BILL_AMT5,BILL_AMT6))
default_credit_corr <-cor(default_credit_num)
corrplot(default_credit_corr, method = "number")

#sampling

set.seed(12)
indx <- sample(2, nrow(default_credit), replace = TRUE, prob = c(0.7,0.3))
train_data <- default_credit[indx == 1,]
test_data <- default_credit[indx == 2 ,]
summary(default_credit)

# 10 equal size folds
folds <- cut(seq(1,nrow(default_credit)),breaks=10,labels=FALSE)
table(folds)
#output
folds


#adaboost
CV_boost <- lapply(1:10, function(x){ 
  model <-boosting(default.payment.next.month ~SEX+MARRIAGE+AGE+EDUCATION+PAY_0+PAY_6+BILL_AMT2+BILL_AMT4 +PAY_AMT3_p+ PAY_AMT5_p, data = default_credit[folds != x,], mfinal = 50)
  preds <- predict.boosting(model, newdata =  default_credit[folds == x,])
  real <- default_credit$default.payment.next.month[folds == x]
  #conf <-confusionMatrix(preds, real)
  print(preds$confusion)
  return(data.frame(preds$class, real))
})
CV_boost <- do.call(rbind, CV_boost)

confusionMatrix(CV_boost$preds, CV_boost$real)

## Decision Tree
control= rpart.control(minsplit = 50,minbucket = 100,cp=0, maxdepth = 6)
CV_default_tree <- lapply(1:10, function(x){ 
  model <-rpart(default.payment.next.month~SEX+MARRIAGE+AGE+EDUCATION+
                  PAY_0+PAY_6+BILL_AMT2+BILL_AMT4+PAY_AMT3_p+PAY_AMT5_p
                ,data = default_credit[folds != x,],method = "class", control=control,
                parms=list(split = "information"))
  preds <- predict(model,  default_credit[folds == x,], type="class")
  real <- default_credit$default.payment.next.month[folds == x]
  conf <-confusionMatrix(preds, real)
  return(data.frame(preds, real))
})
CV_default_tree <- do.call(rbind, CV_default_tree)
confusionMatrix(CV_default_tree$preds, CV_default_tree$real)


## Naive Bayes
CV_default_n <- lapply(1:10, function(x){ 
  model <-naiveBayes(default.payment.next.month~SEX+MARRIAGE+AGE+EDUCATION+PAY_0+PAY_6+BILL_AMT2+BILL_AMT4 +PAY_AMT3_p+ PAY_AMT5_p,data = default_credit[folds != x,] )
  preds <- predict(model,  default_credit[folds == x,], type= "class")
  real <- default_credit$default.payment.next.month[folds == x]
  conf <-confusionMatrix(preds, real)
  return(data.frame(preds, real))
})
CV_default_n <- do.call(rbind, CV_default_n)
confusionMatrix(CV_default_n$preds, CV_default_n$real)

# Logistic Regression
CV_default_logit <- lapply(1:10, function(x){ 
  model <-logit2<-glm(default.payment.next.month~SEX+MARRIAGE+AGE+EDUCATION+
                        PAY_0+PAY_6+BILL_AMT2+BILL_AMT4 +PAY_AMT3_p+ PAY_AMT5_p,
                      data = default_credit[folds != x,],family = 'binomial')
  preds <- predict(model,  default_credit[folds == x,], type="response")
  preds1 = as.factor(ifelse(preds > 0.5, "1", "0"))
  real <- default_credit$default.payment.next.month[folds == x]
  conf <-confusionMatrix(preds1, real)
  return(data.frame(preds1, real))
})
CV_default_logit <- do.call(rbind, CV_default_logit)
confusionMatrix(CV_default_logit$preds1, CV_default_logit$real)

## ROC 

# AdaBoost
pred_ada <- predict.boosting(model_ada, newdata =  test_data)
pr_ada <- prediction(pred_ada$prob[,2],test_data$default.payment.next.month)
perf_ada<-performance(pr_ada,"tpr", "fpr")
plot(perf_ada)
auc = performance(pr_ada, 'auc')
slot(auc, 'y.values')

# DT
pred_dt <- predict(DT_model6,test_data , type="prob")
pr_dt <- prediction(pred_dt[,2],test_data$default.payment.next.month)
perf_dt<-performance(pr_dt,"tpr", "fpr")
plot(perf_dt)
auc = performance(pr_dt, 'auc')
slot(auc, 'y.values')

# Naive
pred_test_naive<-predict(model2, newdata = test_data, type="raw")
p_test_naive<-prediction(pred_test_naive[,2], test_data$default.payment.next.month)
perf_naive<-performance(p_test_naive, "tpr", "fpr")
plot(perf_naive, colorize=T)
performance(p_test_naive, "auc")@y.values

# Logistic
predict_test2 <- predict(logit2, newdata = test_data, type= 'response')
pred_logit <- prediction(predict_test2, test_data$default.payment.next.month)
pr_logit <- performance(pred_logit, 'tpr','fpr')
plot(pr_logit)
auc = performance(pr_logit, 'auc')
slot(auc, 'y.values')


plot(perf_dt)
plot(perf_naive, add= TRUE, col="red")
plot(pr_logit, add= TRUE, col="blue")
plot(perf_ada, add= TRUE, col="cyan")
legend(0.6, 0.3, legend=c("DT", "Naive", "Logistic","AdaBoost"),
       col=c("black", "red","blue","cyan"), lty=1:2, cex=0.8)

#PR curve

#AdaBoost
PR_ada <- performance(pr_ada, 'prec','tpr')
plot(PR_ada)
PRAUC(pred_ada,test_data$default.payment.next.month)

# DT
PR_dt <- performance(pr_dt,'prec', 'tpr')
plot(PR_dt)
PRAUC(pred_dt,test_data$default.payment.next.month)

# Naive
PR_naive <- performance(p_test_naive,'prec', 'tpr')
plot(PR_naive)
PRAUC(pred_test_naive,test_data$default.payment.next.month)

#Logistic
PR_logit <- performance(pred_logit,'prec', 'tpr')
plot(PR_logit)
PRAUC(predict_test2,test_data$default.payment.next.month)

plot(PR_dt)
plot(PR_naive, add= TRUE, col="red")
plot(PR_logit, add= TRUE, col="blue")
plot(PR_ada, add= TRUE, col="cyan")
legend(0.7, 1.0, legend=c("DT", "Naive", "Logistic","AdaBoost"),
       col=c("black", "red","blue","cyan"), lty=1:2, cex=0.8)
