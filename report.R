library(tidyverse)
library(skimr)     
library(corrplot)

library(caret)      
library(mlr)        
library(rpart)     

data<-read.csv('D:/data/heart_failure.csv')

summary(data)
skim(data)
cor(data) %>%
  corrplot(method = "color", type = "lower", tl.col = "black", tl.srt = 45,
           p.mat = cor.mtest(data)$p,
           insig = "p-value", sig.level = -1)



require(smotefamily)
data<-SMOTE(data[,-13],data[,13],K=13)
data<-data$data
data$class<-as.factor(as.numeric(data$class))



library(mlbench)
library(caret)
control <- rfeControl(functions=rfFuncs, method="cv", number=10)
# run the RFE algorithm
results <- rfe(data[,1:12], data[,13], sizes=c(1:12), rfeControl=control)
# summarize the results
print(results)
# list the chosen features
predictors(results)
# plot the results
plot(results, type=c("g", "o"))

xx<-results[["fit"]][["importance"]]
xx<-data.frame(xx)
xx$MeanDecreaseAccuracy<-xx$MeanDecreaseAccuracy*100
library(ggplot2)
ggplot(data = xx, 
       mapping = aes(x = reorder(rownames(xx),-MeanDecreaseAccuracy), y = MeanDecreaseAccuracy))+ geom_bar(stat = 'identity')+theme(axis.text.x=element_text(angle=90),axis.title.x=element_blank())


ggplot(data = xx, 
       mapping = aes(x = reorder(rownames(xx),-MeanDecreaseGini), y = MeanDecreaseGini))+ geom_bar(stat = 'identity')+theme(axis.text.x=element_text(angle=90),axis.title.x=element_blank())



data<-data[,c(results$optVariables,'class')]

set.seed(15)
train_sub = sample(nrow(data),8/10*nrow(data))
train_data = data[train_sub,]
test_data =data[-train_sub,]
#random forest
library(randomForest)
model_randomforest <- randomForest(class ~ ., data = train_data, importance = TRUE)
model_randomforest
temp_randomforest<-data.frame(predict(model_randomforest,test_data[,-12]))

colnames(temp_randomforest)<-'temp_randomforest'
#ROC
library(caret)
aa<-confusionMatrix(test_data$class,temp_randomforest$temp_randomforest)

print(aa[["overall"]][["Accuracy"]])
ntre<-data.frame(c(10,100,200,300,400,500,600,700,800,1000,2000))
ntre$acc<-0
colnames(ntre)[1]<-'ntre'
for (i in 1:9) {
  model_randomforest <- randomForest(class ~ ., data = train_data, importance = TRUE,ntree=ntre[i,1])
  model_randomforest
  temp_randomforest<-data.frame(predict(model_randomforest,test_data[,-13]))
  
  colnames(temp_randomforest)<-'temp_randomforest'
  #ROC
  library(caret)
  aa<-confusionMatrix(test_data$class,temp_randomforest$temp_randomforest)
  
  ntre[i,2]<-aa[["overall"]][["Accuracy"]]
}


set.seed(4)
train_sub = sample(nrow(data),8/10*nrow(data))
train_data = data[train_sub,]
test_data =data[-train_sub,]
#random forest
library(randomForest)
model_randomforest <- randomForest(class ~ ., data = train_data, importance = TRUE)
model_randomforest

temp_randomforest<-data.frame(predict(model_randomforest,test_data[,-12]))

colnames(temp_randomforest)<-'temp_randomforest'
#ROC
library(caret)
confusionMatrix(test_data$class,temp_randomforest$temp_randomforest)
library(pROC)
temp_randomforest_prob<-data.frame(predict(model_randomforest,test_data[,-13],type = 'prob'))

roc_randomforest<-roc(as.integer(test_data$class)-1,temp_randomforest_prob$X1)
plot(roc_randomforest,print.AUC=T,col='red',legacy.axes=T)

#logistic
library(rms)
model_logistic<-lrm(class~.,data=train_data,x=T,y=T)
#summary(model_logistic)

#predict
temp_logistic<-predict(model_logistic,newdata = test_data[,-12],type = "fitted")
temp_logistic<-data.frame(ifelse(temp_logistic>=0.5,1,0))
colnames(temp_logistic)<-'temp_logistic'


confusionMatrix(test_data$class,factor(temp_logistic$temp_logistic))

temp_logistic_prob<-predict(model_logistic,newdata = test_data[,-12],type = "fitted")

roc_logistic<-roc(as.integer(test_data$class)-1,temp_logistic_prob)
plot(roc_logistic,print.AUC=T,add=T,col='blue')



#decision tree
library(rpart)
tc <- rpart.control(minbucket=5,maxdepth=10,xval=5,cp=0.005)
model_decisiontree <- rpart(class ~ ., data=train_data, control="tc")
temp_decisiontree <- data.frame(predict(model_decisiontree, test_data[,-12], type="class"))
colnames(temp_decisiontree)<-'temp_decisiontree'


confusionMatrix(test_data$class,factor(temp_decisiontree$temp_decisiontree))

temp_decisiontree_prob <- predict(model_decisiontree, test_data[,-12], type="prob")
roc_decsiontree<-roc(as.integer(test_data$class)-1,temp_decisiontree_prob[,1])
plot(roc_decsiontree,print.AUC=T,add=T,col='green')


##Adaboost
library(adabag)
model_adaboost<-boosting(class~.,data = train_data,boos = T,mfinal = 10)

temp_adaboost<-data.frame(predict(model_adaboost,test_data[,-12])$class)
colnames(temp_adaboost)<-'temp_adaboost'
temp_adaboost<-ifelse(temp_adaboost=="1",1,0)

confusionMatrix(test_data$class,factor(temp_adaboost))

temp_adaboost_prob<-predict(model_adaboost,test_data[,-12])$prob[,1]
roc_adaboost<-roc(as.integer(test_data$class)-1,temp_adaboost_prob)
plot(roc_adaboost,print.AUC=T,add=T,col='purple')


#plot
legend(x=0.4,y=0.5, legend=c("RandomForest AUC=0.9759","Logistic AUC=0.9217",
                             "Adaboost AUC=0.9586",
                             "Decision Tree AUC=0.9297")
       ,cex =0.6,lwd=2,
       col=c("red","blue","purple","black"),inset=.5,lty=c(1,2,3,4))




p_positive <- temp_randomforest_prob$X0
sor <- order(p_positive)
p_positive <- p_positive[sor]
y <- test_data$class[sor]

y <- ifelse(y == "0",1,0)

groep <- cut2(p_positive, g = 10)

meanpred_rm <- round(tapply(p_positive, groep, mean), 3)
meanobs <- round(tapply(y, groep, mean), 3) 

finall_rm <- data.frame(meanpred_rm = meanpred_rm, meanobs_rm = meanobs)

ggplot(finall_rm,aes(x = meanpred_rm,y = meanobs_rm))+ 
  geom_line(linetype = 2)+ 
  geom_abline(slope = 1,intercept = 0,color = "red")+ 
  labs(x="Predicted Probability",y = "Observed Probability",title = "calibration_curve")


######################logistic################
p_positive <- temp_logistic_prob
sor <- order(p_positive)
p_positive <- p_positive[sor]
y <- test_data$class[sor]

y <- ifelse(y == "1",1,0)

groep <- cut2(p_positive, g = 10)

meanpred_lr <- round(tapply(p_positive, groep, mean), 3)
meanobs <- round(tapply(y, groep, mean), 3) 

finall_lr <- data.frame(meanpred_lr = meanpred_lr, meanobs_lr = meanobs)


ggplot(finall_lr,aes(x = meanpred_lr,y = meanobs_lr))+ 
  geom_line(linetype = 2)+ 
  geom_abline(slope = 1,intercept = 0,lty="solid",color = "red")+ 
  labs(x="Predicted Probability",y = "Observed Probability",title = "calibration_curve")
############Adaboost########

p_positive <- temp_adaboost_prob
sor <- order(p_positive)
p_positive <- p_positive[sor]
y <- test_data$class[sor]

y <- ifelse(y == "1",1,0)

groep <- cut2(p_positive, g = 10)

meanpred_ada <- round(tapply(p_positive, groep, mean), 3)


#######decisiontree#########

p_positive <- temp_decisiontree_prob[,1]
sor <- order(p_positive)
p_positive <- p_positive[sor]
y <- test_data$class[sor]

y <- ifelse(y == "1",1,0)

groep <- cut2(p_positive, g = 10)

meanpred_decision <- round(tapply(p_positive, groep, mean), 3)
fill<-cbind.data.frame(meanobs,meanpred_rm,meanpred_lr,meanpred_ada,meanpred_decision)


ggplot()+geom_line(data = fill,aes(x = meanobs,y = meanpred_rm,colour = "RandomForest"),size=1,linetype = 1)+
  #geom_point(data = data,aes(x = year,y = GDP,colour = "GDP"),size=3)+
  ylim(0,1)+
  geom_line(data = fill,aes(x = meanobs,y = meanpred_lr,colour = "Logistic"),size=1,linetype = 2) + 
  geom_line(data = fill,aes(x = meanobs,y = meanpred_ada,colour = "Adaboost"),size=1,linetype = 3) +
  geom_line(data = fill,aes(x = meanobs,y = meanpred_decision,colour = "Decision Tree"),size=1,linetype = 4) + 
  #geom_line(data = data,aes(x = year,y = FDI,colour ="FDI"),size=1) +
  #scale_colour_manual("",values = c("GDP" = "red","FDI" = "green","DI"="yellow"))+
  geom_abline(slope = 1,intercept = 0,lty="solid",color = "red")+ 
  labs(x="Predicted Probability",y = "Observed Probability",title = "calibration_curve")+
  theme_light() 




