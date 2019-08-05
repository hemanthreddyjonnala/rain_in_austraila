if(!require(InformationValue)) install.packages("InformationValue")
if(!require(caTools)) install.packages("caTools")
if(!require(caret)) install.packages("caret")
if(!require(naivebayes)) install.packages("naivebayes")
if(!require(party)) install.packages("party")
if(!require(rpart.plot)) install.packages("rpart.plot")
if(!require(e1071)) install.packages("e1071")
if(!require(MLMetrics)) install.packages("MLMetrics")
if(!require(varhandle)) install.packages("varhandle")
if(!require(purrr)) install.packages("purrr")
if(!require(corrplot)) install.packages("corrplot")
if(!require(ggplot2)) install.packages("ggplot2")
if(!require(cowplot)) install.packages("cowplot")


library(readr)
library(dplyr)
library(party)
library(rpart)
library(rpart.plot)
library(ROCR)
library(caret)


setwd("/cloud/project")
getwd()

# Import test data
weather_data = "weatherAUS.csv"
raw_data<- read.csv(weather_data)

head(raw_data)  # display the first 6 observations
print(summary(raw_data))

# converting categrical data numeric data
objects <- c("Location", "WindGustDir", "WindDir9am", "WindDir3pm", "RainToday", "RainTomorrow")
for (obj in objects){
  print(rev(levels(raw_data[[obj]])))
  raw_data[[obj]]<- unclass(factor(raw_data[[obj]], levels=rev(levels(raw_data[[obj]]))))
}


# replacing NA values with zero
data <- na.omit(raw_data)
drops <- c("Date","RISK_MM")
data <-data[ , !(names(data) %in% drops)]

# check for correlated factors
numeric <- map_lgl(data, is.numeric)

correlations <- cor(data[,numeric])

diag(correlations) <- 0

high <- apply(abs(correlations) >= 0.8, 2, any)

corrplot(correlations[high, high], method = "number")

# feature extraction using PCA
data_PCA <- prcomp(data,scale.=T)
class(data_PCA)
plot(data_PCA)
summary(data_PCA)
head(data_PCA)
par(data_PCA)
biplot(data_PCA, cex=0.5, cex.axis=0.5)

screeplot(data_PCA, type = "l", npcs = 15, main = "Screeplot of the first 10 PCs")
abline(h = 1, col="red", lty=5)
legend("topright", legend=c("Eigenvalue = 1"),
       col=c("red"), lty=5, cex=0.6)

cumpro <- cumsum(data_PCA$sdev^2 / sum(data_PCA$sdev^2))
plot(cumpro[0:15], xlab = "PC #", ylab = "Amount of explained variance", main = "Cumulative variance plot")
abline(v = 6, col="blue", lty=5)
abline(h = 0.72887, col="blue", lty=5)
legend("topleft", legend=c("Cut-off @ PC6"),
       col=c("blue"), lty=5, cex=0.6)

cor(data,data$RainTomorrow)
keep <-c('MaxTemp','Evaporation','Sunshine','Pressure9am','Temp3pm','RainToday','RainTomorrow')
data <-data[ , (names(data) %in% keep)]
#First, we convert rank to a factor to indicate that rank should be treated as a categorical variable.
data$RainToday[which(data$RainToday == 2)]<- 0 
data$RainTomorrow[which(data$RainTomorrow == 2)]<- 0 
data$RainToday <- factor(data$RainToday)


# Check distribution between the 2 binary classes
table(data$RainTomorrow)
xtabs(~RainTomorrow + RainToday, data = data)
xtabs(~RainTomorrow + Location, data = data)


# Create Training Data
input_ones <- data[which(data$RainTomorrow == 1), ]  # all 1's
input_zeros <- data[which(data$RainToday == 0), ]  # all 0's
set.seed(123)  # for repeatability of samples
input_ones_training_rows <- sample(1:nrow(input_ones), 0.8*nrow(input_ones))  # 1's for training
input_zeros_training_rows <- sample(1:nrow(input_zeros), 0.8*nrow(input_ones))  # 0's for training. Pick as many 0's as 1's
training_ones <- input_ones[input_ones_training_rows, ]  
training_zeros <- input_zeros[input_zeros_training_rows, ]
trainingData <- rbind(training_ones, training_zeros)  # row bind the 1's and 0's 

# Create Test Data
test_ones <- input_ones[-input_ones_training_rows, ]
test_zeros <- input_zeros[-input_zeros_training_rows, ]
testData <- rbind(test_ones, test_zeros)  # row bind the 1's and 0's 

##########################################################################
start_time <- Sys.time()
# apply logistic regression
logit <- glm(RainTomorrow ~ ., data=trainingData)  # build the model
summary(logit)

# CIs using standard errors
confint.default(logit)
#To get the exponentiated coefficients, you tell R that you want to exponentiate (exp)
## odds ratios only
exp(coef(logit))
with(logit, null.deviance - deviance)
distPred <- plogis(predict(logit, testData))  # predict distance
library(InformationValue)
optCutOff <- optimalCutoff(testData$RainTomorrow, distPred)[1] 
optCutOff
misClassError(testData$RainTomorrow, distPred, threshold = optCutOff)
library(MLmetrics)
testData$RainTomorrow <- factor(testData$RainTomorrow)
distPred <- ifelse(distPred < optCutOff, 0 ,1)
distPred<-factor(distPred)
# getting acuracy, ROC, AUC, RECALL, PRECISION AND F VALUES
plotROC(testData$RainTomorrow, as.numeric(distPred))
AUC(testData$RainTomorrow, distPred)
recall(testData$RainTomorrow, distPred)
precision(testData$RainTomorrow, distPred)
F_meas(testData$RainTomorrow, distPred)

confusionMatrix(testData$RainTomorrow, distPred)
end_time <- Sys.time()
#getting time difference
end_time - start_time


##################################################################
start_time <- Sys.time()
#apply naive bayes algorithm
trainingData$RainTomorrow <- factor(trainingData$RainTomorrow)
nb_rain<-naive_bayes(RainTomorrow~.,data=trainingData)
nb_pred<-predict(nb_rain,testData)
summary(nb_rain)

# getting acuracy, ROC, AUC, RECALL, PRECISION AND F VALUES
plotROC(testData$RainTomorrow, as.numeric(nb_pred))
AUC(testData$RainTomorrow, nb_pred)
recall(testData$RainTomorrow, nb_pred)
precision(testData$RainTomorrow, nb_pred)
F_meas(testData$RainTomorrow, nb_pred)
# getting the confusion matrix and stastics
confusionMatrix(testData$RainTomorrow, nb_pred)
end_time <- Sys.time()
#getting time difference
end_time - start_time
######################################################################
#apply decision tree alogorithm
start_time <- Sys.time()

# Conditional partitioning is implemented in the "ctree" method
rtree_fit2 <- ctree(trainingData$RainTomorrow ~ ., 
                    trainingData) 
plot(rtree_fit2)
summary(rtree_fit2)

# We used the fit tree from the train data and test with the test data
predicted= predict(rtree_fit2,testData)

summary(predicted)

# getting acuracy, ROC, AUC, RECALL, PRECISION AND F VALUES
plotROC(testData$RainTomorrow, as.numeric(predicted))
AUC(testData$RainTomorrow, predicted)
recall(testData$RainTomorrow, predicted)
precision(testData$RainTomorrow, predicted)
F_meas(testData$RainTomorrow, predicted)
confusionMatrix(testData$RainTomorrow, predicted)
end_time <- Sys.time()
#getting time difference
end_time - start_time

formulas <- as.formula(trainingData$RainTomorrow ~ .)
#tree construction based on information gain
tree = rpart(formulas, data=trainingData, method = 'class', parms = list(split = "information"))
rpart.plot(tree)
########################################################################
# Fitting K-NN to the Training set and Predicting the Test set results
start_time <- Sys.time()
library(class)
kn_pred = knn(train = trainingData,
              test = testData,
              cl = trainingData$RainTomorrow,
              k = 5,
              prob = TRUE)


# getting acuracy, ROC, AUC, RECALL, PRECISION AND F VALUES
plotROC(testData$RainTomorrow, as.numeric(kn_pred))
AUC(testData$RainTomorrow, kn_pred)
recall(testData$RainTomorrow, kn_pred)
precision(testData$RainTomorrow, kn_pred)
F_meas(testData$RainTomorrow, kn_pred)

#Confusion matrix   
knn_confusion<-confusionMatrix(factor(kn_pred), factor(testData$RainTomorrow),positive="1")
knn_confusion



end_time <- Sys.time()
#getting time difference
end_time - start_time
#############################################################################
# svm
start_time <- Sys.time()
library(e1071)
svmfit = svm(formula = RainTomorrow ~ .,
             data = trainingData,
             type = 'C-classification',
             kernel = 'linear')

# Predicting the Test set results
y_pred_svm = predict(svmfit, newdata = testData)


# getting acuracy, ROC, AUC, RECALL, PRECISION AND F VALUES
plotROC(testData$RainTomorrow, as.numeric(y_pred_svm))
AUC(testData$RainTomorrow, y_pred_svm)
recall(testData$RainTomorrow, y_pred_svm)
precision(testData$RainTomorrow, y_pred_svm)
F_meas(testData$RainTomorrow, y_pred_svm)

# Making the Confusion Matrix and stastics
confusionMatrix(testData$RainTomorrow, y_pred_svm)
end_time <- Sys.time()
#getting time difference
end_time - start_time
###############################################################################

############ ensemble methods
#Taking average of predictions
predicted_dt<-predicted
library(varhandle)
testData$pred_avg<-(unfactor(distPred)+unfactor(predicted_dt)+unfactor(kn_pred)+unfactor(nb_pred)+unfactor(y_pred_svm))/5

#Splitting into binary classes at 0.5
testData$pred_avg<-as.factor(ifelse(testData$pred_avg>0.5,1, 0))
confusionMatrix(testData$RainTomorrow, testData$pred_avg)
# getting acuracy, ROC, AUC, RECALL, PRECISION AND F VALUES
plotROC(testData$RainTomorrow, as.numeric(testData$pred_avg))
AUC(testData$RainTomorrow, testData$pred_avg)
recall(testData$RainTomorrow, testData$pred_avg)
precision(testData$RainTomorrow, testData$pred_avg)
F_meas(testData$RainTomorrow, testData$pred_avg)

# majority voting
#The majority vote
# knn model, svm model, logistic regression selected for majority voting
testData$pred_majority<-as.factor(ifelse(unfactor(distPred)==1 & unfactor(kn_pred)==1,1,
                                         ifelse(unfactor(distPred)==1 & unfactor(y_pred_svm)==1,1,
                                                ifelse(unfactor(kn_pred)==1 & unfactor(y_pred_svm)==1,1,
                                                       ifelse(unfactor(kn_pred)==1 & unfactor(y_pred_svm)==1 &unfactor(distPred) == 1,1,0)))))
confusionMatrix(testData$RainTomorrow, testData$pred_majority)

# getting acuracy, ROC, AUC, RECALL, PRECISION AND F VALUES
plotROC(testData$RainTomorrow, as.numeric(testData$pred_majority))
AUC(testData$RainTomorrow, testData$pred_majority)
recall(testData$RainTomorrow, testData$pred_majority)
precision(testData$RainTomorrow, testData$pred_majority)
F_meas(testData$RainTomorrow, testData$pred_majority)

