#loading the RandomForrest library
library(randomForest)

#loading the Caret library
library(caret)

#Set the seed value
set.seed(415)

#Load the data
df <- read.csv("assessment_challenge.csv", na.strings=c("","NA"))

#All features
features <- names(df)


#Separating the numeric feautuers and string features
stringfeatures = c("from_domain_hash","Domain_extension","day")

features <- features[!features %in% c("read_rate")]
features <- features[!features %in% c("id")]

numericfeatures <- features[!features %in% stringfeatures]

#Handling the missing data
for (each in numericfeatures) {
  df[each][is.na(df[each])] <- colMeans(df[each], na.rm=TRUE)
}

for (each in stringfeatures) {
  df[each][is.na(df[each])] <- names(which.max(table(df[each])))
}

numericdataFrame <- df[, numericfeatures]

stringdataFrame <- df[, stringfeatures]

#assigning Unique Identifiers to string featuers labels
single_stringvec = as.vector(as.matrix(stringdataFrame))

Encoded_strings <- as.numeric(factor(single_stringvec))

mat_strfeatures <- matrix(Encoded_strings,nrow = dim.data.frame(stringdataFrame)[1],ncol = dim.data.frame(stringdataFrame)[2])


Encoded_stringsdf <- as.data.frame(mat_strfeatures)

colnames(Encoded_stringsdf)<- stringfeatures

totalattributes_df <- cbind(df["read_rate"], Encoded_stringsdf, numericdataFrame)

#Cross Validation
sample.ind <- sample(2, nrow(totalattributes_df), replace = T, prob = c(0.7,0.3))

training <- totalattributes_df[sample.ind==1,]

testing <- totalattributes_df[sample.ind==2,]

training_feat <- names(training)

training$read_rate <- as.numeric(training$read_rate)
#Training with numtrees 100
randomForestObj <- randomForest(read_rate ~ ., data = training, ntree=100)

predictions <- predict(randomForestObj, testing)

#mse error
mse <- mean((testing$read_rate - predictions)^2)

print(mse)

varImpPlot(randomForestObj,type=2)

#Training with numtrees 300
randomForestObj <- randomForest(read_rate ~ ., data = training, ntree=300)

predictions <- predict(randomForestObj, testing)

mse <- mean((testing$read_rate - predictions)^2)
print(mse)

varImpPlot(randomForestObj,type=2)

#Grid search for parameter tuning..Time consuming
fitControl <- trainControl(method="repeatedcv", number=3, repeats=2, search="grid")

rf_Grid <- expand.grid(.mtry=c(1:3))

for (ntree in c(100, 300, 500)) {
  
  fit <- train(read_rate ~ ., data = training, method="rf", tuneGrid=rf_Grid, trControl = fitControl,ntree=ntree)
  predictions <- predict(fit, testing)	
  
  mse <- mean((testing$read_rate - predictions)^2)
  
  print(mse)
}

# For extra trees.. computationally very Expensive
for (ntree in c(100, 300, 500)) {
  
  fitEx <- train(read_rate ~ ., data = training, method="extraTrees", tuneGrid=rf_Grid, trControl = fitControl,ntree=ntree)
  predictions <- predict(fitEx, testing)	
  
  mseEx <- mean((testing$read_rate - predictions)^2)
  
  print(mseEx)
}