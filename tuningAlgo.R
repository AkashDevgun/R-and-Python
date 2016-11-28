#install.packages('caret')
library(caret)
#install.packages('randomForest')
library(randomForest)

set.seed(415)

df <- read.csv("assessment_challenge.csv", na.strings=c("","NA"))
features <- names(df)

stringfeatures = c("from_domain_hash","Domain_extension","day")

features <- features[!features %in% c("read_rate")]
features <- features[!features %in% c("id")]

numericfeatures <- features[!features %in% stringfeatures]

for (each in numericfeatures) {
	df[each][is.na(df[each])] <- colMeans(df[each], na.rm=TRUE)
}

for (each in stringfeatures) {
	df[each][is.na(df[each])] <- names(which.max(table(df[each])))
}

numericdataFrame <- df[, numericfeatures]

stringdataFrame <- df[, stringfeatures]

single_stringvec = as.vector(as.matrix(stringdataFrame))

Encoded_strings <- as.numeric(factor(single_stringvec))

mat_strfeatures <- matrix(Encoded_strings,nrow = dim.data.frame(stringdataFrame)[1],ncol = dim.data.frame(stringdataFrame)[2])


Encoded_stringsdf <- as.data.frame(mat_strfeatures)

colnames(Encoded_stringsdf)<- stringfeatures

totalattributes_df <- cbind(df["read_rate"], Encoded_stringsdf, numericdataFrame)

sample.ind <- sample(2, nrow(totalattributes_df), replace = T, prob = c(0.7,0.3))

training <- totalattributes_df[sample.ind==1,]

testing <- totalattributes_df[sample.ind==2,]

training_feat <- names(training)

training$read_rate <- as.numeric(training$read_rate)


fitControl <- trainControl(method="repeatedcv", number=5, repeats=3, search="grid")

rf_Grid <- expand.grid(.mtry=c(1:3))

for (ntree in c(200, 300, 400, 500)) {

fit <- train(read_rate ~ ., data = training, method="rf", tuneGrid=rf_Grid, trControl = fitControl,ntree=ntree)
predictions <- predict(fit, testing)	

mse <- mean((testing$read_rate - predictions)^2)

print(mse)
}















# fitControl <- trainControl(method = "cv", number = 6, #3folds)

# rf_Grid <-  expand.grid(mtry = 6, coefReg = 0.1)

# training$read_rate <- as.numeric(training$read_rate)

# #training_feat <- training_feat[!training_feat %in% c("read_rate")]

# fit <- train(read_rate ~ ., data = training, method = "RRFglobal", trControl = fitControl, verbose = FALSE, tuneGrid = rf_Grid)

# #rf_Grid <-  expand.grid(interaction.depth = 15, n.trees = 400, shrinkage = 0.1, n.minobsinnode = 10)

# #training_feat <- training_feat[!training_feat %in% c("read_rate")]

# #training_features <- paste(training_feat, collapse = "+")

# #randomFormula <- as.formula(paste("read_rate", training_features, sep = " ~ "))



# #randomForestObj <- randomForest(read_rate ~ ., data = training[,training_feat], ntree=500, importance=T)

# predictions <- predict(fit, testing)

# #predictions <- predict(randomForestObj , testing)

