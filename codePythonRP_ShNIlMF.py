from sklearn.linear_model import RandomizedLasso
import argparse
import numpy as np 
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV
from sklearn.linear_model import LassoLarsCV
from sklearn.grid_search import RandomizedSearchCV
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt; plt.rcdefaults()
import pandas as pd 
import matplotlib.pyplot as plt
import matplotlib
from sklearn import preprocessing
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import RandomizedLasso
from sklearn import ensemble
from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import Pipeline
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import mean_squared_error
import sklearn
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import Imputer
from sklearn.base import TransformerMixin



class DataFrameImputer(TransformerMixin):

    def __init__(self):
        """Imputing missing values.

        Columns of dtype object or string objects are imputed with the most frequent value 

        Columns of numeric types are imputed by taking the mean of the column.

        """
    def fit(self, X, y=None):

        self.fill = pd.Series([X[c].mean()
            if X[c].dtype == np.dtype('float64') or X[c].dtype == np.dtype('int64') else X[c].value_counts().index[0] for c in X],
            index=X.columns)

        return self

    def transform(self, X, y=None):
        return X.fillna(self.fill)

    def fit_transform(self,X,y=None):
        return self.fit(X,y).transform(X)




def End ():
    print "Question Finish"


if __name__ == "__main__":
    print('The scikit-learn version is {}.'.format(sklearn.__version__))
    parser = argparse.ArgumentParser(description='Question_1')

    args = parser.parse_args()


    #Loading the Dataset
    train = pd.read_csv('assessment_challenge.csv')
    
    #Computed the Length of Dataset
    lengthTrain = len(train)

    #Seprating y-values or predictors from the dataset
    ratings = train['read_rate'].tolist()
    train.drop('read_rate', axis=1, inplace=True)

    #All feature columns 
    allcolumns = train.columns.values
    
    #Removing irrelevant attributes from dataset
    remove = ['id']
    finalcols1 = []

    for each in allcolumns:
        if each not in remove:
            finalcols1.append(each)


    #Separating Stringfeature and numericfeatures       
    stringfeature = ['from_domain_hash','Domain_extension','day']
    numericfeature = []

    for each in finalcols1:
        if each not in stringfeature:
            numericfeature.append(each)

    train[finalcols1].replace('N/A',np.nan,regex=True, inplace=True)

    numericTraindf = train[numericfeature].apply(pd.to_numeric, errors='coerce')

    stringTraindf = train[stringfeature]

    #Handling the Missing Value in the Last Row

    finalnumeric_df = DataFrameImputer().fit_transform(numericTraindf)

    finalstring_df = DataFrameImputer().fit_transform(stringTraindf)


    #Encoded the string feature to unique numeric identifier
    laEn = preprocessing.LabelEncoder()

    train_X = np.reshape(finalstring_df.values,(len(finalstring_df)*len(finalstring_df.columns.values))) 

    trainLabelEncode = laEn.fit_transform(train_X)


    Encodedtrain_X = np.reshape(trainLabelEncode, (len(finalstring_df),len(finalstring_df.columns.values))) 

    finalEncodedstring_df = pd.DataFrame(Encodedtrain_X, columns = stringfeature)


    #After Encoding the string features, concatenate the numeric and encoded string features
    finaltraindf = pd.concat([finalnumeric_df,finalEncodedstring_df], axis=1)

    cols= finaltraindf.columns.values


    rng = np.random.RandomState(1)


    #Cross Validation Phase by dividing the data into training and testing
    X_dummytrain, X_dummytest, y_dummytrain, y_dummytest = train_test_split(finaltraindf.values, ratings, test_size=0.3, random_state=42)

    #Regressor Models   
    feature_regressors = [
    Lasso(alpha = 0.1, max_iter = 20000),
    Ridge(alpha=0.1, max_iter = 20000),
    ensemble.RandomForestRegressor(bootstrap = True, max_depth=11, n_estimators=400),
    ensemble.ExtraTreesRegressor(bootstrap = True, n_estimators = 300, max_depth = 15),
    ensemble.GradientBoostingRegressor(alpha=0.1, n_estimators = 500, learning_rate = 0.1, max_depth = 10 , random_state = 0)]

    #fitting the models and computing the errors
    print "#######################################################"
    print "fitting the models and computing the errors"
    mse = []
    for clf in feature_regressors:
        estimate = clf.fit(X_dummytrain,y_dummytrain)
        predictions = estimate.predict(X_dummytest)

        errorcomputed = mean_squared_error(y_dummytest, predictions)

        print clf
        print "Error is " + str(errorcomputed)
        print "#######################################################"

        mse.append(errorcomputed)

    objects = ('Lasso', 'Ridge', 'Random Forest', 'Extra Trees', 'Gradient Boosting')
    y_pos = np.arange(len(objects))
 
    plt.bar(y_pos, mse, align='center', alpha=0.5)
    plt.xticks(y_pos, objects)
    plt.ylabel('Mean Square Error')
    plt.title('Mean Square Error of Regressor Models')

    plt.show()



    #Feature Importances of ExtraRegressor Model
    print "Feature Importances of ExtraRegressor Model"
    estimate = ensemble.ExtraTreesRegressor(bootstrap = True, n_estimators = 300, max_depth = 15).fit(X_dummytrain,y_dummytrain)
    predictions = estimate.predict(X_dummytest)
    print mean_squared_error(y_dummytest, predictions)

    print estimate.feature_importances_

    description = finaltraindf.columns.values

    obj = np.arange(len(description))

    objects = [str(numeric_string) for numeric_string in obj]

    y_pos = np.arange(len(objects))
 
    plt.bar(y_pos, estimate.feature_importances_)
    plt.xticks(y_pos, objects)
    plt.ylabel('Feature Importances')
    plt.title('Feature Importances of Extra Trees Regressor')

    plt.show()

    #Finding the best features for each model
    print "Finding the best features for each model"
    for clf in feature_regressors:
        lsvc = clf.fit(X_dummytrain,y_dummytrain)
        model = SelectFromModel(lsvc, prefit = True)

        print clf

        Train_new = model.transform(X_dummytrain)
        print X_dummytrain.shape
        print Train_new.shape
        newindices = model.get_support(True)
        print  cols[newindices]
        print "#######################################################"



    # Consider the case where the in domain Extension there are some noisy data containing the numeric values   
    print "Consider the case where the in domain Extension there are some noisy data containing the numeric values" 

    #Loading the Dataset
    train = pd.read_csv('assessment_challenge.csv')

    #Computed the Length of Dataset
    lengthTrain = len(train)

    cols = train.columns.values
    print lengthTrain

    print "Removing the noisy data"
    remove = train['Domain_extension'].convert_objects(convert_numeric=True)
    removedf = pd.DataFrame(remove, columns = ['Domain_extension'])

    indices = np.where(removedf['Domain_extension'].notnull())[0]

    train = train.drop(train.index[indices])

    train = train.reset_index(drop=True)


    ratings = train['read_rate'].tolist()

    train.drop('read_rate', axis=1, inplace=True)

    allcolumns = train.columns.values

    #Removing irrelevant attributes from dataset
    remove = ['id']
    finalcols1 = []

    for each in allcolumns:
        if each not in remove:
            finalcols1.append(each)

    #separating the string and numeric features
    stringfeature = ['from_domain_hash','Domain_extension','day']
    numericfeature = []

    for each in finalcols1:
        if each not in stringfeature:
            numericfeature.append(each)

    train[finalcols1].replace('N/A',np.nan,regex=True, inplace=True)

    numericTraindf = train[numericfeature].apply(pd.to_numeric, errors='coerce')

    stringTraindf = train[stringfeature]

    finalnumeric_df = DataFrameImputer().fit_transform(numericTraindf)


    finalstring_df = DataFrameImputer().fit_transform(stringTraindf)

    print "Encoding the labels"
    laEn = preprocessing.LabelEncoder()


    train_X = np.reshape(finalstring_df.values,(len(finalstring_df)*len(finalstring_df.columns.values))) 

    trainLabelEncode = laEn.fit_transform(train_X)


    Encodedtrain_X = np.reshape(trainLabelEncode, (len(finalstring_df),len(finalstring_df.columns.values))) 

    finalEncodedstring_df = pd.DataFrame(Encodedtrain_X, columns = stringfeature)

    finaltraindf = pd.concat([finalnumeric_df,finalEncodedstring_df], axis=1)



    rng = np.random.RandomState(1)

    X_dummyPrunedTrain, X_dummyPrunedTest, y_dummyPrunedTrain, y_dummyPrunedTest = train_test_split(finaltraindf.values, ratings, test_size=0.3, random_state=42)


    for clf in feature_regressors:
        estimate = clf.fit(X_dummyPrunedTrain,y_dummyPrunedTrain)
        predictions = estimate.predict(X_dummyPrunedTest)
        print clf
        print "Error after removing the noisy data" + str(mean_squared_error(y_dummyPrunedTest, predictions))
        print "#######################################################"


    #Finding the best tuned parameters for each model
    print "Finding the best tuned parameters for each Ensemble model"
    feature_Tunedregressors = [
    RandomizedSearchCV(ensemble.RandomForestRegressor(), cv= 10, param_distributions = {"bootstrap":[True,False],"max_depth":[10, 11, 12, 15], "n_estimators":[350, 362, 375, 387, 400]}),
    RandomizedSearchCV(ensemble.ExtraTreesRegressor(), cv= 10, param_distributions = {"bootstrap":[True,False],"max_depth":[10, 11, 12, 15], "n_estimators":[300, 350, 375, 387, 400]}),
    RandomizedSearchCV(ensemble.GradientBoostingRegressor(random_state = 0), cv= 10, param_distributions = {"alpha":[0.05, 0.1, 0.3, 0.6, 0.9],"learning_rate":[0.05, 0.1, 0.3, 0.6, 0.9],"max_depth":[10, 11, 12], "n_estimators":[300, 350,400, 450, 500, 600, 700]})]

    for clf in feature_Tunedregressors:
        estimate = clf.fit(X_dummytrain,y_dummytrain)
        predictions = estimate.predict(X_dummytest)

        print (clf.best_estimator_)

        print mean_squared_error(y_dummytest, predictions)
        print "#######################################################"
    
    

    End()
    



