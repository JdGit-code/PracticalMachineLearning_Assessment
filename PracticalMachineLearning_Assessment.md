Assessement for the Practical Machine Learning Course Project
=============================================================

Subject : Predict activity quality from activity monitors  
Date : August 31, 2016  
Author name : Julien D.

Overview
--------

#### Background

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now
possible to collect a large amount of data about personal activity
relatively inexpensively. These type of devices are part of the
quantified self movement - a group of enthusiasts who take measurements
about themselves regularly to improve their health, to find patterns in
their behavior, or because they are tech geeks. One thing that people
regularly do is quantify how much of a particular activity they do, but
they rarely quantify how well they do it. In this project, your goal
will be to use data from accelerometers on the belt, forearm, arm, and
dumbell of 6 participants. They were asked to perform barbell lifts
correctly and incorrectly in 5 different ways. More information is
available from the website here:
<http://groupware.les.inf.puc-rio.br/har> (see the section on the Weight
Lifting Exercise Dataset).

#### Data

The training data for this project are available here:
\[<https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv>\].

The test data are available here:
\[<https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv>\].

The data for this project come from this source:
\[<http://groupware.les.inf.puc-rio.br/har>\].

#### What we should submit

The goal of the project is to predict the manner in which they did the
exercise. This is the "classe" variable in the training set. We may use
any of the other variables to predict with. We should create a report
describing how we built your model, how we used cross validation, what
we think the expected out of sample error is, and why we made the
choices we did. You will also use your prediction model to predict 20
different test cases.

Data preparation
----------------

#### Packages required

First, we install (if necessary) and load packages required.

    inst_pkgs = load_pkgs =  c("caret","data.table","rpart.plot","rpart","randomForest","e1071")
    inst_pkgs = inst_pkgs[!(inst_pkgs %in% installed.packages()[,"Package"])]
    if(length(inst_pkgs)) install.packages(inst_pkgs)
    pkgs_loaded = lapply(load_pkgs, require, character.only=T)

#### Reading data and basic transformation

In this section :

1.  We have a validation sample where we don't know the target
    (variable "classe"). The model with the highest accuracy will be
    chosen as our final model and will be used on this dataset.
2.  We also have sample which we can use to build and optimize
    our models.

Some basic transformations and cleanup will be performed. Columns with
more than 95% of missing values will be deleted. Irrelevant remaining
columns such as "user\_name",
"raw\_timestamp\_part\_1","raw\_timestamp\_part\_2","cvtd\_timestamp"
(columns 1 to 5) will be removed in the subset.

We suppose the files pml-training.csv and pml-testing.csv are in the
DATAS repertory which is in the working directory. We also suppose this
R routine is launched in the working directory.

    # Read the files
    train <- fread("pml-training.csv",sep=",",dec=".", na.strings=c("NA","#DIV/0!",""))
    validation <- fread("pml-testing.csv",sep=",",dec=".", na.strings=c("NA","#DIV/0!",""))
    # Select columns which contain more than 95% of NA
    PercentNaTrain <- apply(train,2,function(x) sum(is.na(x))/length(x)*100)
    PercentNaValidation <- apply(validation,2,function(x) sum(is.na(x))/length(x)*100)
    index1 <- unique(which(as.vector(PercentNaTrain)>95 & as.vector(PercentNaValidation)>95))
    # Remove columns which contain more than 95% of NA and the irrelevant columns
    train <- train[,-c(index1,1:5),with=FALSE]
    validation <- validation[,-c(index1,1:5),with=FALSE]

#### Near zero-variance predictors

There are many models where predictors with a single unique value (also
know as "zero-variance predictors") or very few ("near zero-variance
predictors") will cause the model to fail. The fonction nearZeroVar can
be used to identify near zero-variance predictors in a dataset. So we
remove the column new\_window.

    nZV <- nearZeroVar(train,saveMetrics = TRUE)
    # Columns we have to remove
    rownames(nZV)[nZV$nzv==TRUE]

    ## [1] "new_window"

    train <- train[,-c("new_window"),with=FALSE]
    validation <- validation[,-c("new_window"),with=FALSE]

#### Multicollinearity problems

Also some models are susceptible to multicollinearity (high correlations
between predictors) which may affect interpretability of the model.
Predictors that result in absolute pairwise correlations greater than
0.95 can be removed usind the findCorrelation function. This function
returns an index of column numbers for removal.

    trainCorr <- cor(train[,-c("classe"),with=FALSE])
    highCorr <- findCorrelation(trainCorr,0.95)
    # Remove columns which are involved in high correlations
    train <- train[,-highCorr,with=FALSE]
    validation <- validation[,-highCorr,with=FALSE]

#### Creation of three samples : Training, Test and Validation samples

To evaluate the efficacy of the model we have to create an external test
sample not used in the training process. The train set will be split
into a training set and a test set. The test set will be used only to
evaluate performance (such as to compare models) and the training set
will be used for all other activities. The function createDataPartition
can be used to create stratidied random splits of a data set. In this
case, 75% of the data will be used for model training and the remainder
will be used for evaluating moel performance. The function creates the
random splits within each class so that the overall class distribution
is preserved as well as possible.

    set.seed(1)
    inTrain <- createDataPartition(y=train$classe,p=0.75,list = FALSE)
    DataTrain <- train[inTrain,]
    DataTest <- train[-inTrain,]
    DataValidation <- validation
    DataTest[,":="(classe=as.factor(classe))]
    DataTrain[,":="(classe=as.factor(classe))]
    # The overall class distribution is preserved as well as possible
    round(prop.table(table(DataTrain$classe))*100,3)

    ## 
    ##      A      B      C      D      E 
    ## 28.435 19.350 17.441 16.388 18.386

    round(prop.table(table(DataTest$classe))*100,3)

    ## 
    ##      A      B      C      D      E 
    ## 28.446 19.352 17.435 16.395 18.373

Building, tuning models and prediction of new samples
=====================================================

In this section a decision tree, a random forest and a support vector
machines will be applied to the data. \#\#\#\#\# Decision Tree

    set.seed(2)
    rpartFit <- rpart(classe~., data=DataTrain, method="class")
    rpartPredict <- predict(rpartFit,DataTest, type = "class")
    rpartcfM <- confusionMatrix(rpartPredict, DataTest$classe)$overall
    rpartcfM

    ##       Accuracy          Kappa  AccuracyLower  AccuracyUpper   AccuracyNull 
    ##   7.446982e-01   6.762986e-01   7.322492e-01   7.568553e-01   2.844617e-01 
    ## AccuracyPValue  McnemarPValue 
    ##   0.000000e+00   4.721021e-42

    table(data.frame(rpart.forecast=rpartPredict,real=DataTest$classe))

    ##               real
    ## rpart.forecast    A    B    C    D    E
    ##              A 1245  122   17   33   63
    ##              B   87  606   64  129   90
    ##              C   28  151  677  118   70
    ##              D   32   41   55  503   57
    ##              E    3   29   42   21  621

So, with this model we have a low accuracy. Indeed, the accuracy, which
is a description of sistematic errors, is about **74.5**% in this case.
We can see that plenty of events are not ranked properly.

##### Random Forest

    rfFit <- randomForest(classe~., data=DataTrain, method="class")
    rfPredict <- predict(rfFit,DataTest, type = "class")
    rfcfM <- confusionMatrix(rfPredict, DataTest$classe)$overall
    rfcfM

    ##       Accuracy          Kappa  AccuracyLower  AccuracyUpper   AccuracyNull 
    ##      0.9973491      0.9966467      0.9954712      0.9985878      0.2844617 
    ## AccuracyPValue  McnemarPValue 
    ##      0.0000000            NaN

    table(data.frame(rf.forecast=rfPredict, real=DataTest$classe))

    ##            real
    ## rf.forecast    A    B    C    D    E
    ##           A 1395    3    0    0    0
    ##           B    0  946    2    0    0
    ##           C    0    0  852    4    0
    ##           D    0    0    1  799    2
    ##           E    0    0    0    1  899

So, with this model we have a very high accuracy. Indeed, the accuracy,
which is a description of sistematic errors, is about **99.7**% in this
case. We can see that only a few events are not ranked properly.

##### Support Vector Machines

    svmFit <- svm(classe~., data=DataTrain)
    svmPredict <- predict(svmFit,DataTest)
    svmcfM <- confusionMatrix(svmPredict, DataTest$classe)$overall
    svmcfM

    ##       Accuracy          Kappa  AccuracyLower  AccuracyUpper   AccuracyNull 
    ##      0.9506525      0.9375377      0.9442149      0.9565466      0.2844617 
    ## AccuracyPValue  McnemarPValue 
    ##      0.0000000            NaN

    table(data.frame(svm.forecast=svmPredict, real=DataTest$classe))

    ##             real
    ## svm.forecast    A    B    C    D    E
    ##            A 1379   48    1    1    0
    ##            B    2  872   22    1    1
    ##            C   14   27  828   74   15
    ##            D    0    1    4  728   30
    ##            E    0    1    0    0  855

So, with this model we have a high accuracy. Indeed, the accuracy, which
is a description of sistematic errors, is about **95.1**% in this case.
We can see that only a few events are not ranked properly.

Submission
==========

The confusion matrices show that the random forest algorithm performens
better than decision trees and support vector machines. The accuracy for
the random forest model was **99.7**% compared to **74.5**% for decision
tree model and **95.1**% for support vector machines. The random forest
model is choosen. The expected out-of-sample error is estimated at
**0.3**% (1 - accuracy).

    predictfinal <- predict(rfFit, DataValidation, type="class")
    predictfinal

    ##  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 
    ##  B  A  B  A  A  E  D  B  A  A  B  C  B  A  E  E  A  B  B  B 
    ## Levels: A B C D E
