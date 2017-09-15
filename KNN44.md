---
title: "Pattern recognition in breast cancer data with K-nearest neighbors algorithm"
output: pdf_document
---



                                  
                                **Author : Abas-Hamza**  
                                  
                                  
  
  
  
  
  
* **Abstract .................................................   I**
* **Getting and cleaning the data.............................   II**
* **Exploration Analysis......................................   III**
* **Model Building ..........................................  IV** 
* **Predicting ...............................................    V**  
* **Evaluating model accuracy............................... VI**  

                                  






----------------------------------------------------------------------------------






* Abstract

In the previous project I've applied Quadratic discriminant analysis which is quiet useful when the decision boundaries are not highly linear. In this project I am going to apply K-nearest neighbors algorithm **(KNN)**. KNN is a non-parametric classification algorithm that makes predictions using the training dataset without learning form the training set. KNN makes no assumptions about the functional form of data . Thus KNN is referred to as a non-parametric machine learning algorithm. k nearest neighbor are defined by their characteristic of classifying unlabeled observations by assigning them the class of similar labeled observations. Traditionally It uses Euclidean distance to find out the nearest neighbour.  Euclidean Distance is the ordinary straight line distance between two points in Euclidean Space. there are other popular distances such as Manhattan distance and Minkowski Distance . We often choose the best distance metric based on the properties of data. However the Euclidean distance is widely used. Euclidean distance is specified by the following formula, where p and q are data points to be compared:


```r
knitr::include_graphics("figure/eucli.png")
```

![plot of chunk unnamed-chunk-1](figure/eucli.png)

Choosing the value for K doesn't have any limitation, but it is always a good practice to try many different values for K. By definition k is The decision of how many neighbors to be used for Clasification. Therefore Choosing an appropriate k determines the performance of the model. The other thing that I want to mention is that KNN works well with a small number of features , but struggles when **p** is very large. It is highly necessarily to Normalize the data within a range **ex: [0, 1]** this will avoid Wrong distances when calculatig the **Euclidean distances** . It may also be a good idea to standardize your data. Because KNN performs much better if all of the data has the same scale.  The same rescaling method used on the k-NN training dataset must also be applied to the testset. The formula for normalizing is:



```r
knitr::include_graphics("figure/normalize2.png")
```

![plot of chunk unnamed-chunk-2](figure/normalize2.png)


In the next section I will Build a pattern recognition model for detecting the type of breast cancer by appling k-NN algorithm.  Machine learning can automate the identification of cancerous cells, it improves the efficiency of the detection process, allowing physicians to spend less time diagnosing and more time treating the disease, this is a huge breakthrough in the field of computer science allowing us to use An automated screening system which provide greater detection accuracy. 

The data that I am going to use is Wisconsin Breast Cancer Diagnostic dataset from the UCI Machine Learning Repository at **http://archive.ics.uci.edu/ml** .This data was donated by researchers of the University of Wisconsin and includes the measurements from digitized images of fine-needle aspirate of a breast mass. The data includes 569 observations of cancer biopsies, each with  32 features. The target feature is the cancer diagnosis coded as  "M" to indicate malignant and "B" to indicate benign. The data has been split into two groups: training set (train.csv) and test set (test.csv). The training set should be used to build our machine learning model and The test set should be used to see how well our model performs on unseen data. 



# Getting and cleaning the data




```r
# Getting data and summarizing

maindata <- read.csv("breast_Cancer_data.csv", header = TRUE, stringsAsFactors = FALSE)
dim(maindata)
```

```
## [1] 569  33
```

```r
summary(maindata)
```

```
##        id             diagnosis          radius_mean      texture_mean  
##  Min.   :     8670   Length:569         Min.   : 6.981   Min.   : 9.71  
##  1st Qu.:   869218   Class :character   1st Qu.:11.700   1st Qu.:16.17  
##  Median :   906024   Mode  :character   Median :13.370   Median :18.84  
##  Mean   : 30371831                      Mean   :14.127   Mean   :19.29  
##  3rd Qu.:  8813129                      3rd Qu.:15.780   3rd Qu.:21.80  
##  Max.   :911320502                      Max.   :28.110   Max.   :39.28  
##  perimeter_mean     area_mean      smoothness_mean   compactness_mean 
##  Min.   : 43.79   Min.   : 143.5   Min.   :0.05263   Min.   :0.01938  
##  1st Qu.: 75.17   1st Qu.: 420.3   1st Qu.:0.08637   1st Qu.:0.06492  
##  Median : 86.24   Median : 551.1   Median :0.09587   Median :0.09263  
##  Mean   : 91.97   Mean   : 654.9   Mean   :0.09636   Mean   :0.10434  
##  3rd Qu.:104.10   3rd Qu.: 782.7   3rd Qu.:0.10530   3rd Qu.:0.13040  
##  Max.   :188.50   Max.   :2501.0   Max.   :0.16340   Max.   :0.34540  
##  concavity_mean    concave.points_mean symmetry_mean   
##  Min.   :0.00000   Min.   :0.00000     Min.   :0.1060  
##  1st Qu.:0.02956   1st Qu.:0.02031     1st Qu.:0.1619  
##  Median :0.06154   Median :0.03350     Median :0.1792  
##  Mean   :0.08880   Mean   :0.04892     Mean   :0.1812  
##  3rd Qu.:0.13070   3rd Qu.:0.07400     3rd Qu.:0.1957  
##  Max.   :0.42680   Max.   :0.20120     Max.   :0.3040  
##  fractal_dimension_mean   radius_se        texture_se      perimeter_se   
##  Min.   :0.04996        Min.   :0.1115   Min.   :0.3602   Min.   : 0.757  
##  1st Qu.:0.05770        1st Qu.:0.2324   1st Qu.:0.8339   1st Qu.: 1.606  
##  Median :0.06154        Median :0.3242   Median :1.1080   Median : 2.287  
##  Mean   :0.06280        Mean   :0.4052   Mean   :1.2169   Mean   : 2.866  
##  3rd Qu.:0.06612        3rd Qu.:0.4789   3rd Qu.:1.4740   3rd Qu.: 3.357  
##  Max.   :0.09744        Max.   :2.8730   Max.   :4.8850   Max.   :21.980  
##     area_se        smoothness_se      compactness_se      concavity_se    
##  Min.   :  6.802   Min.   :0.001713   Min.   :0.002252   Min.   :0.00000  
##  1st Qu.: 17.850   1st Qu.:0.005169   1st Qu.:0.013080   1st Qu.:0.01509  
##  Median : 24.530   Median :0.006380   Median :0.020450   Median :0.02589  
##  Mean   : 40.337   Mean   :0.007041   Mean   :0.025478   Mean   :0.03189  
##  3rd Qu.: 45.190   3rd Qu.:0.008146   3rd Qu.:0.032450   3rd Qu.:0.04205  
##  Max.   :542.200   Max.   :0.031130   Max.   :0.135400   Max.   :0.39600  
##  concave.points_se   symmetry_se       fractal_dimension_se
##  Min.   :0.000000   Min.   :0.007882   Min.   :0.0008948   
##  1st Qu.:0.007638   1st Qu.:0.015160   1st Qu.:0.0022480   
##  Median :0.010930   Median :0.018730   Median :0.0031870   
##  Mean   :0.011796   Mean   :0.020542   Mean   :0.0037949   
##  3rd Qu.:0.014710   3rd Qu.:0.023480   3rd Qu.:0.0045580   
##  Max.   :0.052790   Max.   :0.078950   Max.   :0.0298400   
##   radius_worst   texture_worst   perimeter_worst    area_worst    
##  Min.   : 7.93   Min.   :12.02   Min.   : 50.41   Min.   : 185.2  
##  1st Qu.:13.01   1st Qu.:21.08   1st Qu.: 84.11   1st Qu.: 515.3  
##  Median :14.97   Median :25.41   Median : 97.66   Median : 686.5  
##  Mean   :16.27   Mean   :25.68   Mean   :107.26   Mean   : 880.6  
##  3rd Qu.:18.79   3rd Qu.:29.72   3rd Qu.:125.40   3rd Qu.:1084.0  
##  Max.   :36.04   Max.   :49.54   Max.   :251.20   Max.   :4254.0  
##  smoothness_worst  compactness_worst concavity_worst  concave.points_worst
##  Min.   :0.07117   Min.   :0.02729   Min.   :0.0000   Min.   :0.00000     
##  1st Qu.:0.11660   1st Qu.:0.14720   1st Qu.:0.1145   1st Qu.:0.06493     
##  Median :0.13130   Median :0.21190   Median :0.2267   Median :0.09993     
##  Mean   :0.13237   Mean   :0.25427   Mean   :0.2722   Mean   :0.11461     
##  3rd Qu.:0.14600   3rd Qu.:0.33910   3rd Qu.:0.3829   3rd Qu.:0.16140     
##  Max.   :0.22260   Max.   :1.05800   Max.   :1.2520   Max.   :0.29100     
##  symmetry_worst   fractal_dimension_worst    X          
##  Min.   :0.1565   Min.   :0.05504         Mode:logical  
##  1st Qu.:0.2504   1st Qu.:0.07146         NA's:569      
##  Median :0.2822   Median :0.08004                       
##  Mean   :0.2901   Mean   :0.08395                       
##  3rd Qu.:0.3179   3rd Qu.:0.09208                       
##  Max.   :0.6638   Max.   :0.20750
```

```r
str(maindata)
```

```
## 'data.frame':	569 obs. of  33 variables:
##  $ id                     : int  842302 842517 84300903 84348301 84358402 843786 844359 84458202 844981 84501001 ...
##  $ diagnosis              : chr  "M" "M" "M" "M" ...
##  $ radius_mean            : num  18 20.6 19.7 11.4 20.3 ...
##  $ texture_mean           : num  10.4 17.8 21.2 20.4 14.3 ...
##  $ perimeter_mean         : num  122.8 132.9 130 77.6 135.1 ...
##  $ area_mean              : num  1001 1326 1203 386 1297 ...
##  $ smoothness_mean        : num  0.1184 0.0847 0.1096 0.1425 0.1003 ...
##  $ compactness_mean       : num  0.2776 0.0786 0.1599 0.2839 0.1328 ...
##  $ concavity_mean         : num  0.3001 0.0869 0.1974 0.2414 0.198 ...
##  $ concave.points_mean    : num  0.1471 0.0702 0.1279 0.1052 0.1043 ...
##  $ symmetry_mean          : num  0.242 0.181 0.207 0.26 0.181 ...
##  $ fractal_dimension_mean : num  0.0787 0.0567 0.06 0.0974 0.0588 ...
##  $ radius_se              : num  1.095 0.543 0.746 0.496 0.757 ...
##  $ texture_se             : num  0.905 0.734 0.787 1.156 0.781 ...
##  $ perimeter_se           : num  8.59 3.4 4.58 3.44 5.44 ...
##  $ area_se                : num  153.4 74.1 94 27.2 94.4 ...
##  $ smoothness_se          : num  0.0064 0.00522 0.00615 0.00911 0.01149 ...
##  $ compactness_se         : num  0.049 0.0131 0.0401 0.0746 0.0246 ...
##  $ concavity_se           : num  0.0537 0.0186 0.0383 0.0566 0.0569 ...
##  $ concave.points_se      : num  0.0159 0.0134 0.0206 0.0187 0.0188 ...
##  $ symmetry_se            : num  0.03 0.0139 0.0225 0.0596 0.0176 ...
##  $ fractal_dimension_se   : num  0.00619 0.00353 0.00457 0.00921 0.00511 ...
##  $ radius_worst           : num  25.4 25 23.6 14.9 22.5 ...
##  $ texture_worst          : num  17.3 23.4 25.5 26.5 16.7 ...
##  $ perimeter_worst        : num  184.6 158.8 152.5 98.9 152.2 ...
##  $ area_worst             : num  2019 1956 1709 568 1575 ...
##  $ smoothness_worst       : num  0.162 0.124 0.144 0.21 0.137 ...
##  $ compactness_worst      : num  0.666 0.187 0.424 0.866 0.205 ...
##  $ concavity_worst        : num  0.712 0.242 0.45 0.687 0.4 ...
##  $ concave.points_worst   : num  0.265 0.186 0.243 0.258 0.163 ...
##  $ symmetry_worst         : num  0.46 0.275 0.361 0.664 0.236 ...
##  $ fractal_dimension_worst: num  0.1189 0.089 0.0876 0.173 0.0768 ...
##  $ X                      : logi  NA NA NA NA NA NA ...
```


The datset has 569 observations and 33 variables, the target variable That we are interested in is **Diagnosis** coded as "M" to indicate malignant and "B" to indicate benign. This data does not have missing values,but We need to drop the ID col and x col. ID variables should always be excluded otherwise it can lead to erroneous analysis.  



```r
# Getting NA'S
colSums(is.na(maindata))
```

```
##                      id               diagnosis             radius_mean 
##                       0                       0                       0 
##            texture_mean          perimeter_mean               area_mean 
##                       0                       0                       0 
##         smoothness_mean        compactness_mean          concavity_mean 
##                       0                       0                       0 
##     concave.points_mean           symmetry_mean  fractal_dimension_mean 
##                       0                       0                       0 
##               radius_se              texture_se            perimeter_se 
##                       0                       0                       0 
##                 area_se           smoothness_se          compactness_se 
##                       0                       0                       0 
##            concavity_se       concave.points_se             symmetry_se 
##                       0                       0                       0 
##    fractal_dimension_se            radius_worst           texture_worst 
##                       0                       0                       0 
##         perimeter_worst              area_worst        smoothness_worst 
##                       0                       0                       0 
##       compactness_worst         concavity_worst    concave.points_worst 
##                       0                       0                       0 
##          symmetry_worst fractal_dimension_worst                       X 
##                       0                       0                     569
```

```r
maindata <- maindata[-c(1,33)]
```

##  Normalizing Numerical variables


It is important to normalize all numerical variables because the distance calculation for **k-nn** is dependent on the measurement scale of the predictor variables. 



```r
# make a function whcih normalize the variable
normalizefunc <- function(x) {
        norma <- (x-min(x)) / (max(x) - min(x)) 
        return(norma)
}

normaindata <- as.data.frame(lapply(maindata[2:31], normalizefunc))
```


Now We need to understand more about how the target variable is distributed. The target variable  indicates whether the type of tumour is benign  or malignant. 



```r
 table(maindata$diagnosis)
```

```
## 
##   B   M 
## 357 212
```

```r
round(prop.table(table(maindata$diagnosis))* 100, digits = 1)
```

```
## 
##    B    M 
## 62.7 37.3
```

The percent of Bening is higher than malignant. to get more intuition about the target variable let's visualize it.

#  Exploration Analysis


```r
 plot1 <-  ggplot(maindata, aes(x=diagnosis, fill=factor(diagnosis))) + geom_bar() + theme(axis.text.x = element_blank())+
scale_fill_brewer(palette = "Set1") + theme_bw()

plot1
```

![plot of chunk unnamed-chunk-8](figure/unnamed-chunk-8-1.png)


# Model Building


 To understand how well our learner performs on unseen dataset we   divide our data into two portions a training dataset and testset.
The training set should be used to build our machine learning model and The test set should be used to see how well our model performs on unseen data. We will now perform KNN using the knn() function in class library.  knn() function forms predictions using a single command. The function requires four inputs:

* A matrix containing the predictors associated with the training data
* A matrix containing the predictors associated with the test data.
* A factor vector containing the class labels for the training data
* An integer for K, indicating he number of nearest neighbors to be used by the classfier.


### Data preparation 


```r
maindata$diagnosis <- as.factor(maindata$diagnosis)

trainset <- normaindata[1:450,]
testset <- normaindata[451:569,]
trainclassVariable <- maindata[1:450, 1]
testclassVariable <- maindata[451:569, 1]
kvalue <- round(sqrt(nrow(trainset)))
```


Now everything is ready, We've divided the data into training and test, and We've prepared all 4 inputs that we needed to apply for K-NN algorithm. Choosing the value for K doesn't have any limitation, but it is always a good practice to try many diferent values for K. We can try K= 3 or k = square root of n (numbers of the training set). 
Now the knn() function can be used to predict the type of cancer.We set a random seed before we apply knn() to ensure reproducibility of results otherwise R will randomly break the tie. 


```r
set.seed(1)
kvalue <- round(sqrt(nrow(trainset)))
predictionClass <- knn(train =trainset, test = testset, cl=trainclassVariable, k= kvalue)
```

# Evaluating model accuracy

After We build our model We need a method to evaluate the overall accuracy of the model. This allows us to determine if we have a good or bad model. There are number of ways to determine the accuracy of the model that involves classification. for instance We use the confusion matrix which summarizes the performance of classification. We know that **K-NN** is a lazy learner meaning that it does'nt learn from the trainingset but it actually makes classes and at the same time predict the outcome. Binary classifiers such as **K-NN, Logistic Regression, Random forest etc...**  make two types of error the False positive and the False negative.  In this case False positive means predict an individual's type of tumours is **Malignant**  when in fact it is not. And False negative means predict an individual's type of tumours is **Benigne**, when in fact it is not. We will use **Matrix-confusion** and **ROC** to summarize the performance of the Model. 



```r
confusionMatrix(testclassVariable,predictionClass)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction  B  M
##          B 92  0
##          M  2 25
##                                          
##                Accuracy : 0.9832         
##                  95% CI : (0.9406, 0.998)
##     No Information Rate : 0.7899         
##     P-Value [Acc > NIR] : 3.434e-10      
##                                          
##                   Kappa : 0.9508         
##  Mcnemar's Test P-Value : 0.4795         
##                                          
##             Sensitivity : 0.9787         
##             Specificity : 1.0000         
##          Pos Pred Value : 1.0000         
##          Neg Pred Value : 0.9259         
##              Prevalence : 0.7899         
##          Detection Rate : 0.7731         
##    Detection Prevalence : 0.7731         
##       Balanced Accuracy : 0.9894         
##                                          
##        'Positive' Class : B              
## 
```

We have got **98%** of accuarcy, that is quite good, only 2 percent of masses were incorrectly classified by the algorithm. I've tried  various k values to analyse the performance , it turns out that there is no value for K which improves the prediction other than k=21 and k=8. 


```r
knnfunc <- function(kvalues){
        set.seed(1)
        predictionClass <- knn(train =trainset, test = testset, cl=trainclassVariable, k= kvalues)
        accuracy <- mean(testclassVariable == predictionClass)
        return(round(accuracy * 100))
}
knnfunc(8);knnfunc(5);knnfunc(3);knnfunc(9);knnfunc(11)
```

```
## [1] 98
```

```
## [1] 97
```

```
## [1] 95
```

```
## [1] 97
```

```
## [1] 97
```


# Conclusion

In this project I applied **k-NN** algorithm. **KNN** is a non-parametric classification algorithm that makes predictions using the training dataset without learning form the training set. **KNN** makes no assumptions about the functional form of data. That said, **KNN** remains robust algorithm, We've got 98% of accuracy which is pretty good. This non parametric approach is widely used when the decision boundary is not highly linear. Apparently Machine learning can automate the identification of cancerous cells, it improves the efficiency of the detection process, allowing physicians to spend less time diagnosing and more time treating the disease. this is really interesting.  I hope that this article covers the practical approach of  **K-NN** algorithm. 

