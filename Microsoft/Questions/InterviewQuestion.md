https://www.kdnuggets.com/2017/02/17-data-science-interview-questions-answers.html
https://www.kdnuggets.com/2017/02/17-data-science-interview-questions-answers-part-2.html
https://www.kdnuggets.com/2017/03/17-data-science-interview-questions-answers-part-3.html
- Model evaluation and choice of evaluation metrics used in accenture and with Dr. Zielke
- How would you deal with a categorical feature with high cardinality?
- Describe a machine learning project you're proud of
- Pick a machine learning algorithm of your choice and describe it to me
- Describe an ROC chart
- Describe a decision tree
- **Generate a fair coin from a biased one.**

This is originally von Neumann’s clever idea. If we have a biased coin (i.e. a coin that comes up heads with probability different from 1/2), we can simulate a fair coin by tossing pairs of coins until the two results are different. Given that we have different results, the probability that the first is “heads” and the second is “tails” is the same as the probability of “tails” then “heads”. So if we simply return the value of the first coin, we will get “heads” or “tails” with the same probability, i.e. 1/2.

```python
def fairCoin(biasedCoin):
   coin1, coin2 = 0,0
   while coin1 == coin2:
      coin1, coin2 = biasedCoin(), biasedCoin()
   return coin1

from random import random
def biasedCoin():
   return int(random() < 0.2)
# with high probability we will get a number that is close to 2000. I got 2058.   
sum(biasedCoin() for i in range(10000))   
# we should see a value that is approximately 5000. Indeed, when I tried it, I got 4982, 
which is evidence that fairCoin(biasedCoin) returns 1 with probability 1/2 (although I already gave a proof!)
sum(fairCoin(biasedCoin) for i in range(10000))

```

One might wonder how many calls to biasedCoin we expect to make before the function returns. One can recognize the experiment as a geometric distribution and use the known expected value, but it is short so here is a proof. Let s be the probability of seeing two different outcomes in the biased coin flip, and t the expected number of trials until that happens. If after two flips we see the same outcome (HH or TT), then by independence the expected number of flips we need is unchanged. Hence

t = 2s + (1-s)(2 + t)

Simplifying gives t = 2/s, and since we know s = 2p(1-p) we expect to flip the coin \frac{1}{p(1-p)} times.

8. Generate 7 integers with equal probability from a function which returns 1/0 with probability p and (1-p)
9. What are the ROC curve and the meaning of sensitivity, specificity, confusion matrix  
10. How to best select a representative sample of search quieres from 5 miliion serach queries
11. Experiment Design, A/B testing
12. Ask random-forest and lasso 
13. How to builds ads model, basic algorithms.  
14. Can you explain the Naive Bayes fundamentals? How did you set the threshold?
15. Can you explain what MapReduce is and how it works?  
16. Can you explain SVM to me?
17. Describe how gradient boost works
18. Data Mining Explain what heteroskedasticity is and how to solve it
19. Define and explain the differences between clustered and non-clustered indexes.


- **What are some ways I can make my model more robust to outliers?**

There are several ways to make a model more robust to outliers, from different points of view (data preparation or model building). **An outlier** in the question and answer is assumed being unwanted, unexpected, or a must-be-wrong value to the human’s knowledge so far (e.g. no one is 200 years old) rather than a rare event which is possible but rare.

Outliers are usually defined in relation to the distribution. Thus outliers could be removed in the pre-processing step (before any learning step), by using standard deviations (for normality) or interquartile ranges (for not normal/unknown) as threshold levels.

[![img](https://camo.githubusercontent.com/aa80872a0cedaf37753fb62db08b569bf0142d04/68747470733a2f2f7777772e6b646e7567676574732e636f6d2f77702d636f6e74656e742f75706c6f6164732f6f75746c696572732e6a7067)](https://camo.githubusercontent.com/aa80872a0cedaf37753fb62db08b569bf0142d04/68747470733a2f2f7777772e6b646e7567676574732e636f6d2f77702d636f6e74656e742f75706c6f6164732f6f75746c696572732e6a7067) **Outliers.** [Image source](https://www.neuraldesigner.com/blog/3_methods_to_deal_with_outliers)

Moreover, **data transformation** (e.g. log transformation) may help if data have a noticeable tail. When outliers related to the sensitivity of the collecting instrument which may not precisely record small values, **Winsorization** may be useful. This type of transformation (named after Charles P. Winsor (1895–1951)) has the same effect as clipping signals (i.e. replaces extreme data values with less extreme values). Another option to reduce the influence of outliers is using **mean absolute difference** rather mean squared error.

For model building, some models are resistant to outliers (e.g. [tree-based approaches](https://www.quora.com/Why-are-tree-based-models-robust-to-outliers)) or non-parametric tests. Similar to the median effect, tree models divide each node into two in each split. Thus, at each split, all data points in a bucket could be equally treated regardless of extreme values they may have. The study [Pham 2016] proposed a detection model that incorporates interquartile information of data to predict outliers of the data.

- **How do you detect if a new observation is outlier?**

  Some methods to screen outliers are z-scores, modified z-score, box plots, Grubb's test, Tietjen-Moore test exponential smoothing, Kimber test for exponential distribution and moving window filter algorithm. However two of the robust methods in detail are:

  **Inter Quartile Range**

  An outlier is a point of data that lies over 1.5 IQRs below the first quartile (Q1) or above third quartile (Q3) in a given data set.

  - High = (Q3) + 1.5 IQR
  - Low = (Q1) - 1.5 IQR

  **Tukey Method**

  It uses interquartile range to filter very large or very small numbers. It is practically the same method as above except that it uses the concept of "fences". The two values of fences are:

  - Low outliers = Q1 - 1.5(Q3 - Q1) = Q1 - 1.5(IQR)
  - High outliers = Q3 + 1.5(Q3 - Q1) = Q3 + 1.5(IQR)

  Anything outside of the fences is an outlier.

  When you find outliers, you should not remove it without a qualitative assessment because that way you are altering the data and making it no longer pure. It is important to understand the context of analysis or importantly "The Why question - Why an outlier is different from other data points?"

  This reason is critical. If outliers are attributed to error, you may throw it out but if they signify a new trend, pattern or reveal a valuable insight into the data you should retain it.

- **What is bias-variance trade off ?**


- **What are the different ways to return the row count of a table?**

  **Method 1:**

  *Query:*

  ```
  SELECT COUNT(*) FROM Transactions 
  ```

  *Comments:*

  Performs a full table scan. Slow on large tables.

  **Method 2:**

  *Query:*

  ```SQL
  SELECT CONVERT(bigint, rows) 
  FROM sysindexes 
  WHERE id = OBJECT_ID('Transactions') 
  AND indid < 2 
  ```

  *Comments:*

  Fast way to retrieve row count. Depends on statistics and is inaccurate.

  Run DBCC UPDATEUSAGE(Database) WITH COUNT_ROWS, which can take significant time for large tables.

  **Method 3:**

  *Query:*

  ```sql
  SELECT CAST(p.rows AS float) 
  FROM sys.tables AS tbl 
  INNER JOIN sys.indexes AS idx ON idx.object_id = tbl.object_id and
  idx.index_id < 2 
  INNER JOIN sys.partitions AS p ON p.object_id=CAST(tbl.object_id AS int) 
  AND p.index_id=idx.index_id 
  WHERE ((tbl.name=N'Transactions' 
  AND SCHEMA_NAME(tbl.schema_id)='dbo')) 
  ```

  *Comments:*

  The way the SQL management studio counts rows (look at table properties, storage, row count). Very fast, but still an approximate number of rows.

  **Method 4:**

  *Query:*

  ```sql
  SELECT SUM (row_count) 
  FROM sys.dm_db_partition_stats 
  WHERE object_id=OBJECT_ID('Transactions')    
  AND (index_id=0 or index_id=1); 
  ```

  *Comments:*

  Quick (although not as fast as method 2) operation and equally important, reliable.

- **How do you deal with unbalanced binary classification? / What error metric would you use to evaluate how good a binary classifier is? What if the classes are imbalanced? What if there are more than 2 groups?**

  Binary classification involves classifying the data into two groups, e.g. whether or not a customer buys a particular product or not (Yes/No), based on independent variables such as gender, age, location etc.

  As the target variable is not continuous, binary classification model predicts the probability of a target variable to be Yes/No. To evaluate such a model, a metric called the confusion matrix is used, also called the classification or co-incidence matrix. With the help of a confusion matrix, we can calculate important performance measures:

  - - 1. True Positive Rate (TPR) or Hit Rate or Recall or Sensitivity = TP / (TP + FN)
      2. False Positive Rate(FPR) or False Alarm Rate = 1 - Specificity = 1 - (TN / (TN + FP))
      3. Accuracy = (TP + TN) / (TP + TN + FP + FN)
      4. Error Rate = 1 – accuracy or (FP + FN) / (TP + TN + FP + FN)
      5. Precision = TP / (TP + FP)
      6. F-measure: 2 / ( (1 / Precision) + (1 / Recall) )
      7. ROC (Receiver Operating Characteristics) = plot of FPR vs TPR
      8. AUC (Area Under the Curve)
      9. Kappa statistics

  All of these measures should be used with domain skills and balanced, as, for example, if you only get a higher TPR in predicting patients who don’t have cancer, it will not help at all in diagnosing cancer.

  Conventional algorithms are often biased towards the majority class because their loss functions attempt to optimize quantities such as error rate, not taking the data distribution into consideration. In the worst case, minority examples are treated as outliers of the majority class and ignored. The learning algorithm simply generates a trivial classifier that classifies every example as the majority class.

  In the same example of cancer diagnosis data, if only 2% or less of the patients have cancer, then this would be a case of class imbalance, as the percentage of cancer patients is very small compared to rest of the population. There are main 2 approaches to handle this issue:

  - **Use of a cost function**: In this approach, a cost associated with misclassifying data is evaluated with the help of a cost matrix (similar to the confusion matrix, but more concerned with False Positives and False Negatives). The main aim is to reduce the cost of misclassifying. The cost of a False Negative is always more than the cost of a False Positive. e.g. wrongly predicting a cancer patient to be cancer-free is more dangerous than wrongly predicting a cancer-free patient to have cancer.

  ​       Total Cost = Cost of FN * Count of FN + Cost of FP * Count of FP

  - **Use of different sampling methods**: In this approach, you can use over-sampling, under-sampling, or hybrid sampling. In over-sampling, minority class observations are replicated to balance the data. Replication of observations leading to overfitting, causing good accuracy in training but less accuracy in unseen data. In under-sampling, the majority class observations are removed causing loss of information. It is helpful in reducing processing time and storage, but only useful if you have a large data set.

    - [SMOTE](https://www.cs.cmu.edu/afs/cs/project/jair/pub/volume16/chawla02a-html/chawla2002.html) (Synthetic Minority Oversampling TEchnique) Not resampling technique, The idea is to create new minority examples by interpolating between existing ones

    - Over- and undersampling selects examples randomly to adjust their proportions. Other approaches examine the instance space carefully and decide what to do based on their neighborhoods.

      For example, *Tomek links* are pairs of instances of opposite classes who are their own nearest neighbors. In other words, they are pairs of opposing instances that are very close together.

      [![Tomek links](http://www.svds.com/wp-content/uploads/2016/08/ImbalancedClasses_fig10-1024x559.png)](http://www.svds.com/wp-content/uploads/2016/08/ImbalancedClasses_fig10.png)

      Tomek’s algorithm looks for such pairs and removes the majority instance of the pair. The idea is to clarify the border between the minority and majority classes, making the minority region(s) more distinct.

  - **Do nothing.** Sometimes you get lucky and nothing needs to be done. You can train on the so-called *natural* (or *stratified*) distribution and sometimes it works without need for modification.

  - Throw away minority examples and switch to an anomaly detection framework.

  If there are multiple classes in the target variable, then a confusion matrix of dimensions equal to the number of classes is formed, and all performance measures can be calculated for each of the classes. This is called a multiclass confusion matrix. e.g. there are 3 classes X, Y, Z in the response variable, so recall for each class will be calculated as below:

  Recall_X = TP_X/(TP_X+FN_X)

  Recall_Y = TP_Y/(TP_Y+FN_Y)

  Recall_Z = TP_Z/(TP_Z+FN_Z)

  if you *need* a single-number metric, one of these is preferable to accuracy:

  1. [The Area Under the ROC curve (AUC)](https://en.wikipedia.org/wiki/Receiver_operating_characteristic#Area_under_the_curve) is a good general statistic. It is equal to the probability that a random positive example will be ranked above a random negative example.
  2. The [F1 Score](https://en.wikipedia.org/wiki/F1_score) is the harmonic mean of precision and recall. It is commonly used in text processing when an aggregate measure is sought.
  3. [Cohen’s Kappa](https://en.wikipedia.org/wiki/Cohen%27s_kappa) is an evaluation statistic that takes into account how much agreement would be expected by chance.

- **How do you measure distance between data points?**

https://dzone.com/articles/machine-learning-measuring

1. ​
2. What’s the difference between L1 and L2 regularization?
3. Can you explain what REST is?
4. What are the steps for wrangling and cleaning data before applying machine learning algorithms?
5. ​
6. Define variance.
7. Describe the differences between and use cases for box plots and histograms


1. What is a neural network?


