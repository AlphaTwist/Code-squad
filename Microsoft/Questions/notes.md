### Q1. What are Data Science lessons from failure to predict 2016 US Presidential election (and from Super Bowl LI comeback)

**Gregory Piatetsky answers:**

![nytimes-upshot-forecast-trump-15](https://www.kdnuggets.com/wp-content/uploads/nytimes-upshot-forecast-trump-15.jpg)
Just before the Nov 8, 2016 election, most pollsters gave Hillary Clinton an edge of ~3% in popular vote and 70-95% chance of victory in electoral college. Nate Silver's FiveThirtyEight had the highest chances of Trump Victory at ~30%, while New York Times Upshot and Princeton Election Consortium estimated only ~15%, and other pollsters like Huffington Post gave Trump only 2% chance of victory. Still, Trump won. So what are the lessons for Data Scientists?

To make a statistically valid prediction we need

1) enough historical data and

2) assumption that past events are sufficiently similar to current event we are trying to predict.

Events can placed on the scale from deterministic (2+2 will always equal to 4) to strongly predictable (e.g. orbits of planets and moons, avg. number of heads when tossing a fair coin) to weakly predictable (e.g. elections and sporting events) to random (e.g. honest lottery).

If we toss a fair coin 100 million times, we have the expected number of heads (mean) as 50 million, the standard deviation =10,000 (using formula 0.5 * SQRT(N)), and we can predict that 99.7% of the time the expected number of heads will be within [3 standard deviations](https://en.wikipedia.org/wiki/68%E2%80%9395%E2%80%9399.7_rule) of the mean.

But using polling to predict the votes of 100 million people is much more difficult. Pollsters need to get a representative sample, estimate the likelihood of a person actually voting, make many justified and unjustified assumptions, and avoid following their conscious and unconscious biases.

In the case of US Presidential election, correct prediction is even more difficult because of the antiquated Electoral college system when each state (except for Maine and Nebraska) awards the winner all its votes in the electoral college, and the need to poll and predict results for each state separately.

The chart below shows that in 2016 US presidential elections pollsters were off the mark in many states. They mostly underestimated the Trump vote, especially in 3 critical states of Michigan, Wisconsin, and Pennsylvania which all flipped to Trump.

![Us Elections 2016 Poll Shift, according to 538](https://www.kdnuggets.com/images/us-elections-2016-poll-shift-538.jpg)

Source: [**@NateSilver538**](http://twitter.com/NateSilver538) tweet, Nov 9, 2016.

A few statisticians like Salil Mehta [@salilstatistics](https://twitter.com/salilstatistics) were warning about [unreliability of polls](https://twitter.com/salilstatistics/status/796248050851139584), and David Wasserman of 538 actually described this scenario in Sep 2016 [How Trump Could Win The White House While Losing The Popular Vote](http://fivethirtyeight.com/features/how-trump-could-win-the-white-house-while-losing-the-popular-vote/?ex_cid=story-twitter), but most pollsters were way off.

So a good lesson for Data Scientists is to **question their assumptions** and to be very skeptical when predicting a weakly predictable event, especially when based on human behavior.

Other important lessons are

- - - Examine data quality - in this election polls were not reaching all likely voters
    - Beware of your own biases: many pollsters were likely Clinton supporters and did not want to question the results that favored their candidate. For example, Huffington Post had forecast over 95% chance of Clinton Victory.

See also other analyses of 2016 polling failures:

- - - Wired: [Trump’s Win Isn’t the Death of Data—It Was Flawed All Along](https://www.wired.com/2016/11/trumps-win-isnt-death-data-flawed-along).
    - NYTimes [How Data Failed Us in Calling an Election](http://www.nytimes.com/2016/11/10/technology/the-data-said-clinton-would-win-why-you-shouldnt-have-believed-it.html)
    - Datanami [Six Data Science Lessons from the Epic Polling Failure](https://www.datanami.com/2016/11/11/data-science-lessons-epic-polling-failure/)
    - InformationWeek [Trump's Election: Poll Failures Hold Data Lessons For IT](http://www.informationweek.com/big-data/big-data-analytics/trumps-election-poll-failures-hold-data-lessons-for-it/d/d-id/1327455)
    - [Why I Had to Eat a Bug on CNN](http://www.nytimes.com/2016/11/19/opinion/why-i-had-to-eat-a-bug-on-cnn.html?emc=eta1&_r=0), by Sam Wang, Princeton, whose Princeton Election Consortium gave Trump 15% to win.

(*Note: this answer is based on a previous KDnuggets post: /2016/11/trump-shows-limits-prediction.html)*

We had another example of statistically very unlikely event happen in Super Bowl LI on Feb 5, 2017.  After the half time, Atlanta Falcons were leading 21:3 after halftime and 28:9 after 3rd quarter. ESPN estimated Falcons win probability at that time at almost 100%.

![Super Bowl 2017 win probability](https://www.kdnuggets.com/wp-content/uploads/superbowl-2017-win-probability.jpg)

(reference: Salil Mehta tweet [Salil Mehta tweet, Feb 6, 2017](https://twitter.com/salilstatistics/status/828581183558713344))

Never before has a team lost a Super Bowl after holding such advantage.  However, each Super Bowl is different, and this one was turned out to be very different.  Combination of superior skill (Patriots, after all, were favorites before the game) and luck (e.g. a very lucky catch by Julian Edelman in 4th quarter, Patriots winning coin toss in overtime) gave victory to Pats.

This Super Bowl was another good lesson for Data Scientists of danger of having **too much confidence**when predicting weakly predictable events. You need to understand the risk factors when dealing with such events, and try to avoid using probabilities, or if you have to use numbers, have a wide confidence range.

Finally, if the odds seem to be against you but the event is only weakly predictable, go ahead and do your best - sometimes you will be able to beat the odds.

------

 

### Q2. What problems arise if the distribution of the new (unseen) test data is significantly different than the distribution of the training data?

**Gregory Piatetsky and Thuy Pham answer:**

The main problem is that the predictions will be wrong !

If the new test data is sufficiently different in key parameters of the prediction model from the training data, then predictive model is no longer valid.

The main reasons this can happen are sample selection bias, population drift, or non-stationary environment.

**a) Sample selection bias**
Here the data is static, but the training examples have been obtained through a biased method, such as non-uniform selection or non-random split of data into train and test.

If you have a large static dataset, then you should randomly split it into train/test data, and the distribution of test data should be similar to training data.

![Covariate shift](https://www.kdnuggets.com/wp-content/uploads/training-function-shift.jpg)
**b) Covariate shift aka population drift**
Here the data is not static, with one population used as a training data, and another population used for testing.
*(Figure from http://iwann.ugr.es/2011/pdf/InvitedTalk-FHerrera-IWANN11.pdf).*

Sometimes the training data and test data are derived via different processes - eg a drug tested on one population is given to a new population that may have significant differences. As a result, a classifier based on training data will perform poorly.

One proposed solution is to apply a statistical test to decide if the probabilities of target classes and key variables used by the classifier are significantly different, and if they are, to retrain the model using new data.

**c) Non-stationary environments**
Training environment is different from the test one, whether it's due to a temporal or a spatial change.

This is similar to case b, but applies to situation when data is not static -  we have a stream of data and we periodically sample it to develop predictive models of future behavior.  This happens in adversarial classification problems, such as spam filtering and network intrusion detection, where spammers and hackers constantly change their behavior in response. Another typical case is customer analytics where customer behavior changes over time.  A telephone company develops a model for predicting customer churn or a credit card company develops a model to predict transaction fraud.  Training data is historical data, while (new) test data is the current data.

Such models periodically need to be retrained and to determine when you can compare the distribution of key variables in the predictive model in the old data (training set) and the new data, and if there is a sufficiently significant difference, the model needs to be retrained.

For a more detailed and technical discussion, see references below.

**References:**

[1] Marco Saerens, Patrice Latinne, Christine Decaestecker: Adjusting the Outputs of a Classifier to New a Priori Probabilities: A Simple Procedure. Neural Computation 14(1): 21-41 (2002)

[2] Machine Learning in Non-stationary Environments: Introduction to Covariate Shift Adaptation, Masashi Sugiyama, Motoaki Kawanabe, MIT Press, 2012, ISBN 0262017091, 9780262017091

[3] Quora answer to [What could be some issues if the distribution of the test data is significantly different than the distribution of the training data?](https://www.quora.com/What-could-be-some-issues-if-the-distribution-of-the-test-data-is-significantly-different-than-the-distribution-of-the-training-data)

[4] [Dataset Shift in Classification: Approaches and Problems](http://iwann.ugr.es/2011/pdf/InvitedTalk-FHerrera-IWANN11.pdf), Francisco Herrera invited talk, 2011.

[5] [When Training and Test Sets are Different: Characterising Learning Transfer](http://homepages.inf.ed.ac.uk/amos/publications/Storkey2009TrainingTestDifferent.pdf), Amos Storkey, 2013.

------

 

### Q3. What are bias and variance, and what are their relation to modeling data?

**Matthew Mayo answers:**

**Bias** is how far removed a model's predictions are from correctness, while **variance** is the degree to which these predictions vary between model iterations.

![Bias vs Variance](https://www.kdnuggets.com/wp-content/uploads/bias-and-variance.jpg)

**Bias vs Variance**, [Image source](http://scott.fortmann-roe.com/docs/BiasVariance.html)

[As an example](https://www.kdnuggets.com/2016/08/bias-variance-tradeoff-overview.html), using a simple flawed Presidential election survey as an example, errors in the survey are then explained through the twin lenses of bias and variance: selecting survey participants from a phonebook is a source of bias; a small sample size is a source of variance.

Minimizing total model error relies on the balancing of bias and variance errors. Ideally, models are the result of a collection of **unbiased data of low variance**. Unfortunately, however, the more complex a model becomes, its tendency is toward less bias but greater variance; therefore an optimal model would need to consider a balance between these 2 properties.

The statistical evaluation method of cross-validation is useful in both demonstrating the **importance** of this balance, as well as actually **searching** it out. The number of data folds to use -- the value of *k* in *k*-fold cross-validation -- is an important decision; the lower the value, the higher the bias in the error estimates and the less variance.

![Bias variance total error](https://www.kdnuggets.com/wp-content/uploads/bias-variance-total-error.jpg)

Bias and variance contributing to total error

, 

Image source

Conversely, when  k is set equal to the number of instances, the error estimate is then very low in bias but has the possibility of high variance.

The most important takeaways are that bias and variance are two sides of an important trade-off when building models, and that even the most routine of statistical evaluation methods are directly reliant upon such a trade-off.



### Q4. Why might it be preferable to include fewer predictors over many?

**Anmol Rajpurohit answers:**

Here are a few reasons why it might be a better idea to have fewer predictor variables rather than having many of them:

**Redundancy/Irrelevance:**

If you are dealing with many predictor variables, then the chances are high that there are hidden relationships between some of them, leading to redundancy. Unless you identify and handle this redundancy (by selecting only the non-redundant predictor variables) in the early phase of data analysis, it can be a huge drag on your succeeding steps.

It is also likely that not all predictor variables are having a considerable impact on the dependent variable(s). You should make sure that the set of predictor variables you select to work on does not have any irrelevant ones – even if you know that data model will take care of them by giving them lower significance.

*Note: Redundancy and Irrelevance are two different notions –a relevant feature can be redundant due to the presence of other relevant feature(s).*

**Overfitting**:

Even when you have a large number of predictor variables with no relationships between any of them, it would still be preferred to work with fewer predictors. The data models with large number of predictors (also referred to as complex models) often suffer from the problem of overfitting, in which case the data model performs great on training data, but performs poorly on test data.

**Productivity**:

Let’s say you have a project where there are a large number of predictors and all of them are relevant (i.e. have measurable impact on the dependent variable). So, you would obviously want to work with all of them in order to have a data model with very high success rate. While this approach may sound very enticing, practical considerations (such of amount of data available, storage and compute resources, time taken for completion, etc.) make it nearly impossible.

Thus, even when you have a large number of relevant predictor variables, it is a good idea to work with fewer predictors (shortlisted through feature selection or developed through feature extraction). This is essentially similar to the Pareto principle, which states that for many events, roughly 80% of the effects come from 20% of the causes.

Focusing on those 20% most significant predictor variables will be of great help in building data models with considerable success rate in a reasonable time, without needing non-practical amount of data or other resources.

![img](https://www.kdnuggets.com/wp-content/uploads/training-vs-complexity.jpg)
*Training error & test error vs model complexity (Source: Posted on Quora by Sergul Aydore)*

**Understandability**:

Models with fewer predictors are way easier to understand and explain. As the data science steps will be performed by humans and the results will be presented (and hopefully, used) by humans, it is important to consider the comprehensive ability of human brain. This is basically a trade-off – you are letting go of some potential benefits to your data model’s success rate, while simultaneously making your data model easier to understand and optimize.

This factor is particularly important if at the end of your project you need to present your results to someone, who is interested in not just high success rate, but also in understanding what is happening “under the hood”.

------

 

### Q5. What error metric would you use to evaluate how good a binary classifier is? What if the classes are imbalanced? What if there are more than 2 groups?

[**Prasad Pore**](https://www.kdnuggets.com/author/prasad-pore)** answers:**

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

You can find more details about these measures here: [The Best Metric to Measure Accuracy of Classification Models](https://www.kdnuggets.com/2016/12/best-metric-measure-accuracy-classification-models.html).

All of these measures should be used with domain skills and balanced, as, for example, if you only get a higher TPR in predicting patients who don’t have cancer, it will not help at all in diagnosing cancer.

In the same example of cancer diagnosis data, if only 2% or less of the patients have cancer, then this would be a case of class imbalance, as the percentage of cancer patients is very small compared to rest of the population. There are main 2 approaches to handle this issue:

- - 1. **Use of a cost function**: In this approach, a cost associated with misclassifying data is evaluated with the help of a cost matrix (similar to the confusion matrix, but more concerned with False Positives and False Negatives). The main aim is to reduce the cost of misclassifying. The cost of a False Negative is always more than the cost of a False Positive. e.g. wrongly predicting a cancer patient to be cancer-free is more dangerous than wrongly predicting a cancer-free patient to have cancer.

Total Cost = Cost of FN * Count of FN + Cost of FP * Count of FP

- - 1. **Use of different sampling methods**: In this approach, you can use over-sampling, under-sampling, or hybrid sampling. In over-sampling, minority class observations are replicated to balance the data. Replication of observations leading to overfitting, causing good accuracy in training but less accuracy in unseen data. In under-sampling, the majority class observations are removed causing loss of information. It is helpful in reducing processing time and storage, but only useful if you have a large data set.

Find more about class imbalance [here](https://www.kdnuggets.com/2016/08/learning-from-imbalanced-classes.html).

If there are multiple classes in the target variable, then a confusion matrix of dimensions equal to the number of classes is formed, and all performance measures can be calculated for each of the classes. This is called a multiclass confusion matrix. e.g. there are 3 classes X, Y, Z in the response variable, so recall for each class will be calculated as below:

Recall_X = TP_X/(TP_X+FN_X)

Recall_Y = TP_Y/(TP_Y+FN_Y)

Recall_Z = TP_Z/(TP_Z+FN_Z)

------

 

### Q6. What are some ways I can make my model more robust to outliers?

[**Thuy Pham**](https://www.kdnuggets.com/author/thuy-pham)** answers:**

There are several ways to make a model more robust to outliers, from different points of view (data preparation or model building). **An outlier** in the question and answer is assumed being unwanted, unexpected, or a must-be-wrong value to the human’s knowledge so far (e.g. no one is 200 years old) rather than a rare event which is possible but rare.

Outliers are usually defined in relation to the distribution. Thus outliers could be removed in the pre-processing step (before any learning step), by using standard deviations (for normality) or interquartile ranges (for not normal/unknown) as threshold levels.

![img](https://www.kdnuggets.com/wp-content/uploads/outliers.jpg)
**Outliers.** [Image source](https://www.neuraldesigner.com/blog/3_methods_to_deal_with_outliers)

Moreover, **data transformation** (e.g. log transformation) may help if data have a noticeable tail. When outliers related to the sensitivity of the collecting instrument which may not precisely record small values, **Winsorization** may be useful. This type of transformation (named after Charles P. Winsor (1895–1951)) has the same effect as clipping signals (i.e. replaces extreme data values with less extreme values).  Another option to reduce the influence of outliers is using **mean absolute difference**rather mean squared error.

For model building, some models are resistant to outliers (e.g. [tree-based approaches](https://www.quora.com/Why-are-tree-based-models-robust-to-outliers)) or non-parametric tests. Similar to the median effect, tree models divide each node into two in each split. Thus, at each split, all data points in a bucket could be equally treated regardless of extreme values they may have. The study [Pham 2016] proposed a detection model that incorporates interquartile information of data to predict outliers of the data.



# Model metric

Unlike evaluating the accuracy of models that predict a continuous or discrete dependent variable like Linear Regression models, evaluating the accuracy of a classification model could be more complex and time-consuming. Before measuring the accuracy of classification models, an analyst would first measure its robustness with the help of metrics such as AIC-BIC, AUC-ROC, AUC- PR, Kolmogorov-Smirnov chart, etc. The next logical step is to measure its accuracy. To understand the complexity behind measuring the accuracy, we need to know few basic concepts.

**Model Output**

Most of the classification models output a probability number for the dataset.

E.g. – A classification model like Logistic Regression will output a probability number between 0 and 1 instead of the desired output of actual target variable like Yes/No, etc.

The next logical step is to translate this probability number into the target/dependent variable in the model and test the accuracy of the model. To understand the implication of translating the probability number, let’s understand few basic concepts relating to evaluating a classification model with the help of an example given below.

**Goal:** Create a classification model that predicts fraud transactions

**Output:** Transactions that are predicted to be Fraud and Non-Fraud

**Testing:** Comparing the predicted result with the actual results

**Dataset:** Number of Observations: 1 million; Fraud : 100; Non-Fraud: 999,900

The fraud observations constitute just **0.1%** of the entire dataset, representing a typical case of **Imbalanced Class**. Imbalanced Classes arises from classification problems where the classes are not represented equally. Suppose you created a model that predicted 95% of the transactions as Non-Fraud, and all the predictions for Non-Frauds turn out to be accurate. But, that high accuracy for Non-Frauds shouldn’t get you excited since Frauds are just 0.1% whereas the Predicted Frauds constitute 5% of the observations.

Assuming you were able to translate the output of your model to Fraud/Non-Fraud, the predicted result could be compared to actual result and summarized as follows:

**a) True Positives:** Observations where the actual and predicted transactions were fraud

**b) True Negatives:** Observations where the actual and predicted transactions weren’t fraud

**c) False Positives:** Observations where the actual transactions weren’t fraud but predicted to be fraud

**d) False Negatives:** Observations where the actual transactions were fraud but weren’t predicted to be fraud

**Confusion Matrix **is a popular way to represent the summarized findings.

| **True Positives (TP)**  | **False Negatives (FN)** |
| ------------------------ | ------------------------ |
| **False Positives (FP)** | **True Negatives (TN)**  |

 

Typically, a classification model outputs the result in the form of probabilities as shown below:

First 5 rows of the dataset:

| Observation | Actual    | Predicted |
| ----------- | --------- | --------- |
| 1           | Non-Fraud | 0.45      |
| 2           | Non-Fraud | 0.10      |
| 3           | Fraud     | 0.67      |
| 4           | Non-Fraud | 0.60      |
| 5           | Non-Fraud | 0.11      |

 

Suppose we assume 0.5 as the cut-off probability i.e. observations with probability value of 0.5 and above are marked as Fraud and below 0.5 are marked as Non-Fraud as shown in the table below:

Accordingly, the above first 5 rows will be as below:

| Observation | Actual    | Predicted |
| ----------- | --------- | --------- |
| 1           | Non-Fraud | Non-Fraud |
| 2           | Non-Fraud | Non-Fraud |
| 3           | Fraud     | Fraud     |
| 4           | Non-Fraud | Fraud     |
| 5           | Non-Fraud | Non-Fraud |

 

Let’s summarize the results from the model of the entire dataset with the help of the confusion matrix:

| **TP = 90** | **FN = 10**      |
| ----------- | ---------------- |
| **FP = 10** | **TN = 999,890** |

 

We have all non-zero cells in the above matrix. So is this result ideal?

Wouldn’t we love a scenario wherein the model accurately identifies the Frauds and the Non-Frauds i.e. zero entry for cells, FP and FN?

A BIG YES.

Consider a scenario wherein as a marketing analyst; you would like to identify users who were likely to buy but haven’t bought yet. This particular class of users would be the ones who share the characteristics of the users who bought. Such a class would belong to False Positives – Users who were predicted to transact but didn’t transact in reality. Hence, in addition to non-zero entries in TP and TN, you would prefer a non-zero entry in FP too. Thus, the model accuracy depends on the goal of the prediction exercise.

**Key Testing Metrics**

Since we are now comfortable with the interpretation of the Confusion Matrix, let’s look at some popular metrics used for testing the classification models:

**i) Sensitivity/Recall**
Sensitivity also known as the True Positive rate or Recall is calculated as,

Sensitivity = No. of True Positives / (No. of True Positives + No. of False Negatives)

Sensitivity = TP / (TP + FN)

Since the formula doesn’t contain FP and TN, Sensitivity may give you a biased result, especially for imbalanced classes.

In the example of Fraud detection, it gives you the percentage of Correctly Predicted Frauds from the pool of Actual Frauds.

Sensitivity = 90 / (90 + 10) = 0.90

**ii) Specificity**
Specificity, also known as True Negative Rate is calculated as,

Specificity = No. of True Negatives / (No. of True Negatives + No. of False Positives)

Specificity = TN / (TN + FP)

Since the formula does not contain FN and TP, Specificity may give you a biased result, especially for imbalanced classes.

In the example of Fraud detection, it gives you the percentage of Correctly Predicted Non-Frauds from the pool of Actual Non-Frauds.

Specificity = 999,890 / (999,890 + 10) = 1

**iii) Precision**
Precision also known as Positive Predictive Value is calculated as,

Precision = No. of True Positives / (No. of True Positives + No. of False Positives)

Precision = TP / (TP + FP)

Since the formula does not contain FN and TN, Precision may give you a biased result, especially for imbalanced classes.

In the example of Fraud detection, it gives you the percentage of Correctly Predicted Frauds from the pool of Total Predicted Frauds.

Precision = 90 / (90 + 10) = 0.90

**iv) F1 score**

F1 score incorporates both Recall and Precision and is calculated as,

F1 score = 2 * (Precision * Recall) / (Precision + Recall)

The F1 score represents a more balanced view compared to the above 3 metrics but could give a biased result in the scenario discussed later since it doesn’t include TN.

F1 score = 2 * (0.90 * 0.90) / (0.90 + 0.90) = 0.90

**v) Matthews Correlation Coefficient (MCC)**

Unlike the other metrics discussed above, MCC takes all the cells of the Confusion Matrix into consideration in its formula.

MCC = TP * TN – FP * FN / √ (TP +FP) * (TP + FN) * (TN + FP) * (TN + FN)

Similar to Correlation Coefficient, the range of values of MCC lie between -1 to +1. A model with a score of +1 is a perfect model and -1 is a poor model. This property is one of the key usefulness of MCC as it leads to easy interpretability.

MCC = 90*999,890 – 10*10 / √(90+10)*(90+10)*(999,890+10)*(999,890+10)

MCC = 0.90

**Metric Comparison**
We will test and compare the result of the classification model at few probability cut-off values using the above-mentioned testing metrics.

Scenario A: Confusion Matrix at cut-off value of 0.5

We shall take this scenario (cut-off value of 0.5) as the base case and compare the result of the base case with different cut-off values.

Confusion Matrix

| **TP = 90** | **FN = 10**      |
| ----------- | ---------------- |
| **FP = 10** | **TN = 999,890** |

 

Testing Metrics

| Sensitivity | Specificity | Precision | F1 Score | MCC  |
| ----------- | ----------- | --------- | -------- | ---- |
| 0.90        | 1.00        | 0.90      | 0.90     | 0.90 |

 

Scenario B: Confusion Matrix at cut-off value of 0.4

Confusion Matrix

| **TP = 90**   | **FN = 10**      |
| ------------- | ---------------- |
| **FP = 1910** | **TN = 997,990** |

 

It can be clearly observed that for Scenario B, there is a substantial increase in FP compared to Scenario A. Hence, there should be deterioration in the metrics.

Testing Metrics

| Sensitivity | Specificity | Precision | F1 Score | MCC  |
| ----------- | ----------- | --------- | -------- | ---- |
| 0.90        | 1.00        | 0.05      | 0.09     | 0.20 |

 

There is no change in Sensitivity & Specificity, which is constant.

Scenario C: Confusion Matrix at cut-off value of 0.6

Confusion Matrix

| **TP = 90** | **FN = 1910**    |
| ----------- | ---------------- |
| **FP = 10** | **TN = 997,990** |

 

There is a substantial increase in FN compared to Scenario A. Hence, there should be deterioration in the metrics compared to A.

Testing Metrics

| Sensitivity | Specificity | Precision | F1 Score | MCC  |
| ----------- | ----------- | --------- | -------- | ---- |
| 0.05        | 1.00        | 0.90      | 0.09     | 0.20 |

 

Here there is no change in Specificity & Precision while there is a general decline in other metrics.

Based on our findings, we can say that F1 score and MCC is making more sense compared to Sensitivity and Specificity.

In the example, we have built a model to predict Fraud. We can use the same model to predict Non-Fraud. In such a case, the Confusion Matrix will be as given below:

**Scenario D**: Confusion Matrix at cut-off value of 0.5

Confusion Matrix

| **TP = 999,890** | **FN = 10** |
| ---------------- | ----------- |
| **FP = 10**      | **TN = 90** |

 

The above confusion matrix is just the transpose of the matrix given in Scenario A since the model is predicting Non-Frauds instead of Frauds. So the True Negatives in Scenario A will be the True Positives for Scenario D, likewise for other cells. Ideally, the testing metrics should be the same for Scenario A and D.

Testing Metrics

| Sensitivity | Specificity | Precision | F1 Score | MCC  |
| ----------- | ----------- | --------- | -------- | ---- |
| 1           | 0.90        | 1         | 1        | 0.90 |

 

Except for MCC all the other testing metrics have changed.

**Summary of Testing Metrics for all the scenarios:**

| Scenario | Sensitivity | Specificity | Precision | F1 Score | MCC  |
| -------- | ----------- | ----------- | --------- | -------- | ---- |
| A        | 0.90        | 1.00        | 0.90      | 0.90     | 0.90 |
| B        | 0.90        | 1.00        | 0.05      | 0.09     | 0.20 |
| C        | 0.05        | 1.00        | 0.90      | 0.09     | 0.20 |
| D        | 1           | 0.90        | 1         | 1        | 0.90 |

 

**Conclusion**
As an analyst, if you are looking at a metric to measure and maximize the overall accuracy of the classification model, MCC seems to the best bet since it is not only easily interpretable but also robust to changes in the prediction goal.

[Original post](https://blog.clevertap.com/the-best-metric-to-measure-accuracy-of-classification-models). Reposted with permission.

**Bio: Jacob Joseph** works with CleverTap, a digital analytics, user engagement and personalization platform where he is an integral part leading their data science team. His role encompasses deriving key actionable business insights and applying machine learning algorithms to augment CleverTap’s effort to deliver world-class real time analytics to its customers.

**Related:**

- [Interpretability over Accuracy](https://www.kdnuggets.com/2016/08/salford-interpretability-over-accuracy.html)
- [Dealing with Unbalanced Classes, SVMs, Random Forests, and Decision Trees in Python](https://www.kdnuggets.com/2016/04/unbalanced-classes-svm-random-forests-python.html)
- [Statistics – Understanding the Levels of Measurement](https://www.kdnuggets.com/2015/08/statistics-understanding-levels-measurement.html)

### Q7. What is overfitting and how to avoid it?

[**Gregory Piatetsky**](https://www.kdnuggets.com/author/gregory-piatetsky)** answers:**  

*(Note: this is a revised version of the answer given in *[*21 Must-Know Data Science Interview Questions and Answers, part 2*](https://www.kdnuggets.com/2016/02/21-data-science-interview-questions-answers-part2.html)*)*

[Overfitting](https://www.kdnuggets.com/tag/overfitting) is when you build a predictive model that fits the data "too closely",  so that it captures the random noise in the data rather than true patterns.  As a result, the model predictions will be wrong when applied to new data.

We frequently hear about studies that report unusual results (especially if you listen to Wait Wait Don't Tell Me) , or see findings like "[an orange used car is least likely to be a lemon](https://www.kdnuggets.com/2017/01/siegel-data-science-avoiding-prediction-pitfall.html)",  or learn that studies overturn previous established findings (eggs are no longer bad for you).

Many such studies produce questionable results that cannot be repeated.

This is a big problem, especially in social sciences or medicine, when researchers  frequently commit the cardinal sin of Data Science - **Overfitting the data.**

The researchers test too many hypotheses without proper statistical control, until they happen to find something interesting. Then they report it.  Not surprisingly, next time the effect (which was partly due to chance) will be much smaller or absent.

These flaws of research practices were identified and reported by John P. A. Ioannidis in his landmark paper [*Why Most Published Research Findings Are False*](http://www.plosmedicine.org/article/info%3Adoi%2F10.1371%2Fjournal.pmed.0020124) (PLoS Medicine, 2005). Ioannidis found that very often either the results were exaggerated or the findings could not be replicated. In his paper, he presented statistical evidence that indeed most claimed research findings are false!

Ioannidis noted that in order for a research finding to be reliable, it should have:

- Large sample size and with large effects
- Greater number of and lesser selection of tested relationship
- Greater flexibility in designs, definitions, outcomes, and analytical modes
- Minimal bias due to financial and other factors (including popularity of that scientific field)

Unfortunately, too often these rules were violated, producing spurious results, such as S&P 500 index strongly correlated to [production of butter in Bangladesh](https://www.kdnuggets.com/2016/02/21-data-science-interview-questions-answers-part2.html), or US spending on science, space and technology correlated with suicides by hanging, strangulation, and suffocation (from http://tylervigen.com/spurious-correlations)

![Spurious correlations](https://www.kdnuggets.com/images/tylervigen-us-spending-science-vs-suicides-720.jpg)
(Source: [Tylervigen.com](http://tylervigen.com/chart-pngs/1.png))

See more strange and spurious findings at [Spurious correlations](http://www.tylervigen.com/discover) by Tyler Vigen or discover them by yourself using tools such as [Google correlate](http://www.google.com/trends/correlate/).

Several methods can be used to avoid "overfitting" the data:

- Try to find the simplest possible hypothesis
- [Regularization](http://en.wikipedia.org/wiki/Regularization_%28mathematics%29) (adding a penalty for complexity)
- [Randomization Testing](https://www.kdnuggets.com/2014/02/3-ways-to-test-accuracy-your-predictive-models.html) (randomize the class variable, try your method on this data - if it find the same strong results, something is wrong)
- Nested cross-validation  (do feature selection on one level, then run entire method in cross-validation on outer level)
- Adjusting the [False Discovery Rate](http://en.wikipedia.org/wiki/False_discovery_rate)
- Using the [reusable holdout method](https://www.kdnuggets.com/2015/08/feldman-avoid-overfitting-holdout-adaptive-data-analysis.html) - a breakthrough approach proposed in 2015

Good data science is on the leading edge of scientific understanding of the world, and it is data scientists responsibility to avoid overfitting data and educate the public and the media on the dangers of bad data analysis.

See also:

- [4 Reasons Your Machine Learning Model is Wrong (and How to Fix It)](https://www.kdnuggets.com/2016/12/4-reasons-machine-learning-model-wrong.html)
- [When Good Advice Goes Bad](https://www.kdnuggets.com/2016/03/when-good-advice-goes-bad.html)
- [The Cardinal Sin of Data Mining and Data Science: Overfitting](https://www.kdnuggets.com/2014/06/cardinal-sin-data-mining-data-science.html)
- [Big Idea To Avoid Overfitting: Reusable Holdout to Preserve Validity in Adaptive Data Analysis](https://www.kdnuggets.com/2015/08/feldman-avoid-overfitting-holdout-adaptive-data-analysis.html)
- [Overcoming Overfitting with the reusable holdout: Preserving validity in adaptive data analysis](https://www.kdnuggets.com/2015/08/reusable-holdout-preserving-validity-adaptive-data-analysis.html)
- [11 Clever Methods of Overfitting and how to avoid them](https://www.kdnuggets.com/2015/01/clever-methods-overfitting-avoid.html)

------

### Q8. What is the curse of dimensionality?

[**Prasad Pore**](https://www.kdnuggets.com/author/prasad-pore)** answers:**

> "As the number of features or dimensions grows, the amount of data we need to generalize accurately grows exponentially."
>
> - [Charles Isbell](http://ccsubs.com/video/yt%3AQZ0DtNFdDko/curse-of-dimensionality-georgia-tech-machine-learning/subtitles?lang=en), Professor and Senior Associate Dean, School of Interactive Computing, Georgia Tech

Let’s take an example below. Fig. 1 (a) shows 10 data points in one dimension i.e. there is only one feature in the data set. It can be easily represented on a line with only 10 values, x=1, 2, 3... 10.

But if we add one more feature, same data will be represented in 2 dimensions (Fig.1 (b)) causing increase in dimension space to 10*10 =100. And again if we add 3rd feature, dimension space will increase to 10*10*10 = 1000. As dimensions grows, dimensions space increases exponentially.

```
   10^1 = 10

   10^2 = 100

   10^3 = 1000 and so on...
```

![n-dimensional space comparison](https://www.kdnuggets.com/wp-content/uploads/17-qa-2-image01.jpg)

This exponential growth in data causes high sparsity in the data set and unnecessarily increases storage space and processing time for the particular modelling algorithm. Think of image recognition problem of high resolution images 1280 × 720 = 921,600 pixels i.e. 921600 dimensions. OMG. And that’s why it’s called **Curse of Dimensionality**. Value added by additional dimension is much smaller compared to overhead it adds to the algorithm.

Bottom line is, the data that can be represented using 10 space units of one true dimension, needs 1000 space units after adding 2 more dimensions just because we observed these dimensions during the experiment. The true dimension means the dimension which accurately generalize the data and observed dimensions means whatever other dimensions we consider in dataset which may or may not contribute to accurately generalize the data.

------

### Q9. How can you determine which features are the most important in your model?

[**Thuy Pham**](https://www.kdnuggets.com/author/thuy-pham)** answers:**

In applied machine learning, success depends significantly on the quality of data representation (features).  Highly correlated features can make learning/sorting steps in the classification module easy. Conversely if label classes are a very complex function of the features, it can be impossible to build a good model [Dom 2012]. Thus a so-called **feature engineering**[, a process of transforming data into features that are most relevant to the problem, is often needed.](https://www.kdnuggets.com/tags/feature-engineering)

[A ](https://www.kdnuggets.com/tags/feature-engineering)[**feature selection**](https://www.kdnuggets.com/tag/feature-selection) scheme often involves techniques to automatically select salient features from a large exploratory feature pool. Redundant and irrelevant features are well known to cause poor accuracy so discarding these features should be the first task. The relevance is often scored using mutual information calculation. Furthermore, input features should thus offer a high level of discrimination between classes. The separability of features can be measured by distance  or variance ratio between classes. One recent work [Pham 2016] proposed a systematic voting based feature selection that is a data-driven approach incorporating above criteria. This can be used as a common framework for a wide class of problems.

![Feature selection approach](https://www.kdnuggets.com/wp-content/uploads/17-qa-2-image03.jpg)
A data-driven feature selection approach incorporating several saliency criteria [Pham 2016].

Another approach is penalizing on the features that are not very important (e.g., yield a high error metric) when using regularization  methods like Lasso or Ridge.

References:

- [Dom 2012] P. Domingos. A few useful things to know about machine learning. *Communications of the ACM*, 55(10):78–87, 2012. 2.4
- [Pham 2016] T. T. Pham, C. Thamrin, P. D. Robinson, and P. H. W. Leong. Respiratory artefact removal in forced oscillation measurements: A machine learning approach. *Biomedical Engineering, IEEE Transactions on*, accepted, 2016.

------

### Q10. When can parallelism make your algorithms run faster? When could it make your algorithms run slower?

[**Anmol Rajpurohit**](https://twitter.com/hey_anmol)** answers:**

Parallelism is a good idea when the task can be divided into sub-tasks that can be executed independent of each other without communication or shared resources. Even then, efficient implementation is key to achieving the benefits of parallelization. In real-life, most of the programs have some sections that need to be executed in serialized fashion, and the parallelizable sub-tasks need some kind of synchronization or data transfer. Thus, it is hard to predict whether parallelization will actually make the [**algorithm**](https://www.kdnuggets.com/tag/algorithms) run faster (than the serialized approach).

Parallelism would always have overhead compared to the compute cycles required to complete the task sequentially. At the minimum, this overhead will comprise of dividing the task into sub-tasks and compiling together the results of sub-tasks.

**The performance of parallelism against sequential computing is largely determined by how the time consumed by this overhead compares to the time saved due to parallelization.**

Note: The overhead associated with parallelism is not just limited to the run-time of code, but also includes the extra time required for coding and debugging (parallelism versus sequential code).

A widely-known theoretical approach to assessing the benefit of parallelization is Amdahl’s law, which gives the following formula to measure the speedup of running sub-tasks in parallel (over different processors) versus running them sequentially (on a single processor):

![img](https://www.kdnuggets.com/wp-content/uploads/17-qa-2-image02.png) 

where:

- *S**latency** *is the theoretical speedup of the execution of the whole task;
- *s* is the speedup of the part of the task that benefits from improved system resources;
- *p *is the proportion of execution time that the part benefiting from improved resources originally occupied.

To understand the implication of Amdahl’s Law, look at the following figure that illustrates the theoretical speedup against an increasing number of processor cores, for tasks with different level of achievable parallelization:

![Speedup by number of cores](https://www.kdnuggets.com/wp-content/uploads/17-qa-2-image05.png)

It is important to note that not every program can be effectively parallelized. Rather, very few programs will scale with perfect speedups because of the limitations due to sequential portions, inter-communication costs, etc. Usually, large data sets form a compelling case for parallelization. However, it should not be assumed that parallelization would lead to performance benefits. Rather, the performance of parallelism and sequential should be compared on a sub-set of the problem, before investing effort into parallelization.

------

### Q11. What is the idea behind ensemble learning?

[**Prasad Pore**](https://www.kdnuggets.com/author/prasad-pore)** answers:**

> "In [statistics](https://en.wikipedia.org/wiki/Statistics) and [machine learning](https://en.wikipedia.org/wiki/Machine_learning), **ensemble methods** use multiple learning algorithms to obtain better [predictive performance](https://en.wikipedia.org/wiki/Predictive_inference) than could be obtained from any of the constituent learning algorithms alone."
>
> – [Wikipedia](https://en.wikipedia.org/wiki/Ensemble_learning).

Imagine you are playing the game “Who wants to be millionaire?” and reached up to last question of 1 million dollars. You have no clue about the question, but you have audience poll and phone a friend life lines. Thank God. At this stage you don’t want to take any risk, so what will you do to get sure-shot right answer to become millionaire?

You will use both life lines, isn’t it? Let’s say 70% audience is saying right answer is D and your friend is also saying the right answer is D with 90% confidence because he is an expert in the area of the question. Use of both life lines gives you  an average 80% confidence that D is correct and gets you closer to becoming a millionaire.

This is the approach of [**ensemble methods**](https://www.kdnuggets.com/tag/ensemble-methods).

The famous [Netflix Prize](https://en.wikipedia.org/wiki/Netflix_Prize) competition took almost 3 years before the goal of 10% improvement [was reached](https://www.kdnuggets.com/news/2009/n14/1i.html).  The winners used gradient boosted decision trees to combine over [500 models](http://blog.echen.me/2011/10/24/winning-the-netflix-prize-a-summary/).

In ensemble methods, more diverse the models used, more robust will be the ultimate result.

Different models used in ensemble improves overall variance from difference in population, difference in hypothesis generated, difference in algorithms used and difference in parametrization. There are main 3 widely used ensembles techniques:

1. Bagging
2. [Boosting](https://www.kdnuggets.com/tag/boosting)
3. Stacking

So if you have different models built for same data and same response variable, you can use one of the above methods to build ensemble model. As every model used in the ensemble has its own performance measures, some of the models may perform better than ultimate ensemble model and some of them may perform poorer than or equal to ensemble model. But overall the ensemble methods will improve overall accuracy and stability of the model, although at the expense of model understandability.

For more on ensemble methods see:

- [Ensemble Methods: Elegant Techniques to Produce Improved Machine Learning Results](https://www.kdnuggets.com/2016/02/ensemble-methods-techniques-produce-improved-machine-learning.html)
- [Data Science Basics: An Introduction to Ensemble Learners](https://www.kdnuggets.com/2016/11/data-science-basics-intro-ensemble-learners.html)

------

### Q12. In unsupervised learning, if a ground truth about a dataset is unknown, how can we determine the most useful number of clusters to be?

[**Matthew Mayo**](https://www.kdnuggets.com/author/matt-mayo)** answers:**

With **supervised** learning, the number of classes in a particular set of data is known outright, since each data instance in labeled as a member of a particular existent class. In the worst case, we can scan the class attribute and count up the number of unique entries which exist.

With **unsupervised** learning, the idea of class attributes and explicit class membership does not exist; in fact, one of the dominant forms of unsupervised learning -- data clustering -- aims to approximate class membership by minimizing interclass instance similarity and maximizing intraclass similarity. A major drawback with clustering can be the requirement to provide the number of classes which exist in the unlabeled dataset at the onset, in some form or another. If we are lucky, we may know the data’s **ground truth** -- the actual number of classes -- beforehand. However, this is not always the case, for numerous reasons, one of which being that there may actually be no defined number of classes (and hence, clusters) in the data, with the whole point of the unsupervised learning task being to survey the data and attempt to impose some meaningful structure of optimal cluster and class numbers upon it.

Without knowing the ground truth of a dataset, then, how do we know what the optimal number of data clusters are? As one may expect, there are actually numerous methods to go about answering this question. We will have a look at 2 particular popular methods for attempting to answer this question: the elbow method and the silhouette method.

**The Elbow Method**

The elbow method is often the best place to state, and is especially useful due to its ease of explanation and verification via visualization. The elbow method is interested in explaining variance as a function of cluster numbers (the *k* in *k*-means). By plotting the percentage of variance explained against *k*, the first *N* clusters should add significant information, explaining variance; yet, some eventual value of *k* will result in a much less significant gain in information, and it is at this point that the graph will provide a noticeable angle. This angle will be the optimal number of clusters, from the perspective of the elbow method,

It should be self-evident that, in order to plot this variance against varying numbers of clusters, varying numbers of clusters must be tested. Successive complete iterations of the clustering method must be undertaken, after which the results can be plotted and compared.

![Elbow method](https://www.kdnuggets.com/wp-content/uploads/17-qa-2-image04.png)
[Image source](http://elf11.github.io/2015/08/18/Kmeans-analysis.html).

**The Silhouette Method**

The silhouette method measures the similarity of an object to its own cluster -- called cohesion -- when compared to other clusters -- called separation. The silhouette value is the means for this comparison, which is a value of the range [-1, 1]; a value close to 1 indicates a close relationship with objects in its own cluster, while a value close to -1 indicates the opposite. A clustered set of data in a model producing mostly high silhouette values is likely an acceptable and appropriate model.

![Silhouette method](https://www.kdnuggets.com/wp-content/uploads/17-qa-2-image00.png)
[Image source](http://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html).

Read more on the silhouette method [here](http://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html). Find the specifics on computing a silhouette value [here](https://en.wikipedia.org/wiki/Silhouette_(clustering)).

### Q13. What makes a good data visualization?

**Gregory Piatetsky answers:**

*Note: This answer contains excerpts from the recent post What makes a good data visualization – a Data Scientist perspective.*

Data Science is more than just building predictive models - it is also about explaining the models and using them to help people to understand data and make decisions. Data visualization is an integral part of presenting data in a convincing way.

There is a ton of research of good data visualization and how people best perceive information - see work by [Stephen Few](https://www.interaction-design.org/literature/book/the-encyclopedia-of-human-computer-interaction-2nd-ed/data-visualization-for-human-perception) and many others.

Guidelines on improving human perception include:

- position data along a common scale
- bars are more effective than circles or squares in communicating size
- color is more discernible than shape in scatterplots
- avoid pie chart unless it is for showing proportions
- avoid 3D charts and reduce chartjunk
- Sunburst visualization is more effective for hierarchical plots
- use small multiples (even though animation looks cool, it is less effective for understanding changing data.)

See [39 studies about human perception](https://medium.com/@kennelliott/39-studies-about-human-perception-in-30-minutes-4728f9e31a73#.irdxe7jip), by Washington Post graphics editor for a lot more detail.

From Data Science point of view, what makes visualization important is highlighting the key aspects of data - what are the most important variables, what is their relative importance, what are the changes and trends.

![Chart Junk](https://www.kdnuggets.com/wp-content/uploads/chart-junk-diamonds-300.jpg)
Data visualization should be visually appealing but not at the expense of loading a chart with unnecessary junk, like in this extreme example on the right.

**How do we make a good data visualization?**

To do that, choose the right type of chart for your data:

- Line Charts to track changes or trends over time and show the relationship between two or more variables.
- Bar Charts to compare quantities of different categories.
- Scatter Plots show joint variation of two data items.
- Pie Charts to compare parts of a whole - used them sparingly since people have hard time comparing the area of pie slices
- You can show additional variables on a 2-D plot using color, shape, and size
- Use interactive dashboards to allow experiments with key variables

Here is an example of visualization of US Presidential Elections, 1976-2016, that shows multiple variables at once: the electoral college votes difference (y-axis), the % popular vote difference (X-axis), the size of the popular vote (circle area), winner party (color), and winner name and year (label). See my post on [What makes a good data visualization](https://www.kdnuggets.com/2017/03/what-makes-good-data-visualization.html) for more details.

![Visualization Us Elections 1976 2016](https://www.kdnuggets.com/images/visualization-us-elections-1976-2016.jpg)
**US Presidential Elections, 1976-2016**.

References:

- [What makes a good visualization](http://www.informationisbeautiful.net/visualizations/what-makes-a-good-data-visualization/), David McCandless, Information is Beautiful
- [5 Data Visualization Best Practices](https://www.gooddata.com/blog/5-data-visualization-best-practices), GoodData
- [39 studies about human perception in 30 minutes](https://medium.com/@kennelliott/39-studies-about-human-perception-in-30-minutes-4728f9e31a73#.irdxe7jip), Kenn Elliott
- [Data Visualization for Human Perception](https://www.interaction-design.org/literature/book/the-encyclopedia-of-human-computer-interaction-2nd-ed/data-visualization-for-human-perception), landmark work by Stephen Few (key ideas summarized [here](https://viscomvibz.wordpress.com/2012/02/27/data-visualization-for-human-perception/))

------

### Q14. What are some of the common data quality issues when dealing with Big Data? What can be done to avoid them or to mitigate their impact?

**Anmol Rajpurohit answers:**

The most common data quality issues observed when dealing with Big Data can be best understood in terms of the key characteristics of Big Data – Volume, Velocity, Variety, Veracity, and Value.

**Volume:**

In the traditional data warehouse environment, comprehensive data quality assessment and reporting was at least possible (if not, ideal). However, in the Big Data projects the scale of data makes it impossible. Thus, the data quality measurements can at best be approximations (i.e. need to be described in probability and confidence intervals, and not in terms of absolute values). We also need to re-define most of the data quality metrics based on the specific characteristics of the Big Data project so that those metrics can have a clear meaning, be measured (good approximation) and be used for evaluating the alternative strategies for data quality improvement.

Despite the great volume of underlying data, it is not uncommon to find out that some desired data was not captured or is not available for other reasons (such as high cost, delay in getting it, etc.). It is ironical but true that data availability continues to be a prominent data quality concern in the Big Data era.

**Velocity:**

The tremendous pace of data generation and collection makes it incredibly hard to monitor data quality within a reasonable overhead on time and resources (storage, compute, human effort, etc.). So, by the time data quality assessment completes, the output might be outdated and of little use, particularly if the Big Data project is to serve any real-time or near real-time business needs. In such scenarios, you would need to re-define data quality metrics so that they are relevant as well as feasible in the real-time context.

Sampling can help you gain speed for the data quality efforts, but this comes at the cost of a bias (which eventually makes the end result less useful) because of the fact that samples are rarely an accurate representation of the entire data. Lesser samples will give higher speed, but with a bigger bias.

Another impact of velocity is that you might have to do data quality assessments on-the-fly, i.e. somewhere plugged-in within the data collection/transfer/storage processes; as the critical time-constraint does not give you the privilege of making a copy of a selected data subset, storing it elsewhere and running data quality assessments on it.

**Variety:**

One of the biggest data quality issues in Big Data is that the data includes several data types (structured, semi-structured, and unstructured) coming in from different data sources. Thus, often a single data quality metric will not be applicable for the entire data and you would need to separately define data quality metrics for each data type. Moreover, assessing and improving the data quality of unstructured or semi-structured data is way more tricky and complex than that of structured data. For example, when mining the physician notes from medical records across the world (related to a particular medical condition) even if the language (and the grammar) is same the meaning might be very different due to local dialects and slang. This leads to low data interpretability, another data quality measure.

Data from different sources often has serious semantic differences. For example, “profit” can have widely varied definitions across the business units of an organization or external agencies. Thus, the fields with identical names may not mean the same thing. This problem is made worse by the lack of adequate and consistent meta-data from each data source. In order to make sense of data, you need reliable metadata (such as to make sense of sales numbers from a store, you need other information such as date-time, items purchased, coupons used, etc.). Usually, a lot of these data sources are outside an organization and thus, it is very hard to ensure good metadata for such data.

Another common issue is syntactic inconsistencies. For example, “time-stamp” values from different sources would be incompatible unless they are captured along with the time zone information.

![img](http://informationcatalyst.com/wp-content/uploads/2015/09/BD-5Vs.png)
[Image source](http://informationcatalyst.com/index.php/vision-experience/big-data-value/).

**Veracity:**

Veracity, one of the most overlooked Big Data characteristics, is directly related to data quality, as it refers to the inherent biases, noise and abnormality in data. Because of veracity, the data values might not be exact real values, rather they might be approximations. In other words, the data might have some inherent impreciseness and uncertainty. Besides data inaccuracies, Veracity also includes data consistency (defined by the statistical reliability of data) and data trustworthiness (based on data origin, data collection and processing methods, security infrastructure, etc.). These data quality issues in turn impact data integrity and data accountability.

While the other V’s are relatively well-defined and can be easily measured, Veracity is a complex theoretical construct with no standard approach for measurement. In a way this reflects how complex the topic of “data quality” is within the Big Data context.

Data users and data providers are often different organizations with very different goals and operational procedures. Thus, it is no surprise that their notions of data quality are very different. In many cases, the data providers have no clue about the business use cases of data users (data providers might not even care about it, unless they are getting paid for the data). This disconnect between data source and data use is one of the prime reasons behind the data quality issues symbolized by Veracity.

**Value:**

The Value characteristic connects directly to the end purpose. Organizations are harnessing Big Data for many diverse business pursuits, and those pursuits are the real drivers of how data quality is defined, measured, and improved.

A common and old definition of data quality is that it is the “fitness of use” for the data consumer. This means that data quality is dependent on what you plan to do with the data. Thus, for a given data two different organizations with different business goals will most likely have widely different measurements of data quality.This nuance is often not well understood – data quality is a “relative” term. A Big Data project might involve incomplete and inconsistent data, however, it is possible that those data quality issues do not impact the utility of data towards the business goal. In such a case, the business would say that the data quality is great (and will not be interested in investing in data quality improvements). For example, for a producer of mashed potato cans a batch of small potatoes would be of same quality as a batch of big potatoes. However, for a fast food restaurant making fries, the quality of the two batches would be radically different.

The Value aspect also brings in the “cost-benefit” perspective to data quality – whether it would be worth to resolve a given data quality issue, which issues should be resolved on priority, etc.

**Putting it all together:**

Data quality in Big Data projects is a very complex topic, where the theory and practice often differ. I haven’t come across any standard theory yet that is widely-accepted. Rather, I see little interest in the industry towards this goal.In practice, data quality does play an important role in the design of Big Data architecture. All the data quality efforts must start from a solid understanding of high-priority business use cases, and use that insight to navigate various trade-offs (samples given below) to optimize the quality of the final output.

Sample trade-offs related to data quality:

- Is it worth improving the timeliness of data at the expense of data completeness and/or inadequate assessment of accuracy?
- Should we select data for cleaning based on cost of cleaning effort or based on how frequently the data is used or based on its relative importance within the data models consuming it? Or, a combination of those factors? What sort of combination?
- Is it a good idea to improve data accuracy through getting rid of incomplete or erroneous data? While removing some data, how do we ensure that no bias is getting introduced?

Given the magnanimous scope of work and very limited resources (relatively!), one common way for data quality efforts on Big Data projects is to adopt the baseline approach, in which, the data users are surveyed to identify and document the bare minimum data quality needed to ensure that the business processes they support are not disrupted. These minimum satisfactory levels of data quality are referred to as the baseline, and the data quality efforts are focused on ensuring that data quality for each data does not fall beyond its baseline level. It looks like a good starting point and you may later move into more advanced endeavors (based on business needs and available budget).

**Summary of Recommendations to improve data quality in Big Data projects:**

- **Identify and prioritize the business use cases** (then, use them to define data quality metrics, measurement methodology, improvement goals, etc.)
- Based on a strong understanding of the business use cases and the Big Data architecture implemented to achieve them, **design and implement an optimal layer of data governance** (data definitions, metadata requirements, data ownership, data flow diagrams, etc.)
- **Document baseline quality levels for key data** (think of “critical-path” diagram and “throughput-bottleneck” assessment)
- **Define ROI for data quality efforts** (in order to create feedback loop on the ROI metric to improve efficiency and to sustain funding for data quality efforts)
- **Integrate data quality efforts** (to achieve efficiency through minimizing redundancy)
- **Automate data quality monitoring** (to reduce cost as well as to let employees stay focused on complex tasks)

**Do not rely on machine learning to automatically take care of poor data quality** (machine learning is science and not magic!)

### Q15. In an A/B test, how can we ensure that assignment to the various buckets is truly random?

**Matthew Mayo answers:**

First, let’s consider how we can best ensure comparability between buckets prior to bucket assignment, without knowledge of any distribution of attributes in the population.

The answer here is simple: random selection and bucket assignment. Random selection and assignment to buckets without regard to any attribute of the population is a statistically sound approach, given a large enough population to draw from.

For example, let’s say you are testing a change to a website feature and are interested in response from only a particular region, the US. By first splitting into 2 groups (control and treatment) without regard to user region (and given a large enough population size), assignment of US visitors should be split between these groups. From these 2 buckets, visitor attributes can then be inspected for the purposes of testing, such as:

```
  if (region == "US" && bucket == "treatment"):
      # do something treatment-related here
  else:
      if (region == "US" && bucket == "control"):
          # do something control-related here
      else:
          # catch-all for non-US (and not relevant to testing scenario)

```

![Bias AB testing](https://www.kdnuggets.com/wp-content/uploads/bias-ab-testing.png)
[Image Source](https://blog.twitter.com/2015/detecting-and-avoiding-bucket-imbalance-in-ab-tests).

Bear in mind that, even after performing a round of random bucket assignment, statistical testing can be utilized to inspect/verify random distribution of bucket member attributes (e.g. ensure that significantly more US visitors did not get assigned to bucket A). If not, a new random assignment can be attempted (with a similar inspection/verification process), or -- if it is determined that the population does not conform to a cooperative distribution -- an approach such as the following can be pursued.

If we happen to know of some uneven population attribute distribution prior to bucket assignment, [stratified random sampling](https://en.wikipedia.org/wiki/Stratified_sampling) may be helpful in ensuring more evenly distributed sampling. Such a strategy can help eliminate selection bias, which is the archenemy of A/B testing.

**References:**

- [Detecting and avoiding bucket imbalance in A/B tests](https://blog.twitter.com/2015/detecting-and-avoiding-bucket-imbalance-in-ab-tests)
- [What are the methods to ensure that the population split for A/B test is random?](http://datascience.stackexchange.com/questions/10406/what-are-the-methods-to-ensure-that-the-population-split-for-a-b-test-is-random)
- [A/B Testing](https://www.optimizely.com/ab-testing/)

------

### Q16. How would you conduct an A/B test on an opt-in feature?

**Matthew Mayo answers:**

This seems to be a somewhat ambiguous question with a variety of interpretable meanings (an idea supported by [this post](http://stats.stackexchange.com/questions/95620/how-to-conduct-am-a-b-test-for-a-feature-which-cannot-be-accessed-by-every-visit)). Let's first look at the different possible interpretations of this questions and go from there.

1. How would you conduct an A/B test on an opt-in version of a feature to a non-opt-in-version?

   ​

   This would not allow for a fair or meaningful A/B test, since one bucket would be filled from the entire site's users, while the other would be filled from the group which has already opted in. Such a test would be akin to comparing some apples to all oranges, and thus ill-advised.

   ​

2. How would you conduct an A/B test on the adoption (or use) of an opt-in feature (i.e. test the actual opting-in)?

   ​

   This would be testing the actual opting in -- such as the testing between 2 versions of a "click here to sign up" feature -- and as such is just a regular A/B test (see the above question for some insight).

   ​

3. **How would you conduct an A/B test on different versions of an opt-in feature (i.e. for those having already opted in)?**
   This could, again, be construed as one of a few meanings, but I intend to approach it as a complex scenario of the chaining together of events, expanded upon below.

![Choose your A/B weapon](https://www.kdnuggets.com/wp-content/uploads/choose-ab-test-type.png)

Let's flesh out #3 from the list above. Let's first look at a simple chaining of events which can be tested, and then generalize. Suppose you are performing an A/B test on an email campaign. Let's say the variable will be subject line, and that content remains constant between the 2. Suppose the subject lines are as follows:

1. We have something for you
2. The greatest online data science courses are free this weekend! Try now, no commitment!

Contrived, to be sure. All else aside, intuition would say that subject #2 would get more action.

But beyond that, there is psychology at play. Even though the content which follows after clicking either of the subjects is the same, the individual clicking the second subject could reasonably be assumed to have a higher level of excitement and anticipation of what is to follow. This difference in expectations and level of commitment between the groups may lead to a higher percentage of click-throughs for those in the bucket with subject line #2 -- again, even with the same content.

Pivoting slightly... **How would you conduct an A/B test on different versions of an opt-in feature (i.e. for those having already opted in)?**

If my interpretation of evaluating a series of chained events is correct, such an A/B test could commence with different feeder locations to the same opt-in -- of the same content -- and move to to different follow-up landing spots after opt-in, with the intent of measuring what users do on the resulting landing page being the goal.

Do different originating locations to the same opt-in procedure result in different follow-up behavior? Sure, it's still an A/B test, with the same goals, setup, and evaluation; however, the exact user psychology being measured is different.

What does this have to do with an interview question? Beyond being able to identify the basic ideas of A/B testing, being able to walk through imprecise questions is an asset to people working in analytics and data science.

------

### Q17. How to determine the influence of a Twitter user?

**Gregory Piatetsky answers:**

Social networks are at the center of today's web, and determining the influence in a social network is a huge area of research. Twitter influence is a narrow area within the overall social network influence research.

The influence of a Twitter user goes beyond the simple number of followers. We also want to examine how effective are tweets - how likely they are to be retweeted, favorited, or the links inside clicked upon. What exactly is an influential user depends on the definition - different types of influence discussed included celebrities, opinion leaders, influencers, discussers, innovators, topical experts, curators, commentators, and more.

A key challenge is to compute influence efficiently. An additional problem on Twitter is separating humans and bots.

Common measures used to quantify influence on Twitter include many versions of network centrality - how important is the node within the network, and PageRank-based metrics.

![NodeXL KDnuggets](https://www.kdnuggets.com/images/kdnuggets-nodexl-20140525-social-network-600.jpg)
[KDnuggets Twitter Social Network](https://www.kdnuggets.com/2014/05/kdnuggets-social-network-nodexl-may-2014.html), as visualized in NodeXL in May 2014.

Traditional network [measures](https://arxiv.org/pdf/1508.07951.pdf) used include

- Closeness Centrality, based on the length of the shortest paths from a node to everyone else. It measures the visibility or accessibility of each node with respect to the entire network
- Betweenness centrality considers for each node i all the shortest paths that should pass through i to connect all the other nodes in the network. It measures the ability of each node to facilitate communication within the network.

Other proposed measures include retweet impact (how likely is the tweet be retweeted) and variations of PageRank, such as TunkRank - see [A Twitter Analog to PageRank](http://thenoisychannel.com/2009/01/13/a-twitter-analog-to-pagerank).

An important refinement to overall influence is looking at influence within a topic - done by Agilience and RightRelevant. For instance, Justin Bieber may have high influence overall, but he is less influential than KDnuggets in the area of Data Science.

Twitter provides a [REST API](https://dev.twitter.com/) which allows access to key measures, but with limits on the number of requests and the data returned.

There were a number of websites that measured Twitter user influence, but many of their business models did not pan out, since many of them were acquired or went out of business. Ones which are currently active include the following:

**Free:**

- [Agilience](https://www.kdnuggets.com/tag/agilience) (KDnuggets is #1 in Machine Learning, #1 is Data Mining, #2 in Data Science)
- Klout, [klout.com](http://klout.com/)  (KDnuggets Klout score is 79)
- Influence Tracker, [www.influencetracker.com](http://www.influencetracker.com/) , KDnuggets influence metric 39.2
- [Right Relevance](http://www.rightrelevance.com/about) - measures specific relevance of twitter users within a topic.

**Paid:**

- Brandwatch (bought PeerIndex)
- Hubspot
- Simplymeasured

**Relevant KDnuggets posts:**

- [Agilience Top Data Mining, Data Science Authorities](https://www.kdnuggets.com/2016/11/agilience-top-data-mining-data-science-authorities.html)
- [12 Data Analytics Thought Leaders on Twitter](https://www.kdnuggets.com/2016/01/menlotechnologies-12-data-analytics-thought-leaders-twitter.html)
- [The 123 Most Influential People in Data Science](https://www.kdnuggets.com/2015/09/123-influential-people-data-science.html)
- [RightRelevance helps find key topics, top influencers in Big Data, Data Science, and Beyond](https://www.kdnuggets.com/2015/08/rightrelevance-topics-influencers-big-data-science-data-mining.html)

**Relevant KDnuggets tags:**

- [/tag/influencers](https://www.kdnuggets.com/tag/influencers)
- [/tag/big-data-influencers](https://www.kdnuggets.com/tag/big-data-influencers)

For a more in-depth analysis, see technical articles below:

- [What is a good measure of the influence of a Twitter user?](https://www.quora.com/What-is-a-good-measure-of-the-influence-of-a-Twitter-user), Quora
- [Measuring User Influence in Twitter: The Million Follower Fallacy](http://www.aaai.org/ocs/index.php/ICWSM/ICWSM10/paper/viewFile/1538/1826), AAAI, 2010
- [Measuring user influence on Twitter: A survey](https://arxiv.org/abs/1508.07951), arXiv, 2015
- [Measuring Influence on Twitter](http://www.l2f.inesc-id.pt/~fmmb/wiki/uploads/Work/misnis.ref07.pdf), I. Anger and C. Kittl
- [A Data Scientist Explains How To Maximize Your Influence On Twitter](http://www.businessinsider.com/how-to-maximize-your-influence-on-twitter-2014-1), Business Insider, 2014



###  How would you validate a model you created to generate a predictive model of a quantitative outcome variable using multiple regression.

Answer by 

**Matthew Mayo**.

 

Proposed methods

 for model validation: 

- If the values predicted by the model are far outside of the response variable range, this would immediately indicate poor estimation or model inaccuracy.
- If the values seem to be reasonable, examine the parameters; any of the following would indicate poor estimation or multi-collinearity: opposite signs of expectations, unusually large or small values, or observed inconsistency when the model is fed new data.
- Use the model for prediction by feeding it new data, and use the [coefficient of determination](https://en.wikipedia.org/wiki/Coefficient_of_determination) (R squared) as a model validity measure.
- Use data splitting to form a separate dataset for estimating model parameters, and another for validating predictions.
- Use [jackknife resampling](https://en.wikipedia.org/wiki/Jackknife_resampling) if the dataset contains a small number of instances, and measure validity with R squared and [mean squared error](https://en.wikipedia.org/wiki/Mean_squared_error) (MSE).

### Q4. Explain what precision and recall are. How do they relate to the ROC curve?

Answer by 

Gregory Piatetsky

: 

Here is the answer from 

KDnuggets FAQ: Precision and Recall

: 

Calculating precision and recall is actually quite easy. Imagine there are 100 positive cases among 10,000 cases. You want to predict which ones are positive, and you pick 200 to have a better chance of catching many of the 100 positive cases.  You record the IDs of your predictions, and when you get the actual results you sum up how many times you were right or wrong. There are four ways of being right or wrong:

1. **TN / True Negative: **case was negative and predicted negative
2. **TP / True Positive: **case was positive and predicted positive
3. **FN / False Negative: **case was positive but predicted negative
4. **FP / False Positive: **case was negative but predicted positive

Makes sense so far? Now you count how many of the 10,000 cases fall in each bucket, say:

|                    | **Predicted Negative** | **Predicted Positive** |
| ------------------ | ---------------------- | ---------------------- |
| **Negative Cases** | TN: 9,760              | FP: 140                |
| **Positive Cases** | FN: 40                 | TP: 60                 |

Now, your boss asks you three questions:

1. What percent of your predictions were correct? 
   You answer: the "accuracy" was (9,760+60) out of 10,000 = 98.2%
2. What percent of the positive cases did you catch? 
   You answer: the "recall" was 60 out of 100 = 60%
3. What percent of positive predictions were correct? 
   You answer: the "precision" was 60 out of 200 = 30%

See also a very good explanation of 

Precision and recall

 in Wikipedia. 

 

Fig 4: Precision and Recall

. 

ROC curve represents a relation between sensitivity (RECALL) and specificity(NOT PRECISION) and is commonly used to measure the performance of binary classifiers. However, when dealing with highly skewed datasets, 

Precision-Recall (PR) curves

 give a more representative picture of performance. See also this Quora answer: 

What is the difference between a ROC curve and a precision-recall curve?

. 

### Q5. How can you prove that one improvement you've brought to an algorithm is really an improvement over not doing anything?

Answer by 

**Anmol Rajpurohit**.

 

Often it is observed that in the pursuit of rapid innovation (aka "quick fame"), the principles of scientific methodology are violated leading to misleading innovations, i.e. appealing insights that are confirmed without rigorous validation. One such scenario is the case that given the task of improving an algorithm to yield better results, you might come with several ideas with potential for improvement. 

An obvious human urge is to announce these ideas ASAP and ask for their implementation. When asked for supporting data, often limited results are shared, which are very likely to be impacted by selection bias (known or unknown) or a misleading global minima (due to lack of appropriate variety in test data). 

Data scientists do not let their human emotions overrun their logical reasoning. While the exact approach to prove that one improvement you've brought to an algorithm is really an improvement over not doing anything would depend on the actual case at hand, there are a few common guidelines:

- Ensure that there is no selection bias in test data used for performance comparison
- Ensure that the test data has sufficient variety in order to be symbolic of real-life data (helps avoid overfitting)
- Ensure that "controlled experiment" principles are followed i.e. while comparing performance, the test environment (hardware, etc.) must be exactly the same while running original algorithm and new algorithm
- Ensure that the results are repeatable with near similar results
- Examine whether the results reflect local maxima/minima or global maxima/minima

 

One common way to achieve the above guidelines is through A/B testing, where both the versions of algorithm are kept running on similar environment for a considerably long time and real-life input data is randomly split between the two. This approach is particularly common in Web Analytics. 

- ### Q4. Explain what precision and recall are. How do they relate to the ROC curve?

  Answer by 

  Gregory Piatetsky

  : 

  Here is the answer from 

  KDnuggets FAQ: Precision and Recall

  : 

  Calculating precision and recall is actually quite easy. Imagine there are 100 positive cases among 10,000 cases. You want to predict which ones are positive, and you pick 200 to have a better chance of catching many of the 100 positive cases.  You record the IDs of your predictions, and when you get the actual results you sum up how many times you were right or wrong. There are four ways of being right or wrong:

  1. **TN / True Negative: **case was negative and predicted negative
  2. **TP / True Positive: **case was positive and predicted positive
  3. **FN / False Negative: **case was positive but predicted negative
  4. **FP / False Positive: **case was negative but predicted positive

  Makes sense so far? Now you count how many of the 10,000 cases fall in each bucket, say:

  |                    | **Predicted Negative** | **Predicted Positive** |
  | ------------------ | ---------------------- | ---------------------- |
  | **Negative Cases** | TN: 9,760              | FP: 140                |
  | **Positive Cases** | FN: 40                 | TP: 60                 |

  Now, your boss asks you three questions:

  1. What percent of your predictions were correct? 
     You answer: the "accuracy" was (9,760+60) out of 10,000 = 98.2%
  2. What percent of the positive cases did you catch? 
     You answer: the "recall" was 60 out of 100 = 60%
  3. What percent of positive predictions were correct? 
     You answer: the "precision" was 60 out of 200 = 30%

  See also a very good explanation of 

  Precision and recall

   in Wikipedia. 

   

  Fig 4: Precision and Recall

  . 

  ROC curve represents a relation between sensitivity (RECALL) and specificity(NOT PRECISION) and is commonly used to measure the performance of binary classifiers. However, when dealing with highly skewed datasets, 

  Precision-Recall (PR) curves

   give a more representative picture of performance. See also this Quora answer: 

  What is the difference between a ROC curve and a precision-recall curve?

  . 

  ### Q5. How can you prove that one improvement you've brought to an algorithm is really an improvement over not doing anything?

  Answer by 

  **Anmol Rajpurohit**.

   

  Often it is observed that in the pursuit of rapid innovation (aka "quick fame"), the principles of scientific methodology are violated leading to misleading innovations, i.e. appealing insights that are confirmed without rigorous validation. One such scenario is the case that given the task of improving an algorithm to yield better results, you might come with several ideas with potential for improvement. 

  An obvious human urge is to announce these ideas ASAP and ask for their implementation. When asked for supporting data, often limited results are shared, which are very likely to be impacted by selection bias (known or unknown) or a misleading global minima (due to lack of appropriate variety in test data). 

  Data scientists do not let their human emotions overrun their logical reasoning. While the exact approach to prove that one improvement you've brought to an algorithm is really an improvement over not doing anything would depend on the actual case at hand, there are a few common guidelines:

  - Ensure that there is no selection bias in test data used for performance comparison
  - Ensure that the test data has sufficient variety in order to be symbolic of real-life data (helps avoid overfitting)
  - Ensure that "controlled experiment" principles are followed i.e. while comparing performance, the test environment (hardware, etc.) must be exactly the same while running original algorithm and new algorithm
  - Ensure that the results are repeatable with near similar results
  - Examine whether the results reflect local maxima/minima or global maxima/minima

   

  One common way to achieve the above guidelines is through A/B testing, where both the versions of algorithm are kept running on similar environment for a considerably long time and real-life input data is randomly split between the two. This approach is particularly common in Web Analytics. 

  ### Explain what precision and recall are. How do they relate to the ROC curve?

  1. What percent of your predictions were correct? 
     You answer: the "accuracy" was (9,760+60) out of 10,000 = 98.2%
  2. What percent of the positive cases did you catch? 
     You answer: the "recall" was 60 out of 100 = 60%
  3. What percent of positive predictions were correct? 
     You answer: the "precision" was 60 out of 200 = 30%
  4. ​

  ROC curve represents a relation between sensitivity (RECALL) and specificity(NOT PRECISION) and is commonly used to measure the performance of binary classifiers. However, when dealing with highly skewed datasets, [Precision-Recall (PR) curves](http://pages.cs.wisc.edu/~jdavis/davisgoadrichcamera2.pdf) give a more representative picture of performance.

  There is a very important difference between what a ROC curve represents vs that of a PRECISION vs RECALL curve.

  Remember, a ROC curve represents a relation between sensitivity (RECALL) and specificity(NOT PRECISION). Sensitivity is the other name for recall but specificity is not PRECISION. 

  Recall/Sensitivity is the measure of the probability that your estimate is 1 given all the samples whose true class label is 1. It is a measure of how many of the positive samples have been identified as being positive.

  Specificity is the measure of the probability that your estimate is 0 given all the samples whose true class label is 0. It is a measure of how many of the negative samples have been identified as being negative.

  PRECISION on the other hand is different. It is a measure of the probability that a sample is a true positive class given that your classifier said it is positive. It is a measure of how many of the samples predicted by the classifier as positive is indeed positive. Note here that this changes when the base probability or prior probability of the positive class changes. Which means PRECISION depends on how rare is the positive class. In other words, it is used when positive class is more interesting than the negative class.

  So, if your problem involves kind of searching a needle in the haystack when for ex: the positive class samples are very rare compared to the negative classes, use a precision recall curve. Othwerwise use a ROC curve because a ROC curve remains the same regardless of the baseline prior probability of your positive class (the important rare class).

  ### Q6. What is root cause analysis?

  Answer by 

  Gregory Piatetsky

  According to Wikipedia, 

  > [Root cause analysis (RCA)](https://en.wikipedia.org/wiki/Root_cause_analysis) is a method of problem solving used for identifying the root causes of faults or problems. A factor is considered a root cause if removal thereof from the problem-fault-sequence prevents the final undesirable event from recurring; whereas a causal factor is one that affects an event's outcome, but is not a root cause.

  Root cause analysis was initially developed to analyze industrial accidents, but is now widely used in other areas, such as healthcare, project management, or software testing. 

  Here is a useful 

  Root Cause Analysis Toolkit from the state of Minnesota. 

  Essentially, you can find the root cause of a problem and show the relationship of causes by repeatedly asking the question, "Why?", until you find the root of the problem. This technique is commonly called "5 Whys", although is can be involve more or less than 5 questions. 

  ### 8. What is statistical power?

  Answer by 

  Gregory Piatetsky

  : 

  Wikipedia defines 

  Statistical power or sensitivity

   of a binary hypothesis test is the probability that the test correctly rejects the null hypothesis (H0) when the alternative hypothesis (H1) is true. 

  To put in another way, 

  Statistical power

   is the likelihood that a study will detect an effect when the effect is present. The higher the statistical power, the less likely you are to make a Type II error (concluding there is no effect when, in fact, there is).  

  ### 9. Explain what resampling methods are and why they are useful. Also explain their limitations.

  Answer by 

  Gregory Piatetsky

  : 

  Classical statistical parametric tests compare observed statistics to theoretical sampling distributions. Resampling a data-driven, not theory-driven methodology which is based upon repeated sampling within the same sample. 

  Resampling refers to methods for doing one of these

  - Estimating the precision of sample statistics (medians, variances, percentiles) by using subsets of available data (jackknifing) or drawing randomly with replacement from a set of data points (bootstrapping)
  - Exchanging labels on data points when performing significance tests (permutation tests, also called exact tests, randomization tests, or re-randomization tests)
  - Validating models by using random subsets (bootstrapping, cross validation)

### 10. Is it better to have too many false positives, or too many false negatives? Explain.

Answer by 

**Devendra Desale**

. 

It depends on the question as well as on the domain for which we are trying to solve the question. 

In medical testing, false negatives may provide a falsely reassuring message to patients and physicians that disease is absent, when it is actually present. This sometimes leads to inappropriate or inadequate treatment of both the patient and their disease. So, it is desired to have too many false positive. 

For spam filtering, a false positive occurs when spam filtering or spam blocking techniques wrongly classify a legitimate email message as spam and, as a result, interferes with its delivery. While most anti-spam tactics can block or filter a high percentage of unwanted emails, doing so without creating significant false-positive results is a much more demanding task. So, we prefer too many false negatives over many false positives. 

### 11. What is selection bias, why is it important and how can you avoid it?

Answer by 

**Matthew Mayo**

. 

Selection bias, in general, is a problematic situation in which error is introduced due to a non-random population sample. For example, if a given sample of 100 test cases was made up of a 60/20/15/5 split of 4 classes which actually occurred in relatively equal numbers in the population, then a given model may make the false assumption that probability could be the determining predictive factor. Avoiding non-random samples is the best way to deal with bias; however, when this is impractical, techniques such as 

resampling

, 

boosting, and weighting are strategies which can be introduced to help deal with the situation. ]

### Bonus Question: Explain what is overfitting and how would you control for it

This question was not part of the original 20, but probably is the most important one in distinguishing real data scientists from fake ones. 

Answer by [Gregory Piatetsky](https://www.kdnuggets.com/author/gregory-piatetsky).

 

Overfitting is finding spurious results that are due to chance and cannot be reproduced by subsequent studies. 

We frequently see newspaper reports about studies that overturn the previous findings, like eggs are no longer bad for your health, or 

saturated fat is not linked to heart disease

. The problem, in our opinion is that many researchers, especially in social sciences or medicine, too frequently commit the cardinal sin of Data Mining - 

Overfitting the data.

 

The researchers test too many hypotheses without proper statistical control, until they happen to find something interesting and report it.  Not surprisingly, next time the effect, which was (at least partly) due to chance, will be much smaller or absent. 

These flaws of research practices were identified and reported by John P. A. Ioannidis in his landmark paper 

*Why Most Published Research Findings Are False*

 (PLoS Medicine, 2005). Ioannidis found that very often either the results were exaggerated or the findings could not be replicated. In his paper, he presented statistical evidence that indeed most claimed research findings are false. 

Ioannidis noted that in order for a research finding to be reliable, it should have:

- Large sample size and with large effects
- Greater number of and lesser selection of tested relationship
- Greater flexibility in designs, definitions, outcomes, and analytical modes
- Minimal bias due to financial and other factors (including popularity of that scientific field)

 

Unfortunately, too often these rules were violated, producing irreproducible results. For example, S&P 500 index was found to be strongly related to Production of butter in Bangladesh (from 19891 to 1993) (

here is PDF

) 

 

See more interesting (and totally spurious) findings which you can discover yourself using tools such as 

Google correlate

 or 

Spurious correlations

 by Tyler Vigen. 

Several methods can be used to avoid "overfitting" the data

- Try to find the simplest possible hypothesis
- [Regularization](http://en.wikipedia.org/wiki/Regularization_(mathematics)) (adding a penalty for complexity)
- Randomization Testing (randomize the class variable, try your method on this data - if it find the same strong results, something is wrong)
- Nested cross-validation  (do feature selection on one level, then run entire method in cross-validation on outer level)
- Adjusting the [False Discovery Rate](http://en.wikipedia.org/wiki/False_discovery_rate)
- Using the [reusable holdout method](https://www.kdnuggets.com/2015/08/feldman-avoid-overfitting-holdout-adaptive-data-analysis.html) - a breakthrough approach proposed in 2015

 

Good data science is on the leading edge of scientific understanding of the world, and it is data scientists responsibility to avoid overfitting data and educate the public and the media on the dangers of bad data analysis. 

See also

- [The Cardinal Sin of Data Mining and Data Science: Overfitting](https://www.kdnuggets.com/2014/06/cardinal-sin-data-mining-data-science.html)
- [Big Idea To Avoid Overfitting: Reusable Holdout to Preserve Validity in Adaptive Data Analysis](https://www.kdnuggets.com/2015/08/feldman-avoid-overfitting-holdout-adaptive-data-analysis.html)
- [Overcoming Overfitting with the reusable holdout: Preserving validity in adaptive data analysis](https://www.kdnuggets.com/2015/08/reusable-holdout-preserving-validity-adaptive-data-analysis.html)
- [11 Clever Methods of Overfitting and how to avoid them](https://www.kdnuggets.com/2015/01/clever-methods-overfitting-avoid.html)
- [Tag: Overfitting](https://www.kdnuggets.com/tag/overfitting)

 

------

### Q12. Give an example of how you would use experimental design to answer a question about user behavior.

Answer by 

**Bhavya Geethika**

. 

Step 1: Formulate the Research Question: 

What are the effects of page load times on user satisfaction ratings? 

Step 2: Identify variables: 

We identify the cause & effect. Independent variable -page load time, Dependent variable- user satisfaction rating 

Step 3: Generate Hypothesis: 

Lower page download time will have more effect on the user satisfaction rating for a web page. Here the factor we analyze is page load time. 

 

Fig 12: There is a flaw in your experimental design (cartoon from 

here

) 

Step 4: Determine Experimental Design. 

We consider experimental complexity i.e vary one factor at a time or multiple factors at one time in which case we use factorial design (2^k design). A design is also selected based on the type of objective (Comparative, Screening, Response surface) & number of factors. 

Here we also identify within-participants, between-participants, and mixed model.For e.g.: There are two versions of a page, one with Buy button (call to action) on left and the other version has this button on the right. 

Within-participants design - both user groups see both versions. 

Between-participants design - one group of users see version A & the other user group version B. 

Step 5: Develop experimental task & procedure: 

Detailed description of steps involved in the experiment, tools used to measure user behavior, goals and success metrics should be defined. Collect qualitative data about user engagement to allow statistical analysis. 

Step 6: Determine Manipulation & Measurements 

Manipulation: One level of factor will be controlled and the other will be manipulated. We also identify the behavioral measures:

1. Latency- time between a prompt and occurrence of behavior (how long it takes for a user to click buy after being presented with products).
2. Frequency- number of times a behavior occurs (number of times the user clicks on a given page within a time)
3. Duration-length of time a specific behavior lasts(time taken to add all products)
4. Intensity-force with which a behavior occurs ( how quickly the user purchased a product)

Step 7: Analyze results

 

Identify user behavior data and support the hypothesis or contradict according to the observations made for e.g. how majority of users satisfaction ratings compared with page load times. 

------

### Q13. What is the difference between "long" ("tall") and "wide" format data?

Answer by [Gregory Piatetsky](https://www.kdnuggets.com/author/gregory-piatetsky).

 

In most data mining / data science applications there are many more records (rows) than features (columns) - such data is sometimes called "tall" (or "long") data. 

In some applications like genomics or bioinformatics you may have only a small number of records (patients), eg 100, but perhaps 20,000 observations for each patient. The standard methods that work for "tall" data will lead to overfitting the data, so special approaches are needed. 

 

Fig 13. Different approaches for tall data and wide data

, from presentation 

Sparse Screening for Exact Data Reduction

, by Jieping Ye. 

The problem is not just reshaping the data (here there are 

useful R packages

), but avoiding false positives by reducing the number of features to find most relevant ones. 

### 16. How would you screen for outliers and what should you do if you find one?

Answer by 

**Bhavya Geethika**

. 

Some methods to screen outliers are z-scores, modified z-score, box plots, Grubb's test, Tietjen-Moore test exponential smoothing, Kimber test for exponential distribution and moving window filter algorithm. However two of the robust methods in detail are: 

Inter Quartile Range

 

An outlier is a point of data that lies over 1.5 IQRs below the first quartile (Q1) or above third quartile (Q3) in a given data set.

- High = (Q3) + 1.5 IQR
- Low = (Q1) - 1.5 IQR

 

Tukey Method

 

It uses interquartile range to filter very large or very small numbers. It is practically the same method as above except that it uses the concept of "fences". The two values of fences are:

- Low outliers = Q1 - 1.5(Q3 - Q1) = Q1 - 1.5(IQR)
- High outliers = Q3 + 1.5(Q3 - Q1) = Q3 + 1.5(IQR)

 

Anything outside of the fences is an outlier. 

When you find outliers, you should not remove it without a qualitative assessment because that way you are altering the data and making it no longer pure. It is important to understand the context of analysis or importantly "The Why question - Why an outlier is different from other data points?" 

This reason is critical. If outliers are attributed to error, you may throw it out but if they signify a new trend, pattern or reveal a valuable insight into the data you should retain it. 



### Q17. How would you use either the extreme value theory, Monte Carlo simulations or mathematical statistics (or anything else) to correctly estimate the chance of a very rare event?

Answer by 

**Matthew Mayo**.

 

Extreme value theory

 (EVT) focuses on rare events and extremes, as opposed to classical approaches to statistics which concentrate on average behaviors. EVT states that there are 

3 types of distributions needed to model the the extreme data points of a collection of random observations from some distribution

: the Gumble, Frechet, and Weibull distributions, also known as the Extreme Value Distributions (EVD) 1, 2, and 3, respectively. 

The EVT states that, if you were to generate N data sets from a given distribution, and then create a new dataset containing only the maximum values of these N data sets, this new dataset would only be accurately described by one of the EVD distributions: Gumbel, Frechet, or Weibull. The Generalized Extreme Value Distribution (GEV) is, then, a model combining the 3 EVT models as well as the EVD model. 

Knowing the models to use for modeling our data, we can then use the models to fit our data, and then evaluate. Once the best fitting model is found, analysis can be performed, including calculating possibilities. 

------

### 18. What is a recommendation engine? How does it work?

Answer by [Gregory Piatetsky](https://www.kdnuggets.com/author/gregory-piatetsky):

 

We are all familiar now with recommendations from Netflix - "Other Movies you might enjoy" or from Amazon - Customers who bought X also bought Y., 

Such systems are called recommendation engines or more broadly recommender systems. 

They typically produce recommendations in one of two ways: using 

collaborative

 or 

content-based

filtering. 

**Collaborative filtering **

methods build a model based on users past behavior (items previously purchased, movies viewed and rated, etc) and use decisions made by current and other users. This model is then used to predict items (or ratings for items) that the user may be interested in. 

**Content-based filtering**

 methods use features of an item to recommend additional items with similar properties. These approaches are often combined in Hybrid Recommender Systems. 

Here is a comparison of these 2 approaches used in two popular music recommender systems - Last.fm and Pandora Radio. (example from 

Recommender System

 entry)

- Last.fm creates a "station" of recommended songs by observing what bands and individual tracks the user has listened to on a regular basis and comparing those against the listening behavior of other users. Last.fm will play tracks that do not appear in the user's library, but are often played by other users with similar interests. As this approach leverages the behavior of users, it is an example of a collaborative filtering technique.
- Pandora uses the properties of a song or artist (a subset of the 400 attributes provided by the Music Genome Project) in order to seed a "station" that plays music with similar properties. User feedback is used to refine the station's results, deemphasizing certain attributes when a user "dislikes" a particular song and emphasizing other attributes when a user "likes" a song. This is an example of a content-based approach.

 

Here is a good 

Introduction to Recommendation Engines

 by Dataconomy and an overview of 

building a Collaborative Filtering Recommendation Engine

 by Toptal. For latest research on recommender systems, check 

ACM RecSys conference

. 

------

### 19. Explain what a false positive and a false negative are. Why is it important to differentiate these from each other?

Answer by [Gregory Piatetsky](https://www.kdnuggets.com/author/gregory-piatetsky):

 

In binary classification (or medical testing), False positive is when an algorithm (or test) indicates presence of a condition, when in reality it is absent. A false negative is when an algorithm (or test) indicates absence of a condition, when in reality it is present. 

In statistical hypothesis testing false positive is also called type I error and false negative - type II error. 

It is obviously very important to distinguish and treat false positives and false negatives differently because the costs of such errors can be hugely different. 

For example, if a test for serious disease is false positive (test says disease, but person is healthy), then an extra test will be made that will determine the correct diagnosis. However, if a test is false negative (test says healthy, but person has disease), then treatment will be done and person may die as a result. 

------

### 20. Which tools do you use for visualization? What do you think of Tableau? R? SAS? (for graphs). How to efficiently represent 5 dimension in a chart (or in a video)?

Answer by [Gregory Piatetsky](https://www.kdnuggets.com/author/gregory-piatetsky):

 

There are many good tools for Data Visualization. R, Python, Tableau and Excel are among most commonly used by Data Scientists. 

Here are useful KDnuggets resources:

- [Visualization and Data Mining Software](https://www.kdnuggets.com/software/visualization.html)
- [Overview of Python Visualization Tools](https://www.kdnuggets.com/2015/11/overview-python-visualization-tools.html)
- [21 Essential Data Visualization Tools](https://www.kdnuggets.com/2015/05/21-essential-data-visualization-tools.html)
- [Top 30 Social Network Analysis and Visualization Tools](https://www.kdnuggets.com/2015/06/top-30-social-network-analysis-visualization-tools.html)
- [Tag: Data Visualization](https://www.kdnuggets.com/tag/data-visualization)

 

There are many ways to representing more than 2 dimensions in a chart. 3rd dimension can be shown with a 3D scatter plot which can be rotate. You can use color, shading, shape, size. Animation can be used effectively to show time dimension (change over time). 

Here is a good example. 

 

Fig 20a: 5-dimensional scatter plot of Iris data

, with size: sepal length; color: sepal width; shape: class; x-column: petal length; y-column: petal width, from 

here

. 

For more than 5 dimensions, one approach is 

Parallel Coordinates

, pioneered by Alfred Inselberg. 

 

Fig 20b: Iris data in parallel coordinates

 

See also

- [Quora: What's the best way to visualize high-dimensional data?](https://www.quora.com/Whats-the-best-way-to-visualize-high-dimensional-data) and
- pioneering work of Georges Grinstein and his colleagues on [High-Dimensional Visualizations ](http://www.cs.uml.edu/~mtrutsch/research/High-Dimensional_Visualizations-KDD2001-color.pdf).