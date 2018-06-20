https://www.kdnuggets.com/2017/02/17-data-science-interview-questions-answers.html
https://www.kdnuggets.com/2017/02/17-data-science-interview-questions-answers-part-2.html
https://www.kdnuggets.com/2017/03/17-data-science-interview-questions-answers-part-3.html
1. Model evaluation and choice of evaluation metrics used in accenture and with Dr. Zielke
2. How would you deal with a categorical feature with high cardinality?
3. Describe a machine learning project you're proud of
4. Pick a machine learning algorithm of your choice and describe it to me
5. Describe an ROC chart
6. Describe a decision tree
7. *Generate a fair coin from a biased one.*
This is originally von Neumann’s clever idea. If we have a biased coin (i.e. a coin that comes up heads with probability different from 1/2), we can simulate a fair coin by tossing pairs of coins until the two results are different. Given that we have different results, the probability that the first is “heads” and the second is “tails” is the same as the probability of “tails” then “heads”. So if we simply return the value of the first coin, we will get “heads” or “tails” with the same probability, i.e. 1/2.

One might wonder how many calls to biasedCoin we expect to make before the function returns. One can recognize the experiment as a geometric distribution and use the known expected value, but it is short so here is a proof. Let s be the probability of seeing two different outcomes in the biased coin flip, and t the expected number of trials until that happens. If after two flips we see the same outcome (HH or TT), then by independence the expected number of flips we need is unchanged. Hence

t = 2s + (1-s)(2 + t)

Simplifying gives t = 2/s, and since we know s = 2p(1-p) we expect to flip the coin \frac{1}{p(1-p)} times.

8. Generate 7 integers with equal probability from a function which returns 1/0 with probability p and (1-p)
9. What are the ROC curve and the meaning of sensitivity, specificity, confusion matrix  
11. How to best select a representative sample of search quieres from 5 miliion serach queries
12. Experiment Design, A/B testing
13. Ask random-forest and lasso 
14. How to builds ads model, basic algorithms.  
15. Can you explain the Naive Bayes fundamentals? How did you set the threshold?
16. Can you explain what MapReduce is and how it works?  
17. Can you explain SVM to me?
18. How do you detect if a new observation is outlier? What is bias-variance trade off ? 
19. Describe how gradient boost works
20. Data Mining Explain what heteroskedasticity is and how to solve it
21. Define and explain the differences between clustered and non-clustered indexes.
22. What are the different ways to return the rowcount of a table?
23. How would you improve ETL (Extract, Transform, Load) throughput?
24. What is a neural network?
25. How do you deal with unbalanced binary classification?
26. What’s the difference between L1 and L2 regularization?
27. Can you explain what REST is?
28. What are the steps for wrangling and cleaning data before applying machine learning algorithms?
29. How do you measure distance between data points?
30. Define variance.
31. Describe the differences between and use cases for box plots and histograms



