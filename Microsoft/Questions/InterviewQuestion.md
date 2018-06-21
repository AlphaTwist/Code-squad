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
17. How do you detect if a new observation is outlier? What is bias-variance trade off ? 
18. Describe how gradient boost works
19. Data Mining Explain what heteroskedasticity is and how to solve it
20. Define and explain the differences between clustered and non-clustered indexes.




- **What are the different ways to return the rowcount of a table?**

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

1. How would you improve ETL (Extract, Transform, Load) throughput?
2. What is a neural network?
3. How do you deal with unbalanced binary classification?
4. What’s the difference between L1 and L2 regularization?
5. Can you explain what REST is?
6. What are the steps for wrangling and cleaning data before applying machine learning algorithms?
7. How do you measure distance between data points?
8. Define variance.
9. Describe the differences between and use cases for box plots and histograms



