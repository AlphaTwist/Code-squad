# ML/DS Question
- Describe for me in detail the difference between L1 and L2 regularization, specifically as regards the difference in their 
impact on the model training process itself
- what's bias variance tradeoff
- How is XGBoost handling bias-variance tradeoff?
- what is random forest? why is naive bayes good?  
- We do pre-screening on the data to remove fraud threats — so how do we find a data sample that we can use to determine a real representation of fraud events?
- Suppose you have 100,000 files spread across multiple servers and you wanted to process all of them? How would you do that in Hadoop?
- How would you transfer data from one Hadoop cluster to another?
- How can you handle the daily tedious tasks that go hand in hand with processing metadata for hundreds of titles?
- In terms of data flow and accessibility, how do you measure success in a hidden time frame where the nucleus overloads the border structure of the over complicated file system that redirects computer energy to the cellar dome?
- 
- How do you take millions of users with 100's of transactions each, amongst 10k's of products and group the users together 
in a meaningful segments? 
[link1](https://www.glassdoor.com/Interview/How-do-you-take-millions-of-users-with-100-s-of-transactions-each-amongst-10k-s-of-products-and-group-the-users-together-i-QTN_548791.htm) 
- You have time series of sensors, predict the next reading
- What is the difference between Python and Scala?
- Explain LRU Cache.
- How would you design a client — server model where the client sends location data every minute?
- What are different types of memories in Java?
- What is your experience with psychophysical experiments?(Research Portfolio based question)
- What is your expertise in characterization? What do you usually use that for? How did you use that in your research and find interesting results?
- How do you deal with failure analysis?
- How many iPhones are sold in US every year?
- A birthday cake needs to be cut into 8 equal pieces with just 3 cuts. How can you do this?
- Explain about the significance of selection bias
- What are the limitations resampling methods bring in?
- How can you create a logistic regression model through ANN?
- Which command is used to create a Histogram visualization in R language?
- How to deal with unbalanced data where the ratio of positive and negative is huge?
- how to deal with missing data
- Given an iTunes type of app that pulls down lots of images that get stale over time, what strategy would you use to flush disused images over time?
- how to detect outlier
- how do you handle big data sets
- assumptions of liner regression


# [Probability](https://www.wallstreetoasis.com/forums/hardest-probability-and-statistics-interview-questions)
- If it rains on Saturday with probability 45% and it rains on Sunday with probability 15%, what is the probability that it will rain this weekend?
- You are given 7 marbles, 6 of which weigh exactly the same, but one marble weighs less than the other 6. You also have a weighing scale. Determine the minimum number of attempts to find the lightest marble.
- you have 100 coins laying flat on a table each with a head side and a tail side. 10 of them are heads up, 90 are tails up. you can't feel, see or in any other way find out which side is up. Split the coins into two piles such taht there are the same number of heads in each pile.
- How many children are born every day
- If you have 2 eggs, and you want to figure out what's the highest floor from which you can drop the egg without breaking it, how would you do it? What's the optimal solution?
- If you're given a jar with a mix of fair and unfair coins, and you pull one out and flip it 3 times, and get the specific sequence heads heads tails, what are the chances that you pulled out a fair or an unfair coin?
- There are three boxes, one contains only apples, one contains only oranges, and one contains both apples and oranges. The boxes have been incorrectly labeled such that no label identifies the actual contents of the box it labels. Opening just one box, and without looking in the box, you take out one piece of fruit. By looking at the fruit, how can you immediately label all of the boxes correctly

# sql and db
- Create market basket output using SQL.
- Given a table with 1B of user ID and product IDs that the users bought, and another table with product ID mapped with product name. We are trying to find the paired products that are often purchased together by the same user, such as wine and bottle opener, chips and beer. How to find the top 100 of these co-existed pairs of products?
  Assuming that Every line in the input data contains user-id and list of product ids. 
In the map phase, we will first extract all products purchased by a user and pair them up with the count.
e.g. CUST_123, PROD_1, PROD2, PROD3

result of map phase.
(PROD_1:PROD_2,1)
(PROD_1:PROD_3,1)
(PROD_2:PROD_3,1)

In the reduce phase, we will collect all such results from all users and then add all counts and then return top 100.
- Someone put distribute Random()*ID in a Hive script to prevent data skew. What would be the problem here?
  Problem here is , same id will get different partition number if using Random()*ID and hence will go to different reducers. Aggregation functions based on ID will result in incorrect results.


# Programming Python
- Check if a binary tree is a mirror image on left and right sub-trees.
- Check word is pallindrome, string is pallindrome
- find anagrams in a list and print out list of anagrams.
- fibonacci sequence, recursive and iterative model with complexity analysis 
- Set cover problem 
  [Wikipedia](https://en.wikipedia.org/wiki/Set_cover_problem) [geekforgeek](https://www.geeksforgeeks.org/set-cover-problem-set-1-greedy-approximate-algorithm/)

# HR 
- Why Apple
- tell me something that you have done in your life which you are particularly pround of
- what are your failures and how have you learned from them
- describe an interesting problem and how you solved it
- why do you want to jon Apple and what will you miss at your current work iof apple hired you.
- What do you like to do in next 5 years.
- What is your life so far?
- What do you like to do?
- describe yourself what exccites you
- how would you test av taoster
- Why should we hire you?
- Are you creative? What's something creative that you can think of?
- If you could have one superpower, what would it be?
- Have you ever disagreed with a manager's decision, and how did you approach the disagreement? Give a specific example and explain how you rectified this disagreement, what the final outcome was, and how that individual would describe you today.
- 
