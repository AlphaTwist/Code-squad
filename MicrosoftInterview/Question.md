## Question
- Overfitting, l1 l2 and elastic regularization
- bayes theorem and NBC
- logits model
- How do you detect if a new observation is outlier? What is bias-variance trade off ?
  - define variance and bias
- Can you explain the Naive Bayes fundamentals? How did you set the threshold?
- Can you explain what MapReduce is and how it works?  
- Three friends in Seattle told you it's rainy. Each has a probability of 1/3 of lying. What's the probability of Seattle is rainy. 
- Experiment Design, A/B testing  
- PCA
- RNN and LSTM
- RNN seq to seq model
- word2vec
- confusion matrix, ROC, Sensivity, specificity
- There are 6 marbles in a bag - 1 is white. You reach in the bag 100 times. After drawing a marble, it is placed back in the bag. 
  What is the probability of drawing the white marble at least once? 
- How to compute an inverse matrix faster by playing around with some computational tricks?  
- Difference between box plot and histogram
- How to measure distance between data point?  
- Describe how gradient boost works
- Design and optimize an elevator question
- Generate a fair coin from a biased one.
- What is a good/bad Data visualisation ?  
- 

## Program
- Find the maximum of sub sequence in an integer list. 
- Given a infinite list, how can you find and then remove the second to last element in the list?
```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None
class Solution:
    def removeNthFromEnd(self, head, n):
        """
        :type head: ListNode
        :type n: int set n = 2
        :rtype: ListNode
        """
        dummy = ListNode(0)
        dummy.next = head
        first = second = dummy
        
        #Advance first pointer so that the gap between first and secons is n nodes aprat
        for i in range(n): first = first.next
        while first.next :
            first = first.next
            second = second.next
            
        second.next = second.next.next
        return dummy.next
```
- convert binary search tree to sorted doubly linked list. not doubly
```python
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    #Recursion
    head = None
    def flatten(self, root):
        """
        :type root: TreeNode
        :rtype: void Do not return anything, modify root in-place instead.
        """
        if root:
            self.flatten(root.left)
            self.flatten(root.right)
            if root.left:
                p = root.left
                while p.right:
                    p = p.right
                p.right = root.right
                root.right = root.left
            root.left = None
```
- convert sorted linked list to BST
```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
# recursively
    def sortedListToBST(self, head):
        if not head:
            return 
        if not head.next:
            return TreeNode(head.val)
        # here we get the middle point,
        # even case, like '1234', slow points to '2',
        # '3' is root, '12' belongs to left, '4' is right
        # odd case, like '12345', slow points to '2', '12'
        # belongs to left, '3' is root, '45' belongs to right
        slow, fast = head, head.next.next
        while fast and fast.next:
            fast = fast.next.next
            slow = slow.next
        # tmp points to root
        tmp = slow.next
        # cut down the left child
        slow.next = None
        root = TreeNode(tmp.val)
        root.left = self.sortedListToBST(head)
        root.right = self.sortedListToBST(tmp.next)
        return root
```
- Create a function that checks if a word is a palindrome  
- Find max sum subsequence.
