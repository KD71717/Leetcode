# import
from collections import defaultdict
import sys
from typing import List
from itertools import zip_longest 

# 80. remove duplicates from sorted array 2, 41 67
# class Solution:
#     def removeDuplicates(self, nums: List[int]) -> int:
#         c = 1 # count of current number
#         n = nums[0] # current number
#         j = 1 # last number that was empty spot

#         for i in range(1, len(nums)):
#             if nums[i] != n:
#                 nums[j] = nums[i]
#                 n = nums[i]
#                 j += 1
#                 c = 1
#             elif c == 1:
#                 nums[j] = nums[i]
#                 j += 1
#                 c += 1
#         return j
# # best solution 1, 15 67
# class Solution:
#     def removeDuplicates(self, nums: List[int]) -> int:
#         k = 0
#         for i in nums:
#             if k < 2 or i != nums[k - 2]:
#                 nums[k] = i
#                 k += 1
#         return k   
# # best solution 2, similar to above but use range instead of iterating over nums
# # 48 95 
# class Solution:
#     def removeDuplicates(self, nums: List[int]) -> int:
#         k = 1  # Start from 1 because the first element is always unique
#         n = len(nums)
#         for i in range(1, n):
#             if k==1 or  nums[i] != nums[k - 2]:
#                 nums[k] = nums[i]
#                 k += 1
#         return k
# # another attempt 60, 67
# class Solution:
#     def removeDuplicates(self, nums: List[int]) -> int:
#         n = len(nums)
#         if n < 3:
#             return n
#         k = 2
#         for i in range(2, n):
#             if nums[i] != nums[k - 2]:
#                 nums[k] = nums[i]
#                 k += 1
#         return k

# 189. rotate array, 29 62, this creates extra space (length of nums)
class Solution:
    def rotate(nums: List[int], k: int) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        l = len(nums)
        x = nums[-k%l:]+nums[:-k%l]
        for i, e in enumerate(x):
            nums[i] = e
    
# test 
x = [1,2,3,4,5,6,7] # [5,6,7,1,2,3,4]
y = [-1,-100,3,99] # [3,99,-1,-100]
z = [1,2] # [2,1]
Solution.rotate(x, 3)
print(x)
Solution.rotate(y, 2)
print(y)
Solution.rotate(z, 3)
print(z)