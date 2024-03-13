# import
from typing import List

# The Below Section is of Top Interview 150
# 1. Merge Sorted Array (easy)
# class Solution:
#     def merge(nums1: List[int], m: int, nums2: List[int], n: int) -> None:
#         """
#         Do not return anything, modify nums1 in-place instead.
#         """
#         # approach, create a new list with value of nums1 called nums3
#         # find the max int and put it in the back of nums1
#         nums3 = nums1[:m]
#         i=0
#         while i < n + m and nums2 and nums3:
#             print(nums1)
#             print(nums2)
#             print(nums3)
#             print(i)
#             if nums3[-1] > nums2[-1]:
#                 nums1[n + m - i - 1] = nums3.pop()
#                 print('result3: ')
#                 print(nums1)
#             else:
#                 nums1[n + m - i - 1] = nums2.pop()
#                 print('result2: ')
#                 print(nums1)
#             i += 1
#             print(' ')

#         # if one of the list was not exhausted
#         # use it to fill in the rest of the empty space in nums1
#         if nums2:
#             nums1[:len(nums2)]=nums2
#         elif nums3:
#             nums1[:len(nums3)]=nums3

#         print('final')
#         print(nums1)
#         print('')

#         return
# test code
# Solution.merge([1,2,3,0,0,0],3,[2,5,6],3)
# Solution.merge([2,0],1,[1],1)
# Solution.merge([1,2,4,5,6,0],5,[3],1)
    
# cheat solution: merge the arrays then use native sort
# for j in range(n):
#         nums1[m+j] = nums2[j]
#     nums1.sort()

# best solution: similar to mine but using pointers
# class Solution(object):
#     def merge(self, nums1, m, nums2, n):
#         i = m - 1 # pointer for nums1
#         j = n - 1 # pointer for nums2
#         k = m + n - 1 # pointer for number in nums1 to be changed
        
#         while j >= 0:
#             # while nums 2 exist
#             if i >= 0 and nums1[i] > nums2[j]:
#                 # if nums1 numbers are still there
#                 # and is higher, substitute it in
#                 nums1[k] = nums1[i]
#                 i -= 1
#             else:
#                 # if nums1 numbers is gone or lower
#                 # substitues in nums2 number
#                 nums1[k] = nums2[j]
#                 j -= 1
#             k -= 1

# break break break

# 2. Remove Element (easy)
class Solution:
    def removeElement(nums: List[int], val: int) -> int:
        i = 0
        j = len(nums)-1
        k = 0
        t = 0

        if i == j:
            if nums[i] == val:
                return 0
            else:
                return 1

        while i < j:
            while nums[j] == val:
                j -= 1
                t = nums[j]

            if nums[i]==val:
                nums[j] = val
                nums[i] = t
            i += 1
            k += 1
        
        print('final')
        print(nums)
        print(k)
        return k
# test code
Solution.removeElement([3,2,2,3],3)
Solution.removeElement([0,1,2,2,3,0,4,2],2)
Solution.removeElement([2],3)
