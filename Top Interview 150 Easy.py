# import
from collections import defaultdict
import sys
from typing import List
from itertools import zip_longest 

# 88. Merge Sorted Array (easy)
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

# 27. Remove Element (easy)
# class Solution:
#     def removeElement(nums: List[int], val: int) -> int:
#         # From hint, consider the elements to be removed as non-existent. In a single pass, if we keep copying the visible elements in-place, that should also solve this problem for us.
#         # loop through nums, add 1 to k if value=val, and copy the value to empty spot (occupied by non-existent)
#         i = 0 # pointer for last empty spot
#         j = 0 # loop variable
#         k = 0 # record

#         while j < len(nums):
#             if nums[j]!=val:
#                 nums[i]=nums[j]
#                 k+=1
#                 i+=1
#             j += 1
#         print(nums)
#         print(k)
#         return k
# # test code
# Solution.removeElement([3,2,2,3],3)
# Solution.removeElement([0,1,2,2,3,0,4,2],2)
# Solution.removeElement([2],3)
# Solution.removeElement([3,3],3)
# Solution.removeElement([3,3],1)
# best solution, similar to mine but removed a few uneeded operations using for loop
# and that i did not have to keep track of k since index kept track of it
# class Solution:
#     def removeElement(self, nums: List[int], val: int) -> int:
#         index = 0 # keep track of empty spot
#         for i in range(len(nums)):
#             if nums[i] != val:
#                 nums[index] = nums[i]
#                 index += 1
#         return index

# break break break

# 26. Remove Duplicates From Sorted Array
# similar to above
# class Solution:
#     def removeDuplicates(nums: List[int]) -> int:
#         index = 0 # for the last unique number

#         for i in range(1,len(nums)):
#             if nums[i]!=nums[index]:
#                 nums[index+1]=nums[i]
#                 index+=1
#         return index+1
# # test code
# Solution.removeDuplicates([1,1,2])
# Solution.removeDuplicates([0,0,1,1,1,2,2,3,3,4])
# # best solution is similar
# class Solution:
#     def removeDuplicates(self, nums: List[int]) -> int:
#         j = 1
#         for i in range(1, len(nums)):
#             if nums[i] != nums[i - 1]:
#                 nums[j] = nums[i]
#                 j += 1
#         return j

# 88. Majority Element
# class Solution:
#     def majorityElement(nums: List[int]) -> int:
#         nums.sort()
#         print(nums[len(nums)//2])
#         return nums[len(nums)//2]
# Solution.majorityElement([3,2,3])
# Solution.majorityElement([2,2,1,1,1,2,2])
# # best solution 1
# # The intuition behind using a hash map is to count the occurrences 
# # of each element in the array and then identify the element that 
# # occurs more than n/2 times. By storing the counts in a hash map, 
# # we can efficiently keep track of the occurrences of each element.
# class Solution:
#     def majorityElement(self, nums: List[int]) -> int:
#         n = len(nums)
#         m = defaultdict(int)
#         for num in nums:
#             m[num] += 1
#         n = n // 2
#         for key, value in m.items():
#             if value > n:
#                 return key
#         return 0
# # best solution 2
# # The intuition behind the Moore's Voting Algorithm is based on 
# # the fact that if there is a majority element in an array, it 
# # will always remain in the lead, even after encountering other elements.
# class Solution:
#     def majorityElement(self, nums: List[int]) -> int:
#         count = 0
#         candidate = 0
#         for num in nums:
#             if count == 0:
#                 candidate = num
#             if num == candidate:
#                 count += 1
#             else:
#                 count -= 1
#         return candidate

# # 121. Best Time to Buy and Sell Stock
# class Solution:
#     def maxProfit(prices: List[int]) -> int:
#         front = prices[0]
#         curmax = 0
#         for p in prices:
#             if p < front:
#                 front = p
#             elif p > front:
#                 curmax = max(curmax, p - front)
#         print(curmax)
#         return curmax
# # test code
# Solution.maxProfit([7,1,5,3,6,4])
# Solution.maxProfit([7,6,4,3,1])
# # best solution 1
# # similar to mine but less operations
# class Solution:
#     def maxProfit(self, prices: List[int]) -> int:
#         min_price = prices[0]
#         max_profit = 0
#         for price in prices[1:]:
#             max_profit = max(max_profit, price - min_price)
#             min_price = min(min_price, price)
#         return max_profit
    
# 13. Roman to Integer
# class Solution:
#     def romanToInt(s: str) -> int:
#         d={"I":1,"V":5,"X":10,"L":50,"C":100,"D":500,"M":1000}
#         v = 0
#         for c in s:
#             v += d[c]
#         v = v-s.count("IV")*2-s.count("IX")*2-s.count("XL")*20-s.count("XC")*20-s.count("CD")*200-s.count("CM")*200
#         return v
# # test code
# Solution.romanToInt("III")
# Solution.romanToInt("LVIII")
# Solution.romanToInt("MCMXCIV")
# # best solution 1
# # my_mind.replace(original_roman_system, your_roman_system)
# class Solution:
#     def romanToInt(self, s: str) -> int:
#         translations = {
#             "I": 1,
#             "V": 5,
#             "X": 10,
#             "L": 50,
#             "C": 100,
#             "D": 500,
#             "M": 1000
#         }
#         number = 0
#         s = s.replace("IV", "IIII").replace("IX", "VIIII")
#         s = s.replace("XL", "XXXX").replace("XC", "LXXXX")
#         s = s.replace("CD", "CCCC").replace("CM", "DCCCC")
#         for char in s:
#             number += translations[char]
#         return number
# # simplified
# class Solution:
#     def romanToInt(self, s: str) -> int:
#         roman_to_integer = {'I': 1,
#                             'V': 5,
#                             'X': 10,
#                             'L': 50,
#                             'C': 100,
#                             'D': 500,
#                             'M': 1000}

#         s = s.replace('IV', 'IIII') \
#              .replace('IX', 'VIIII') \
#              .replace('XL', 'XXXX') \
#              .replace('XC', 'LXXXX') \
#              .replace('CD', 'CCCC') \
#              .replace('CM', 'DCCCC')
        
#         return sum(map(roman_to_integer.get, s))
# # best solution 2
# # The key intuition lies in the fact that in Roman numerals, 
# # when a smaller value appears before a larger value, it represents 
# # subtraction, while when a smaller value appears after or equal to a 
# # larger value, it represents addition.
# class Solution:
#     def romanToInt(self, s: str) -> int:
#         m = {
#             'I': 1,
#             'V': 5,
#             'X': 10,
#             'L': 50,
#             'C': 100,
#             'D': 500,
#             'M': 1000
#         }
#         ans = 0
        
#         for i in range(len(s)):
#             if i < len(s) - 1 and m[s[i]] < m[s[i+1]]:
#                 # not last digit, and is smaller than digit after it
#                 ans -= m[s[i]]
#             else:
#                 ans += m[s[i]]
#         return ans
    
# 14 length of last word, best solution similar but account for edge case of no words
# class Solution:
#     def lengthOfLastWord(s: str) -> int:
#         x=s.split()
#         if x:
#             return len(s.split()[-1])
#         return 0
# # test code
# print(Solution.lengthOfLastWord(""))
# print(Solution.lengthOfLastWord("   fly me   to   the moon  "))
# print(Solution.lengthOfLastWord("luffy is still joyboy"))

# 14. longest common prefix
# class Solution:
#     def longestCommonPrefix(strs: List[str]) -> str:
#         x = "" # result string
#         y = len(strs) # number of elements in the list
#         z = len(strs[0]) # shortest string
#         for i in strs:
#             z = min(len(i),z)
        
#         if y == 1:
#             return strs[0]
#         if y >= 2:
#             for a in range(z):
#                 temp = x + strs[0][a]
#                 for b in strs:
#                     if not b.startswith(temp):
#                         return x
#                 x = temp
#         return x
# # test
# print(Solution.longestCommonPrefix(["flower","flow","flight"]))
# print(Solution.longestCommonPrefix(["dog","racecar","car"]))
# print(Solution.longestCommonPrefix([""]))
# # best solution
# # Initialize an empty string ans to store the common prefix.
# # Sort the input list v lexicographically. This step is necessary because the common prefix should be common to all the strings, so we need to find the common prefix of the first and last string in the sorted list.
# # Iterate through the characters of the first and last string in the sorted list, stopping at the length of the shorter string.
# # If the current character of the first string is not equal to the current character of the last string, return the common prefix found so far.
# # Otherwise, append the current character to the ans string.
# # Return the ans string containing the longest common prefix.
# class Solution:
#     def longestCommonPrefix(self, v: List[str]) -> str:
#         if len(v) == 0: # i added on this part for the edge case
#             return ""
#         ans=""
#         v.sort()
#         first=v[0]
#         last=v[-1]
#         for i in range(min(len(first),len(last))):
#             if(first[i]!=last[i]):
#                 return ans
#             ans+=first[i]
#         return ans 

# 28. find the index of the first occurence in a string
# class Solution:
#     def strStr(haystack: str, needle: str) -> int:
#         try:
#             return haystack.index(needle)
#         except:
#             return -1
# # test
# print(Solution.strStr("sadbutsad","sad"))
# print(Solution.strStr("leetcode","leeto"))
# print(Solution.strStr("",""))
# # best solution 1, better since we do not have to use the try except block
# def strStr(haystack: str, needle: str) -> int:
#     return haystack.find(needle)
# # best solution 2
# # makes sure we don't iterate through a substring that is shorter than needle
# class Solution:
#     def strStr(self, haystack, needle):
#         # makes sure we don't iterate through a substring that is shorter than needle
#         for i in range(len(haystack) - len(needle) + 1):
#             # check if any substring of haystack with the same length as needle is equal to needle
#             if haystack[i : i+len(needle)] == needle:
#                 # if yes, we return the first index of that substring
#                 return i
#         # if we exit the loop, return -1        
#         return -1
    
# # 125. valid palindrome
# class Solution:
#     def isPalindrome(s: str) -> bool:
#         x = s.lower()
#         z = ""
#         for w in x:
#             if w.isalnum():
#                 z += w
#         return z==z[::-1]
# # test
# print(Solution.isPalindrome("A man, a plan, a canal: Panama"))
# print(Solution.isPalindrome("race a car"))
# print(Solution.isPalindrome(" "))
# # best solution 1, similar to mine but much more simpler
# class Solution:
#     def isPalindrome(s: str) -> bool:
#         s = [i for i in s.lower() if i.isalnum()]
#         return s == s[::-1]
# # best solution 2, using pointers
# # iterative solution
# class Solution:
#     def isPalindrome(s: str) -> bool:
#         i, j = 0, len(s) - 1
#         while i < j:
#             a, b = s[i].lower(), s[j].lower()
#             if a.isalnum() and b.isalnum():
#                 if a != b: return False
#                 else:
#                     i, j = i + 1, j - 1
#                     continue
#             i, j = i + (not a.isalnum()), j - (not b.isalnum())
#         return True
    
# 392. is subsequence
# class Solution:
#     def isSubsequence(s: str, t: str) -> bool:
#         # basically two pointers compare
#         a, b, c, d= 0, len(s), 0, len(t)
#         while a < b and c < d:
#             if s[a]==t[c]:
#                 a+=1
#             c+=1
#         return a == b
# # another version, better space, worse time
# class Solution:
#     def isSubsequence(self, s: str, t: str) -> bool:
#         a, c= 0, 0
#         while a < len(s) and c < len(t):
#             if s[a]==t[c]:
#                 a+=1
#             c+=1
#         return a == len(s)
# # test
# print(Solution.isSubsequence("abc","ahbgdc"))
# print(Solution.isSubsequence("axc","ahbgdc"))
# print(Solution.isSubsequence(" "," "))

# 383. ransom note
# very slow implementation
# class Solution:
#     def canConstruct(ransomNote: str, magazine: str) -> bool:
#         a, b, c, d= 0, sorted(ransomNote), 0, sorted(magazine)
#         while a < len(b) and c < len(d):
#             if b[a]==d[c]:
#                 a+=1
#             c+=1
#         return a == len(b)
# self solution 2, much better
# class Solution:
#     def canConstruct(ransomNote: str, magazine: str) -> bool:
#         a = list(magazine)
#         for i in ransomNote:
#             try:
#                 a.remove(i)
#             except:
#                 return False
#         return True
# # best solution 1
# class Solution:
#     def canConstruct(ransomNote: str, magazine: str) -> bool:
#         for i in ransomNote:
#             if i in magazine:
#                 magazine = magazine.replace(i,"",1)
#             else: return False
#         return True
# # hashmap implementation
# class Solution(object):
#     def canConstruct(ransomNote, magazine):
#         # Create a dictionary to store character counts
#         d = {}
#         # Iterate through the magazine and count characters
#         for i in magazine:
#             if i not in d:
#                 d[i] = 1
#             else:
#                 d[i] += 1
#         # Iterate through the ransom note and check character counts
#         for j in ransomNote:
#             if j in d and d[j] > 0:
#                 d[j] -= 1
#             else:
#                 return False
#         return True
# # try except slows down runtime
# # test
# print(Solution.canConstruct("a","b"))
# print(Solution.canConstruct("aa","ab"))
# print(Solution.canConstruct("aa","aab"))
# print(Solution.canConstruct("abc","ahbgdc"))
# print(Solution.canConstruct("axc","ahbgdc"))
# print(Solution.canConstruct(" "," "))

# 205. isomorphic strings
# horrendous implementation but it works
# class Solution:
#     def isIsomorphic(s: str, t: str) -> bool:
#         if len(s) != len(t):
#             return False
#         a, b = {}, {}
#         for i in range(len(s)):
#             if s[i] in a:
#                 a[s[i]] += str(i)
#             else:
#                 a[s[i]] = str(i)
#         for j in range(len(t)):
#             if t[j] in b:
#                 b[t[j]] += str(j)
#             else:
#                 b[t[j]] = str(j)
#         print(list(a.values()))
#         print(list(b.values()))
#         return list(a.values()) == list(b.values())
# best solution 1
# based on the fact that if that isomorphic strings should only have
# one pair of corresponding letters for each letter, same as the number of
# unique letters
#  => zip function would pair the first item of first iterator 
# (i.e s here) to the first item of second iterator (i.e t here). 
# set() would remove duplicate items from the zipped tuple. It is 
# like the first item of first iterator mapped to the first item of 
# second iterator as it would in case of a hashtable or dictionary.
# class Solution:
#     def isIsomorphic(self, s: str, t: str) -> bool:
#         zipped_set = set(zip(s, t))
#         return len(zipped_set) == len(set(s)) == len(set(t))
# test
# print(Solution.isIsomorphic("egg","add"))
# print(Solution.isIsomorphic("foo","bar"))
# print(Solution.isIsomorphic("aa","aab"))
# print(Solution.isIsomorphic("abc","ahbgdc"))
# print(Solution.isIsomorphic("badc","baba"))
# print(Solution.isIsomorphic(" "," "))

# 290. word pattern
# class Solution:
#     def wordPattern(pattern: str, s: str) -> bool:
#         a = list(pattern)
#         b = s.split()
#         if len(a)!= len(b):
#             return False
#         return len(set(pattern))==len(set(b))==len(set(zip(a,b)))
# best solution 1
# class Solution:
#     def wordPattern(pattern: str, s: str) -> bool:
#         s = s.split()
#         return (len(set(pattern)) ==
#                 len(set(s)) ==
#                 len(set(zip_longest(pattern,s))))
    # zip: The zip function iterates over each element of the provided 
    # iterables in parallel and creates tuples by combining corresponding 
    # elements from each iterable. It stops when the shortest input iterable 
    # is exhausted. If the input iterables are of different lengths, 
    # the resulting iterator will only have as many elements as the shortest input iterable.
    
    # zip_longest: The zip_longest function is available in the itertools 
    # module. It iterates over each element of the provided iterables in 
    # parallel, similar to zip, but it continues until the longest input 
    # iterable is exhausted. If the input iterables are of different lengths, 
    # zip_longest fills in missing values with a specified fill value (default is None).

    # in this case zip_longest eliminate the need to check the length of the inputs
# test
# print(Solution.wordPattern("abba","dog cat cat dog"))
# print(Solution.wordPattern("abba","dog cat cat fish"))
# print(Solution.wordPattern("aaaa","dog cat cat dog"))

# 242. valid anagram
# class Solution:
#     def isAnagram(s: str, t: str) -> bool:
#         return sorted(s)==sorted(t)
# # best solution 1
# # This approach counts the frequency of characters in one string 
# # and then adjusts the count by decrementing for the other string. 
# # If the strings are anagrams, the frequency of each character will 
# # cancel out, resulting in a map with all zero frequencies.
# class Solution:
#     def isAnagram(self, s: str, t: str) -> bool:
#         count = defaultdict(int)
#         if len(s) != len(t):
#             return False
        
#         # Count the frequency of characters in string s
#         # Decrement the frequency of characters in string t
#         # the version with for x in s and for x in t (two separate for in is faster)
#         for i in range(len(s)):
#             count[s[i]] += 1
#             count[t[i]] -= 1
                
#         # Check if any character has non-zero frequency
#         for val in count.values():
#             if val != 0:
#                 return False
#         return True
# # best solution 2
# # same as above but use array instead
# class Solution:
#     def isAnagram(self, s: str, t: str) -> bool:
#         count = [0] * 26
        
#         # Count the frequency of characters in string s
#         for x in s:
#             count[ord(x) - ord('a')] += 1
        
#         # Decrement the frequency of characters in string t
#         for x in t:
#             count[ord(x) - ord('a')] -= 1
        
#         # Check if any character has non-zero frequency
#         for val in count:
#             if val != 0:
#                 return False
        
#         return True
# # test
# print(Solution.isAnagram("anagram","nagaram"))
# print(Solution.isAnagram("rat","car"))

# 1. two sum
# class Solution:
#     def twoSum(nums: List[int], target: int) -> List[int]:
#         x = len(nums)
#         for i in range(x):
#             for j in range(i+1,x):
#                 if (nums[i] + nums[j]) == target:
#                     return [i,j]
# best solution 1, Two-pass Hash Table
    # Approach using a hash table:
    # Create an empty hash table to store elements and their indices.
    # Iterate through the array from left to right.
    # For each element nums[i], calculate the complement by subtracting it from the target: complement = target - nums[i].
    # Check if the complement exists in the hash table. If it does, we have found a solution.
    # If the complement does not exist in the hash table, add the current element nums[i] to the hash table with its index as the value.
    # Repeat steps 3-5 until we find a solution or reach the end of the array.
    # If no solution is found, return an empty array or an appropriate indicator.
    # This approach has a time complexity of O(n) since hash table lookups take constant
    # time on average. It allows us to solve the Two Sum problem efficiently by making just 
    # one pass through the array.
# class Solution:
#     def twoSum(self, nums: List[int], target: int) -> List[int]:
#         numMap = {}
#         n = len(nums)

#         # Build the hash table
#         for i in range(n):
#             numMap[nums[i]] = i

#         # Find the complement
#         for i in range(n):
#             complement = target - nums[i]
#             if complement in numMap and numMap[complement] != i:
#                 return [i, numMap[complement]]

#         return []  # No solution found
# best solution 2, one pass hash table
# class Solution:
#     def twoSum(self, nums: List[int], target: int) -> List[int]:
#         numMap = {}
#         n = len(nums)

#         for i in range(n):
#             complement = target - nums[i]
#             if complement in numMap:
#                 return [numMap[complement], i]
#             numMap[nums[i]] = i

#         return []  # No solution found
# # test
# print(Solution.twoSum([2,7,11,15], 9))
# print(Solution.twoSum([3,2,4], 6))
# print(Solution.twoSum([3,3], 6))

# 202. happy number
# basically if loop -> not happy, if it goes to 1 happy
# it will never goes to infinity (test out numbers like 99999)
# class Solution:
#     def isHappy(n: int) -> bool:
#         d = {}
#         n = str(n)
#         s = 0
#         while True:
#             for i in n:
#                 s+=int(i)**2
#             if s == 1:
#                 return True
#             elif s in d:
#                 return False
#             else:
#                 d[s]=1
#                 n = str(s)
#                 s=0
# # best solution 1
# # tortoise hare technique to detect cycle, two pointers going at different speed
# def isHappy(self, n: int) -> bool:
# 	#20 -> 4 -> 16 -> 37 -> 58 -> 89 -> 145 > 42 -> 20
# 	slow = self.squared(n)
# 	fast = self.squared(self.squared(n))

# 	while slow!=fast and fast!=1:
# 		slow = self.squared(slow)
# 		fast = self.squared(self.squared(fast))

# 	return fast==1

# def squared(self, n):
# 	result = 0
# 	while n>0:
# 		last = n%10
# 		result += last * last
# 		n = n//10
# 	return result
# # best solution 2
# # similar to mine but use set instead
# class Solution:
#     def isHappy(self, n: int) -> bool:
#         hset = set()
#         while n != 1:
#             if n in hset: return False
#             hset.add(n)
#             n = sum([int(i) ** 2 for i in str(n)])
#         else:
#             return True
# # test
# print(Solution.isHappy(19))
# print(Solution.isHappy(7))
# print(Solution.isHappy(1))

# 219 contain duplicates 2
# attampt 1, too slow
# class Solution:
#     def containsNearbyDuplicate(nums: List[int], k: int) -> bool:
#         d = {}
#         for i in range(len(nums)):
#             for j in range(-k+i,k+i):
#                 if j in d and d[j] == nums[i]:
#                     return True
#             d[i]=nums[i]
#         return False
# attempt 2 really good result 90%, 20%
# I think in this case we have to realized because it is <= k
# we can always update the hashtable with the most recently encountered
# number
# class Solution:
#     def containsNearbyDuplicate(nums: List[int], k: int) -> bool:
#         d = {}
#         for i in range(len(nums)):
#             if nums[i] in d and abs(i-d[nums[i]])<=k:
#                 # note the abs part is not needed because i in this case
#                 # would always be bigger than d[nums[i]]
#                 # since we are moving from left to right
#                 return True
#             d[nums[i]]=i
#         return False
# # best solution 1, 96 68
# class Solution:
#     def containsNearbyDuplicate(self, nums: List[int], k: int) -> bool:
#         d = {}
#         for i, n in enumerate(nums):
#           if n in d and (i - d[n]) <= k:
#             return True
#           else:
#             d[n] = i
#         return False
# # test
# print(Solution.containsNearbyDuplicate([1,2,3,1],3))
# print(Solution.containsNearbyDuplicate([1,0,1,1], 1))
# print(Solution.containsNearbyDuplicate([1,2,3,1,2,3], 2))

# 228. Summary Ranges
# class Solution:
#     def summaryRanges(nums: List[int]) -> List[str]:
#         n = len(nums)-1
#         if n <0:
#             return []
#         r = []
#         f = nums[0]
#         l = nums[0]
#         for i in range(1,n+1):
#             if nums[i] == l+1:
#                 l = nums[i]
#             else:
#                 if f==l:
#                     r.append(f'{f}')
#                 else:
#                     r.append(f'{f}->{l}')
#                 f = nums[i]
#                 l = f
#         if f==l:
#             r.append(f'{f}')
#         else:
#             r.append(f'{f}->{l}')
#         return r
# best solution 1
# class Solution:
#     def summaryRanges(self, nums: List[int]) -> List[str]:
#         ranges = [] # [start, end] or [x, y]
#         for n in nums:
#             # if there are entries in ranges and the end of last entry is sequential to n
#             if ranges and ranges[-1][1] == n-1:
#                 # update end
#                 ranges[-1][1] = n
#             else:
#                 # append completed sequence
#                 ranges.append([n, n])
#         # list comprehension
#         return [f'{x}->{y}' if x != y else f'{x}' for x, y in ranges]
# # test
# print(Solution.summaryRanges([0,1,2,4,5,7]))
# print(Solution.summaryRanges([0,2,3,4,6,8,9]))

# 20 valid parentheses
# class Solution: # 87, 98, forgot to take out the print statements...
#     def isValid(s: str) -> bool:
#         d = {")":"(","}":"{","]":"["}
#         f = [")","}","]"]
#         h = []
#         for i in s:
#             print(i)
#             if i in f:
#                 if not h or h.pop() != d[i]:
#                     return False
#             else:
#                 h.append(i)
#         return not h and True
# best solution 1, basically mine but simplified
# class Solution(object):
#     def isValid(s):
#         d = {'(':')', '{':'}','[':']'}
#         stack = []
#         for i in s:
#             if i in d:  # 1
#                 stack.append(i)
#             elif len(stack) == 0 or d[stack.pop()] != i:  # 2
#                 return False
#         return len(stack) == 0 # 3
# 1. if it's the left bracket then we append it to the stack
# 2. else if it's the right bracket and the stack is empty(meaning no matching left bracket), or the left bracket doesn't match
# 3. finally check if the stack still contains unmatched left bracket
# test
# print(Solution.isValid("()"))
# print(Solution.isValid("()[]{}"))
# print(Solution.isValid("(}"))
# print(Solution.isValid("("))

# 141. linked list cycle
# Definition for singly-linked list. 72, 97
class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None
# class Solution:
#     def hasCycle(head:[ListNode]) -> bool:
#         a,b = head, head
#         while True:
#             if a == None or b == None:
#                 return False
#             a = a.next
#             b = b.next
#             if a == None or b == None:
#                 return False
#             b = b.next
#             if a == b:
#                 return True
# # best solution 1
# class Solution:
#     def hasCycle(self, head: Optional[ListNode]) -> bool:
#         slow_pointer = head
#         fast_pointer = head
#         # genius of them to use the while loop to make sure fast pointer
#         # .next.next can be called
#         while fast_pointer and fast_pointer.next:
#             slow_pointer = slow_pointer.next
#             fast_pointer = fast_pointer.next.next
#             if slow_pointer == fast_pointer:
#                 return True
#         return False
# # best solution 2 hash table, this doesnt seem very space efficient
# class Solution:
#     def hasCycle(self, head: Optional[ListNode]) -> bool:
#         visited_nodes = set()
#         current_node = head
#         while current_node:
#             if current_node in visited_nodes:
#                 return True
#             visited_nodes.add(current_node)
#             current_node = current_node.next
#         return False
# test 
# x = ListNode(3)
# x.next = ListNode(2)
# x.next.next = ListNode(0)
# x.next.next.next = ListNode(-4)
# x.next.next.next.next = x.next
# print(Solution.hasCycle(x))
# y = ListNode(1)
# y.next = ListNode(2)
# y.next.next = y
# print(Solution.hasCycle(y))
# z = ListNode(1)
# z.next = None
# print(Solution.hasCycle(z))

# 21. Merge Two Sorted Lists, 36, 58 or 56, 11
# got stuck on this the main thing is you have to check is the node is none
# and if node.val is equals to none (but the latter is not required for leetcode)
# so the standard solution is similar to mine with less check, also at the end it
# simply added the left over list to the end of dummy load list, which is more efficient
# class Solution:
#     def mergeTwoLists(list1:[ListNode], list2:[ListNode]) -> [ListNode]: 
#         r = ListNode(0)
#         cr = r
#         while list1 and list2 and list1.val != None and list2.val != None:
#             if list1.val < list2.val:
#                 cr.next = list1
#                 list1 = list1.next
#             else:
#                 cr.next = list2
#                 list2 = list2.next
#             cr = cr.next
#         while list1 and list1.val != None:
#             cr.next = list1
#             list1 = list1.next
#             cr = cr.next
#         while list2 and list2.val != None:
#             cr.next = list2
#             list2 = list2.next
#             cr = cr.next
#         return r.next
# best solution 1
# class Solution:
#     def mergeTwoLists(list1: [ListNode], list2: [ListNode]) -> [ListNode]:
#         cur = dummy = ListNode(0)
#         while list1 and list2:               
#             if list1.val < list2.val: # do no
#                 cur.next = list1
#                 list1, cur = list1.next, list1
#             else:
#                 cur.next = list2
#                 list2, cur = list2.next, list2
#         if list1 or list2:
#             cur.next = list1 if list1 else list2
#         return dummy.next
# # test
# x = ListNode(1)
# x.next = ListNode(2)
# x.next.next = ListNode(4)
# y = ListNode(1)
# y.next = ListNode(3)
# y.next.next = ListNode(4)
# r1 = Solution.mergeTwoLists(x, y)
# print("start")
# while r1:
#     print(r1.val)
#     r1 = r1.next
# a = ListNode(None)
# b = ListNode(None)
# r2 = Solution.mergeTwoLists(a, b)
# print("start")
# while r2:
#     print(r2.val)
#     r2 = r2.next
# c = ListNode(0)
# r3 = Solution.mergeTwoLists(a, c)
# print("start")
# while r3:
#     print(r3.val)
#     r3 = r3.next

# 104 maximum depth of binary tree
# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
# class Solution: # 79, 84
#     def maxDepth(self, root: [TreeNode]) -> int:
#         x = 0
#         y = 0
#         r = 0
#         if root:
#             r = 1
#             if root.left:
#                 x = self.maxDepth(root.left) + 1
#             if root.right:
#                 y = self.maxDepth(root.right) + 1
#         return max(r,x,y)
# # best solution 1
# class Solution:
#     def maxDepth(self, root: Optional[TreeNode]) -> int:
#         # return 0 is the root is none
#         # return 1 + recursion result
#         return 0 if not root else 1 + max(self.maxDepth(root.left), self.maxDepth(root.right)) 

# 100. same tree
# class Solution: # 35 10
#     def isSameTree(self, p: Optional[TreeNode], q: Optional[TreeNode]) -> bool:
#         # if p and not q or not p and q: 
#         #     return False
#         # if not p and not q:
#         #     return True
#         # if p and q and p.val == q.val:
#         #     return True
#         # if p and q and p.val != q.val:
#         #     return False
#         return (not p and not q) or (not(p and not q)) and (not(q and not p)) and p.val == q.val and self.isSameTree(p.left, q.left) and self.isSameTree(p.right, q.right)
# # best solution 1, i guess sometimes python runs faster when you write things out
# class Solution:
#     def isSameTree(self, p: Optional[TreeNode], q: Optional[TreeNode]) -> bool:
#         if not p and not q:
#             return True
#         if not p or not q: 
#             return False # so to reach here q is null or p is null
#         return p.val == q.val and self.isSameTree(p.left, q.left) and self.isSameTree(p.right, q.right)
# # best solution 2, 91 59
# # this is the fastest solution so the sequence of your logic matters
# class Solution:
#     def isSameTree(self, p: Optional[TreeNode], q: Optional[TreeNode]) -> bool:
#         if p and q:
#             return p.val == q.val and self.isSameTree(p.left, q.left) and self.isSameTree(p.right, q.right)
#         return p is q 
# # The last return statement only runs in these 3 cases: 
# # (1) p and q are both None in which case it is return None is None which is return True, 
# # (2) q is None but p is a TreeNode, 
# (3) p is None but q is a TreeNode. 
# In cases 2 and 3 it would be comparing None is Treenode which False so it would return False.

# 226. invert binary tree, 56 86
# class Solution:
#     def invertTree(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
#         if not root:
#             return # can be return root too but its null anyways
#         x = root.left
#         root.left  = self.invertTree(root.right)
#         root.right = self.invertTree(x)
#         return root
# # best solution 1, 27 34? basically the same as mine
# class Solution:
#     def invertTree(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
#         if not root: #Base Case
#             return root
#         self.invertTree(root.left) #Call the left substree
#         self.invertTree(root.right)  #Call the right substree
#         # Swap the nodes
#         root.left, root.right = root.right, root.left
#         return root # Return the root

# 101. symmetric tree, 73, 16
# class Solution:
#     def isSymmetric(self, root: Optional[TreeNode]) -> bool:
#         # adding if not root return True will slow down to 43, but use less space, 64
#         return self.isSymmetricHelper(root.left, root.right)

#     def isSymmetricHelper(self, rootleft: Optional[TreeNode], rootright: Optional[TreeNode]) -> bool:
#         if rootleft and rootright:
#             return rootleft.val == rootright.val and self.isSymmetricHelper(rootleft.right, rootright.left) and self.isSymmetricHelper(rootleft.left, rootright.right)
#         return rootleft is rootright

# 112. path sum, 43 87
# class Solution:
#     def hasPathSum(self, root: Optional[TreeNode], targetSum: int) -> bool:
#         if root:
#             if not root.left and not root.right:
#                 return targetSum == root.val
#             return self.hasPathSum(root.left, targetSum-root.val) or self.hasPathSum(root.right, targetSum-root.val)
#         return False
#         # the original version has a simple count down with targetSum and check if there
#         # is no root and targetsum is 0 then return true. But this does not work with the
#         # case of empty tree given and targetsum = 0 (it will return true) even tho there
#         # is no tree
# # best version 1, 62 87
# class Solution:
#     def hasPathSum(self, root: Optional[TreeNode], targetSum: int) -> bool:
#         if not root:
#             return False
#         if not root.left and not root.right:
#             return targetSum == root.val
#         left_sum = self.hasPathSum(root.left, targetSum - root.val)
#         right_sum = self.hasPathSum(root.right, targetSum - root.val)
#         return left_sum or right_sum
#         # same as mine but different sequence
#         # i think putting the recursive ones last help with runtime

# 222. count complete tree nodes, 8 37, 47 37, 
# class Solution:
#     def countNodes(self, root: Optional[TreeNode]) -> int:
#         if not root:
#             return 0
#         x = self.countNodes(root.left) + self.countNodes(root.right) + 1
#         return x
# # best solution 1, 93 37, 98 9
# # The main idea of this algorithm is to first find the height of the 
# # leftmost and rightmost subtree of the given complete binary tree, 
# # then check if the height of both subtrees is the same. If it is the 
# # same, then the tree is a perfect binary tree and we can easily calculate 
# # the total number of nodes in the tree. Otherwise, we recursively count 
# # the nodes in the left and right subtree.
# class Solution:
#   def countNodes(self, root: Optional[TreeNode]) -> int:
#     if not root:
#         return 0
#     l = root
#     r = root
#     hl = 0
#     hr = 0
#     while l:
#         l = l.left
#         hl += 1
#     while r:
#         r = r.right
#         hr += 1
#     if hl == hr:
#         return pow(2, hl) - 1
#     return 1 + self.countNodes(root.left) + self.countNodes(root.right)
#     # if we do x = self.countNodes(root.left) + self.countNodes(root.right) + 1, return x
#     # run time goes down and space usage go up
# # BFS
# class Solution:
# 	def countNodes(self, root: Optional[TreeNode]) -> int:
# 		def BFS(node):
# 			if node == None:
# 				return 0
# 			queue = [node]
# 			self.result = 0
# 			while queue:
# 				current_node = queue.pop()
# 				self.result = self.result + 1
# 				if current_node.left != None:
# 					queue.append(current_node.left)
# 				if current_node.right != None:
# 					queue.append(current_node.right)
# 			return self.result
# 		return BFS(root)

# 637. average level in a binary tree, 36 62
# class Solution:
#     def averageOfLevelsHelper(self, q: List[TreeNode], r: List[float]) -> List[float]:
#         if not q:
#             return r
#         t = []
#         s = 0
#         c = 0

#         while q:
#             cn = q.pop()
#             s += cn.val
#             c += 1
#             if cn.left:
#                 t.append(cn.left)
#             if cn.right:
#                 t.append(cn.right)
#         r.append(s/c)

#         x = self.averageOfLevelsHelper(t, r)

#         return x

#     def averageOfLevels(self, root: Optional[TreeNode]) -> List[float]:
#         if not root:
#             return []
#         q = [root]
#         r = self.averageOfLevelsHelper(q, [])
#         return r
# # best solution 1 DFS, 86 25
# def averageOfLevels(self, root: TreeNode) -> List[float]:
# 	lvlcnt = defaultdict(int)
# 	lvlsum = defaultdict(int)
# 	def dfs(node=root, level=0):
# 		if not node: return
# 		lvlcnt[level] += 1
# 		lvlsum[level] += node.val
# 		dfs(node.left, level+1)
# 		dfs(node.right, level+1)
# 	dfs()
# 	return [lvlsum[i] / lvlcnt[i] for i in range(len(lvlcnt))]
# # best solution 2
# class Solution:
#     def averageOfLevels(self, root: TreeNode) -> List[float]:
#         if not root:
#             # Quick response for empty tree
#             return []
#         traversal_q = [root]
#         average = []
#         while traversal_q:
#             # compute current level average
#             cur_avg = sum( (node.val for node in traversal_q if node) ) / len(traversal_q)
#             # add to result
#             average.append( cur_avg )
#             # update next level queue
#             next_level_q = [ child for node in traversal_q for child in (node.left, node.right) if child ]
#             # update traversal queue as next level's
#             traversal_q = next_level_q
#         return average

# # 530. Minimum Absolute Differences in BST, 66 99
# # key is inorder traverse (implemented in helper method)
# class Solution:
#     def getMinimumDifferenceHelper(self, root: Optional[TreeNode], r: List[int]) -> List[int]:
#         if root.left:
#             # Traverse the left subtree, i.e., call Inorder(left->subtree)
#             self.getMinimumDifferenceHelper(root.left, r)
#         # Visit the root.
#         r.append(root.val)
#         if root.right: 
#             # Traverse the right subtree, i.e., call Inorder(right->subtree)
#             self.getMinimumDifferenceHelper(root.right, r)
#         return r
#     def getMinimumDifference(self, root: Optional[TreeNode]) -> int:
#         x = self.getMinimumDifferenceHelper(root, [])
#         y = sys.maxsize
#         for i in range(1, len(x)):
#             y = min(y, x[i]-x[i-1])
#         return y

# 108. convert sorted array to binary search tree
# it passed with 7, 12
# did not realize that we only need to return one of the version for even BST
# class Solution:
#     def sortedArrayToBST(self, nums: List[int]) -> Optional[TreeNode]:
#         x = len(nums)
#         r1 = None
#         r2 = None
#         # if BST is empty, return none
#         if x == 0:
#             return r1
#         # if we have only one node left, return a single node with left and right as null
#         elif x == 1:
#             r1 = TreeNode(nums[0])
#             r1.left = None
#             r1.right = None
#             return r1
#         elif x >= 1:
#             if x % 2 != 0: # odd
#                 r1 = TreeNode(nums[x//2])
#                 r1.left = self.sortedArrayToBST(nums[0:x//2])
#                 r1.right = self.sortedArrayToBST(nums[x//2+1:])
#             else: # even
#                 # v1
#                 r1 = TreeNode(nums[x//2])
#                 r1.left = self.sortedArrayToBST(nums[0:x//2])
#                 r1.right = self.sortedArrayToBST(nums[x//2+1:])
#                 # v2
#                 r1 = TreeNode(nums[x//2-1])
#                 r1.left = self.sortedArrayToBST(nums[0:x//2-1])
#                 r1.right = self.sortedArrayToBST(nums[x//2:])
#         if r1 and r2:
#             return r1, r2
#         return r1
# # my implementation 2.0, 76 12
# class Solution:
#     def sortedArrayToBST(self, nums: List[int]) -> Optional[TreeNode]:
#         x = len(nums)
#         r1 = None
#         r2 = None
#         # if we have only one node left, return a single node with left and right as null
#         if x == 1:
#             r1 = TreeNode(nums[0])
#             r1.left = None
#             r1.right = None
#             return r1
#         elif x >= 1:
#             r1 = TreeNode(nums[x//2])
#             r1.left = self.sortedArrayToBST(nums[0:x//2])
#             r1.right = self.sortedArrayToBST(nums[x//2+1:])           
#         return r1
# # best solution 1, 91 45, besically passing index instead of passing the whole array
# # the whole array is accessible anyways since we are running a helper method
# class Solution:
#     def sortedArrayToBST(self, nums: List[int]) -> TreeNode:
#         def recurse(l, r):
#             # base case, l must always be <= r
#             # l == r is the case of a leaf node.
#             if l > r: return None
#             mid = (l+r)//2
#             node = TreeNode(nums[mid])
#             node.left = recurse(l, mid-1)
#             node.right = recurse(mid+1, r)
#             return node
#         # both the indices are inclusive,
#         # mathematically given by: [0, len(nums)-1]
#         return recurse(0, len(nums)-1)

# 35 search insert position 46 17
# class Solution:
#     def searchInsert(self, nums: List[int], target: int) -> int:
#         def searchInsertHelper(l: int, r: int) -> int:
#             x = (l+r)//2 
#             # using example we need l+r // 2 because if l is 2 and r is 3 the result is 2.5
#             # the old version has r-l which will result in 1 which is out of range
#             if l == r:
#                 return l
#             if target == nums[x]:
#                 return x
#             elif target < nums[x]:
#                 return searchInsertHelper(l, x-1) # x-1 because we need to exclude x
#             else: 
#                 return searchInsertHelper(x+1, r) # x+1 because we need to exclude x

#         z = searchInsertHelper(0, len(nums)-1)
#         return z
# # best solution 1, 86 50
# class Solution:
#     def searchInsert(self, nums: List[int], target: int) -> int:
#         low, high = 0, len(nums)
#         while low < high:
#             mid = (low + high) // 2
#             if target > nums[mid]:
#                 low = mid + 1
#             else:
#                 high = mid
#         return low
# # best solution 2, a bit cheating i think but seems efficient in real world usage
# import bisect
# class Solution:
#     def searchInsert(self, nums: List[int], target: int) -> int:
#         return bisect.bisect_left(nums, target)

# 67. add binary, 38 71
# class Solution:
#     def addBinary(self, a: str, b: str) -> str:
#         def addBinaryHelper(l: int, r:int, c:int) -> tuple[int, int]:
#                 x = l + r + c
#                 if x >= 2:
#                     return x-2, 1
#                 else:
#                     return x, 0 
#         r = ""
#         cm = 0
#         while a and b: 
#             x, cm = addBinaryHelper(int(a[-1]), int(b[-1]), cm)
#             a = a[:-1]
#             b = b[:-1]
#             r = str(x) + r
#         while a:
#             x, cm = addBinaryHelper(int(a[-1]), 0, cm) 
#             a = a[:-1]
#             r = str(x) + r
#         while b:
#             x, cm = addBinaryHelper(0, int(b[-1]), cm) 
#             b = b[:-1]
#             r = str(x) + r
#         if cm:
#             r = "1" + r
#         return r
# # best solution 1
# # Function to add two binary numbers represented as strings
#     def addBinary(self, a, b):
#         # List to store the result
#         result = []
#         # Variable to store the carry-over value
#         carry = 0
#         # Initialize two pointers to traverse the binary strings from right to left
#         i, j = len(a)-1, len(b)-1
#         # Loop until both pointers have reached the beginning of their respective strings and there is no carry-over value left
#         while i >= 0 or j >= 0 or carry:
#             total = carry
#             # Add the current binary digit in string a, if the pointer is still within bounds
#             if i >= 0:
#                 total += int(a[i])
#                 i -= 1
#             # Add the current binary digit in string b, if the pointer is still within bounds
#             if j >= 0:
#                 total += int(b[j])
#                 j -= 1
#             # Calculate the next binary digit in the result by taking the remainder of the sum divided by 2
#             result.append(str(total % 2))
#             # Calculate the next carry-over value by dividing the sum by 2
#             carry = total // 2
#         # Reverse the result and join the elements to form a single string
#         return ''.join(reversed(result))
# # best solution 2
# class Solution:
#     def addBinary(self, a: str, b: str) -> str:
#         result = ""
#         carry = 0
#         i = j = 0
#         # Iterate over both strings from right to left
#         while i < len(a) or j < len(b) or carry:
#             # Extract the current digits from a and b
#             digit_a = int(a[-1 - i]) if i < len(a) else 0
#             digit_b = int(b[-1 - j]) if j < len(b) else 0
#             # Calculate the sum of the current digits along with the carry
#             current_sum = digit_a + digit_b + carry
#             # Update the carry for the next iteration
#             carry = current_sum // 2
#             # Append the sum modulo 2 to the result
#             result += str(current_sum % 2)
#             # Move to the next digit
#             i += 1
#             j += 1
#         # Reverse the result string and return it
#         return result[::-1]

# 190 reverse bits, 76, 94
# class Solution:
#     def reverseBits(n: int) -> int:
#         x = int(f"{n:032b}"[::-1], 2)
#         # the f part turn n into binary string
#         # and then reverse it
#         # and the int(n, 2) turn it back into an integer
#         return x
# # best solution 1
#     # Algorithm:
#     # Initialize the reversed number to 0.
#     # Iterate over all 32 bits of the given number.
#     # In each iteration, left shift the reversed number by 1 and add the last bit of the given number to it.
#     # To add the last bit of the given number to the reversed number, perform an AND operation with the given number and 1.
#     # Right shift the given number by 1 to remove the last bit.
#     # Repeat steps 3-5 for all 32 bits of the given number.
#     # Return the reversed number
# class Solution:
#     def reverseBits(self, n: int) -> int:
#         # Initialize the reversed number to 0
#         reversed_num = 0
#         # Iterate over all 32 bits of the given number
#         for i in range(32):
#             # Left shift the reversed number by 1 and add the last bit of the given number to it
#             reversed_num = (reversed_num << 1) | (n & 1) # when left shift, a 0 is added to the end of the original number
#             # To add the last bit of the given number to the reversed number, perform an AND operation with the given number and 1
#             n >>= 1 # get rid of one digit from n
#         # Return the reversed number
#         return reversed_num
# # best solution 2, best overall
# class Solution:
#     def reverseBits(self, n: int) -> int:
#         reversed_int = 0
#         binary = bin(n)[2:] # covert to binary, get rid of 0b at the front
#         prefix = (32 - len(binary)) * "0"
#         binary = prefix + binary # add in 0 prefix
#         for i, bit in enumerate(binary): # iterate through binary number
#             if bit == "1": # if bit is 1, 
#                 reversed_int += 2**i # add in the 1
#                 # the 2**i part ensure that 1 is added to the digit in appropirate places
#         return reversed_int

# 191. number of 1 bits, 73 42
# class Solution:
#     def hammingWeight(self, n: int) -> int:
#         x = bin(n)[2:]
#         y = x.count('1')
#         return y
# # attempt 2, 99, 7
# class Solution:
#     def hammingWeight(self, n: int) -> int:
#         y = bin(n)[2:].count('1')
#         return y
# # best solution 1, best overall, same approach using different method, 99 87
# class Solution:
#     def hammingWeight(self, n: int) -> int:
#         b = format(n, 'b')
#         return b.count("1")
# # best solution 2, 84 87
#     # Brian Kernighan's Algorithm:
#     # Initialize a count variable to 0.
#     # Iterate through each bit of the number using the following steps:
#     #     If the current bit is 1, increment the count.
#     #     Set the current bit to 0 using the expression n = n & (n - 1).
#     # The count variable now holds the number of 1 bits.
# class Solution:
#     def hammingWeight(self, n: int) -> int:
#         count = 0
#         while n != 0:
#             n &= (n - 1)
#             count += 1
#         return count

# 136. single number, 87 92
# class Solution:
#     def singleNumber(self, nums: List[int]) -> int:
#         r = 0
#         for i in nums:
#             r = r ^ i # use xor, if there are duplicated numbers, xor gets rid of them
#         return r
# best solution same as above with different implementations

# 9. palindrome number
# class Solution:
#     def isPalindrome(self, x: int) -> bool:
#         return str(x) == str(x)[::-1]
# without converting to string version, best solution 1, 55 60
# class Solution:
#     def isPalindrome(self, x: int) -> bool:
#         if x < 0 or (x != 0 and x % 10 == 0):
#             return False
#         # the % 10 part If x is non-zero and ends with a zero, 
#         # it cannot be a palindrome because leading zeros are not 
#         # allowed in palindromes. We return false for such cases.
#         reversed_num = 0

#         while x > reversed_num:
#             reversed_num = reversed_num * 10 + x % 10
#             x //= 10

#         return x == reversed_num or x == reversed_num // 10
# #  my practice version 75 60
# class Solution:
#     def isPalindrome(self, x: int) -> bool:
#         if x < 0 or (x > 0 and x % 10 == 0):
#             return False
#         r = 0
#         while x > r:
#             r = r * 10 + x % 10
#             x = x // 10
#         y = x == r or x == r // 10
#         return y

# 66. plus one, 22 81
class Solution:
    def plusOne(self, digits: List[int]) -> List[int]:
        x = len(digits)-1
        c = 0
        r = []
        digits[x] += 1
        for i in range(x, -1, -1):
            digits[i] += c
            r.insert(0, digits[i] % 10)
            c = digits[i] >= 10
        if c:
            r.insert(0, 1)
        return r
# best solution 1
class Solution:
    def plusOne(self, digits: List[int]) -> List[int]:
        for i, d in reversed(list(enumerate(digits))):
            if d < 9:
                digits[i] += 1
                return digits
            digits[i] = 0
        return [1] + digits
    # going from the back, only need to add 1 to digits
    # if the digit is 9 we keep going until we find a digit thats not 9
    # and just increment it
    # otherwise we add 1 to the fron of digits

# 69. sqrt(x) 41, 83 this version is more efficient
    # If x is 0, return 0.
    # Initialize first to 1 and last to x.
    # While first is less than or equal to last, do the following:
    # a. Compute mid as first + (last - first) / 2.
    # b. If mid * mid equals x, return mid.
    # c. If mid * mid is greater than x, update last to mid - 1.
    # d. If mid * mid is less than x, update first to mid + 1.
    # Return last.
# class Solution:
#     def mySqrt(self, x: int) -> int:
#         if x == 0:
#             return 0
#         first, last = 1, x
#         while first <= last:
#             mid = first + (last - first) // 2
#             if mid == x // mid: # does the same as mid * mid =- x
#                 return mid
#             elif mid > x // mid: # does the same as mid * mid > x
#                 last = mid - 1
#             else:
#                 first = mid + 1
#         return last
        # the last is returned since when we break out of loop because 'first>last' 
        # - so our 'last' is at the lower index(floor) and 'first' is at the high index
        # (ceil). Here if our 'mid' never matches means we do not have an perfect sqrt, 
        # so then we return the floor - which is at the position 'last'.

# 70. climbing stairs
# recursion, time limit
class Solution:
    def climbStairs(self, n: int) -> int:
        if n <= 3:
            return n
        else:
            return self.climbStairs(n-1)+self.climbStairs(n-2)
# iterative 5 77
class Solution:
    def climbStairs(self, n: int) -> int:
        if n <= 3:
            return n
        n1 = 2
        n2 = 3
        r = 0
        for i in range(n-3):
            r = n1 + n2
            n1 = n2
            n2 = r
        return r
# best solution 1, 67 77
# Explanation: The tabulation solution eliminates recursion and uses a bottom-up 
# approach to solve the problem iteratively. It creates a DP table (dp) of size n+1 
# to store the number of ways to reach each step. The base cases (0 and 1 steps) are
#  initialized to 1 since there is only one way to reach them. Then, it iterates from 
# 2 to n, filling in the DP table by summing up the values for the previous two steps. 
# Finally, it returns the value in the last cell of the DP table, which represents the 
# total number of ways to reach the top.
class Solution:
    def climbStairs(self, n: int) -> int:
        if n == 0 or n == 1:
            return 1
        dp = [0] * (n+1) # an array of 0 length of n+1
        dp[0] = dp[1] = 1
        for i in range(2, n+1): # end at n
            dp[i] = dp[i-1] + dp[i-2]
        return dp[n]

