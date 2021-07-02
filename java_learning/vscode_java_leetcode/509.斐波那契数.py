#
# @lc app=leetcode.cn id=509 lang=python3
#
# [509] 斐波那契数
#

# @lc code=start
class Solution:
    def fib(self, n: int) -> int:
        if n<2:
            return n
        else:
            list1=[0,1]
            for i in range(2,n+1):
                list1.append(list1[i-1]+list1[i-2])
            return list1[n]
                

# @lc code=end

