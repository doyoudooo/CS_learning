#
# @lc app=leetcode.cn id=70 lang=python
#
# [70] 爬楼梯
#

# @lc code=start
class Solution(object):
    def climbStairs(self, n):
        """
        :type n: int
        :rtype: int
        """
        # 1.数组及其下标
        # 每一层对应的含义，我的意思是：dp[i]：爬i层，有dp[i]种方法
        # 2.初始化
        dp=[0,1,2]
        # 爬一层有一种方法，爬两层有两种方法
        # 3.递推公式
        if (n<3):
            return n
        for i in range(3,n+1):
            dp.append(dp[i-1]+dp[i-2])
        return dp[n]

# @lc code=end

