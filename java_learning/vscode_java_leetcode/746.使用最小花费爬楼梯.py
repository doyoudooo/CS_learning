#
# @lc app=leetcode.cn id=746 lang=python
#
# [746] 使用最小花费爬楼梯
#

# @lc code=start
class Solution(object):
    def minCostClimbingStairs(self, cost):
        """
        :type cost: List[int]
        :rtype: int
        """
        """
        思考？
        从第0阶开始，可以爬0阶也可以爬1阶，只要爬完了支付对应的体力值
    
        只要支付到达某个台阶的体力值
        """
        # 1.数组及下标
        # 我们假定只要支付了爬第i个台阶的费用，那么接下来的爬1阶或2阶就都不用费用了
        # 2.递推公式
        # 那么我们该怎么选取花费呢？
        # 可以这么思考，不管爬那一层楼梯，我们都要支付改层的费用，那么在爬下一级台阶的时候，我们可以尽量从费用较少的爬起
        # dp[i]=min{dp[1-1],dp[i-2]}+cost[i]
        #3.初始化，只需要最开始的dp状态   

        #4.遍历顺序 

        dp=[99]*len(cost)
        dp[0]=cost[0]
        dp[1]=cost[1]
        len1=len(cost)
        for i in range(2,len1):
            dp[i]=min(dp[i-1],dp[i-2])+cost[i]
        return min(dp[len1-1],dp[len1-2])



# @lc code=end

