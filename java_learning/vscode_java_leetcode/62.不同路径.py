#
# @lc app=leetcode.cn id=62 lang=python
#
# [62] 不同路径
#

# @lc code=start
class Solution(object):
    def uniquePaths(self, m, n):
        """
        :type m: int
        :type n: int
        :rtype: int
        """
        # 1.数  确定数组及其下标含义
        # 2.递 确定递推公式
        # 3.初始化  
        # 4.遍历顺序
        # 5.可打印输出

        #dp[i][j]表示到达i,j有多少种走法
        # python中二维数组的初始化方法
        # 1.
        #2.间接定义
        dp=[[0 for i in range(n+1)] for j in  range(m+1)]
        # print(dp)
        dp[0][0]=1
        for i in range(1,m+1):
            dp[i][0]=1
        for j in range(1,n+1):
            dp[0][j]=1
        for i in range(1,m+1):
            for j in range(1,n+1):
                dp[i][j]=dp[i-1][j]+dp[i][j-1]

        return dp[m-1][n-1]
        # print(dp[m-1][n-1])
# @lc code=end

if __name__ == '__main__':
    solu1=Solution()
    solu1.uniquePaths(1,1)
