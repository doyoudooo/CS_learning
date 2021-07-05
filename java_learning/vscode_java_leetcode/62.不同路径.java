/*
 * @lc app=leetcode.cn id=62 lang=java
 *
 * [62] 不同路径
 */

// @lc code=start
class Solution {
    public int uniquePaths(int m, int n) {
//1.要明确数组及其下标的含义
        int [][]dp=new int[m][n];
        dp[0][0]=1;
        for (int i = 0; i < m; i++) {
            dp[i][0]=1;
        }
        for (int j = 0; j < n; j++) {
            dp[0][n]=1;
        }

//        2.初始化，递推公式，遍历顺序

        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                dp[i][j]=dp[i-1][j]+dp[i][j-1];
            }

        }
        return dp[m-1][n-1];
    }

    public static void main(String[] args) {

        Solution s1=new Solution();
       int a= 0;
       a=s1.uniquePaths(1,1)
        System.out.println(a);
    }
}
// @lc code=end

