package dynamic_programmer;
class Solution {
    public int climbStairs(int n) {
        int []dp=new int [n+1];
        dp[0]=1;
        dp[1]=1;
        dp[2]=2;
        if (n<3){
            return n;
        }

        else{
            for(int i=3;i<=n;i++){
                dp[i]=dp[i-1]+dp[i-2];

            }
        }
        return dp[n];
    }
}
public class _70_palouti {
    public static void main(String[] args) {
        Solution solution1=new Solution();
        System.out.println(     solution1.climbStairs(4 ));
    }
}
