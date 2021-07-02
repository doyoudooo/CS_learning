package dynamic_programmer;

public class _509_febonaqie {
public   long febonaqie(int n){
    long [] dp=new long[n+1];

//    1.确定dp数组和下标含义
    /*
    在这里用一个一维数组来存放结果
    dp[i]的定义就是：第i个数的斐波那契数值为dp[i]
     */
//    2.确定递推公式
    /*
    dp[i]=dp[i-1]+dp[i-2]
     */
//    3.数组初始化
    dp[0]=0;
    dp[1]=1;
//    4.确认遍历顺序
    /*
    从后往前
     */
//    5.可以打印输出
            if (n<2){
                return  dp[n];
            }else {
                for (int i = 2; i <= n; i++) {
                    dp[i]=dp[i-1]+dp[i-2];
                }
                return dp[n];
            }
}


    public static void main(String[] args) {
        _509_febonaqie fbnq = new _509_febonaqie();
        long startTime =  System.currentTimeMillis();

        System.out.println(fbnq.febonaqie(100));
        long endTime =  System.currentTimeMillis();
        long usedTime = (endTime-startTime);
        System.out.println(usedTime);
    }
}


