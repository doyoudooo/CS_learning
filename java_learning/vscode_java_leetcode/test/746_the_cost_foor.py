def min_cost(cost):
    dp=[99]*len(cost)
    dp[0]=cost[0]
    dp[1]=cost[1]
    len1=len(cost)
    for i in range(2,len1):
        dp[i]=min(dp[i-1],dp[i-2])+cost[i]
    return min(dp[len1-1],dp[len1-2])

if __name__ == '__main__':
    cost=[10,15,20]
    cost1=[1,100,1,1,1,100,1,1,100,1]

    print( min_cost(cost1))