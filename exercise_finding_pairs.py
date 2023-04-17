import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt

# Set plotting options
plt.rc('figure', figsize=(16, 9))

# just set the seed for the random number generator
np.random.seed(2018)
# use returns to create a price series
drift = 100
r1 = np.random.normal(0, 1, 1000) 
s1 = pd.Series(np.cumsum(r1), name='s1') + drift
# s1.plot(figsize=(14,6))
# plt.show()

offset = 10
noise = np.random.normal(0, 1, 1000)
s2 = s1 + offset + noise
s2.name = 's2'
pd.concat([s1, s2], axis=1).plot(figsize=(15,6))
#plt.show()

price_ratio = s2/s1
price_ratio.plot(figsize=(15,7)) 
# plt.axhline(price_ratio.mean(), color='black') 
# plt.xlabel('Days')
# plt.legend(['s2/s1 price ratio', 'average price ratio'])
# plt.show()
# print(f"average price ratio {price_ratio.mean():.4f}")

#Calculate hedge ratio with regression

#Linear Regression
# type(s1)
# type(s1.values)
s1.values.reshape(-1,1).shape
lr = LinearRegression()
lr.fit(s1.values.reshape(-1,1),s2.values.reshape(-1,1))
hedge_ratio = lr.coef_[0][0]
intercept = lr.intercept_[0]
#print(f"hedge ratio from regression is {hedge_ratio:.4f}, intercept is {intercept:.4f}")


#Calculate the spread
spread = s2 - s1 * hedge_ratio
# print(f"Average spread is {spread.mean()}")
# spread.plot(figsize=(15,7)) 
# plt.axhline(spread.mean(), color='black') 
# plt.xlabel('Days')
# plt.legend(['Spread: s2 - hedge_ratio * s1', 'average spread'])
# plt.show()

#Let's see what we get if we include the intercept of the regression
spread_with_intercept = s2 - (s1 * hedge_ratio + intercept)
#print(f"Average spread with intercept included is {spread_with_intercept.mean()}")

spread_with_intercept.plot(figsize=(15,7)) 
# plt.axhline(spread_with_intercept.mean(), color='black') 
# plt.xlabel('Days')
# plt.legend(['Spread: s2 - (hedge_ratio * s1 + intercept)', 'average spread'])
# plt.show()


#Quiz
#Check if spread is stationary using Augmented Dickey Fuller Test
def is_spread_stationary(spread, p_level=0.05):
    """
    spread: obtained from linear combination of two series with a hedge ratio
    
    p_level: level of significance required to reject null hypothesis of non-stationarity
    
    returns:
        True if spread can be considered stationary
        False otherwise
    """
    #TODO: use the adfuller function to check the spread
    # adfuller(x, maxlag=None, regression='c', autolag='AIC', store=False, regresults=False)[source]

    # adf (float) – Test statistic
    # pvalue (float) – p-value

    adf_result = adfuller(spread, maxlag=None, regression='c', autolag='AIC', store=False, regresults=False)
    
    print()
    print("adf_result")
    print(adf_result)
    #print(f"adf_result.adf: {adf_result.adf}")
    print(f"adf_result.pvalue: {adf_result[1]}")
    # print(f"adf_result.usedlag: {adf_result.usedlag}")
    # print(f"adf_result.nobs: {adf_result.nobs}")
    # print(f"adf_result.icbest: {adf_result.icbest}")

    #get the p-value
    pvalue = adf_result[1]
    
    print(f"pvalue {pvalue:.4f}")
    if pvalue <= p_level:
        print(f"pvalue is <= {p_level}, assume spread is stationary")
        return True
    else:
        print(f"pvalue is > {p_level}, assume spread is not stationary")
        return False

# Try out your function

print()
print("is_spread_stationary")
is_spread_stationary = is_spread_stationary(spread)
#print(f"Are the two series candidates for pairs trading? {is_spread_stationary(spread)}")