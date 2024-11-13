import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

xeqt = pd.read_csv("xeqt.csv")
vsp = pd.read_csv("vsp.csv")

xeqt_prices = xeqt["Close"]
vsp_prices = vsp["Close"]

# calculate r 
mean_differences = []

for i, _ in enumerate(xeqt_prices):
    mean_differences.append((xeqt_prices[i] - xeqt_prices.mean()) * (vsp_prices[i] - vsp_prices.mean()))

a = np.array(mean_differences).sum()

xeqt_mean_diff_squared = []
vsp_mean_diff_squared = []

for i, _ in enumerate(xeqt_prices):
    xeqt_mean_diff_squared.append((xeqt_prices[i] - xeqt_prices.mean())**2)
    vsp_mean_diff_squared.append((vsp_prices[i] - vsp_prices.mean())**2)

b = np.sqrt(np.array(xeqt_mean_diff_squared).sum() * np.array(vsp_mean_diff_squared).sum())

r = a / b

print(f'standard deviation of XEQT {np.array(xeqt_prices).std()}')
print(f'standard deviation of VSP {np.array(vsp_prices).std()}')

xeqt_daily_returns = []
vsp_daily_returns = []

for i in range(len(xeqt_prices) - 1, 0, -1):
    xeqt_daily_returns.append((xeqt_prices[i] - xeqt_prices[i - 1])/xeqt_prices[i - 1]) 

for i in range(len(vsp_prices) - 1, 0, -1):
    vsp_daily_returns.append((vsp_prices[i] - vsp_prices[i - 1])/vsp_prices[i - 1]) 

xeqt_expected_return = np.array(xeqt_daily_returns).mean()
vsp_expected_return = np.array(vsp_daily_returns).mean()

mean_differences_products = []

for i, _ in enumerate(xeqt_prices):
    mean_differences_products.append((xeqt_prices[i] - xeqt_expected_return) * (vsp_prices[i] - vsp_expected_return))

print('--------------- Problem 1 ----------------')
print(f'Correlation coefficient matrix {np.corrcoef(xeqt_prices, vsp_prices)}')

print('--------------- Problem 2 & 3 ----------------')
xeqt_daily_log_returns = []

for i in xeqt_daily_returns:
    xeqt_daily_log_returns.append(np.log(1+i))

vsp_daily_log_returns = []

for i in vsp_daily_returns:
    vsp_daily_log_returns.append(np.log(1+i))

# print(f'XEQT Log returns {xeqt_daily_log_returns}')
# print(f'VSP Log returns {vsp_daily_log_returns}')

# time_series = np.arange(len(xeqt_prices)-1)
# plt.scatter(time_series, xeqt_daily_returns)
# plt.scatter(time_series, xeqt_daily_log_returns)
# plt.show()

# time_series = np.arange(len(vsp_prices)-1)
# plt.scatter(time_series, vsp_daily_returns)
# plt.scatter(time_series, vsp_daily_log_returns)
# plt.show()


print('--------------- Problem 4 ----------------')
xeqt_log_returns_mean = np.mean(xeqt_daily_log_returns) / 253
xeqt_log_returns_std = np.std(xeqt_daily_log_returns) / np.sqrt(253)

threshold = 1000000

simulation_count = 1000

intial_investment = 1000000

dipped_below_threshold_count = 0


# for i in range(simulation_count):
#     days_return = np.random.normal(xeqt_log_returns_mean, xeqt_log_returns_std, 45)

#     cum_returns = np.cumsum(days_return)

#     prices = intial_investment * np.exp(cum_returns)

#     for price in prices:
#         if price < threshold:
#             dipped_below_threshold_count += 1
#             break

# print(dipped_below_threshold_count / simulation_count)

print('--------------- Problem 5 ----------------')

dipped_above_threshold_count = 0
dipped_below_threshold = 0

for i in range(simulation_count):
    days_return = np.random.normal(xeqt_log_returns_mean, xeqt_log_returns_std, 100)

    cum_returns = np.cumsum(days_return)

    prices = intial_investment * np.exp(cum_returns)

    if prices[len(prices) - 1] > intial_investment:
        dipped_above_threshold_count += 1
        break

    if prices[len(prices) - 1] < intial_investment:
       dipped_below_threshold_count += 1
       break

p_dipping_above = dipped_above_threshold_count / simulation_count
p_dipping_below = dipped_below_threshold_count / simulation_count

# Section 2.5
# Problem 1a

below_990 = 0

for _ in range(10000):
    log_return = np.random.normal(0.001, 0.015)

    price = 1000 * np.exp(log_return)

    if price < 990:
        below_990 += 1

print(f'Probability of investment being worth under $990 after 1 day: {below_990/10000}')

below_990 = 0

# Problem 1b
for _ in range(10000):
    log_returns = np.random.normal(0.001, 0.015, 5)

    prices = 1000 * np.exp(log_returns)

    if any(price < 990 for price in prices):
        below_990 += 1

print(f'Probability of investment being worth under $990 after 5 days: {below_990/10000}')

# Problem 2
curr_price = 100
above_110_count = 0

for _ in range(10000):
    yearly_log_return = np.random.normal(0.1, 0.2)

    price = curr_price * np.exp(yearly_log_return)

    if price >= 110:
        above_110_count += 1

print(f'Probability of the stock increasing to $110 or more after a year is {above_110_count/10000}')

# Problem 3
curr_price = 80
above_90_count = 0

for _ in range(10000):
    yearly_log_returns = np.random.normal(0.08, 0.15, 2)

    cum_returns = np.sum(yearly_log_returns)

    price = curr_price * np.exp(cum_returns)

    if price >= 90:
        above_90_count += 1

print(f'Probability of the stock increasing to $90 or more after a 2 years is {above_90_count/10000}')

# Problem 4
p_1 = 95
p_2 = 103
p_3 = 98

r_3_2 = np.log(p_3 / p_1)

print(f'The log return from period 3 to period 1 is {r_3_2}')

# Problem 5
data = np.array([[1,2,3,4],
                 [52,54,53,59],
                 [0.2,0.2,0.2,0.25]
                ])

R_2 = ((data[1,1] + data[2,1]) / (data[1,0])) - 1
R_4_3 = (((data[1,3] + data[2,3]) / (data[1,2])) * ((data[1,2] + data[2,2]) / (data[1,1])) *  ((data[1,1] + data[2,1]) / (data[1,0]))) - 1
r_3 = np.log((data[1,2] + data[2,2]) / data[1,1])

# Problem 6
data = np.array([[1,2,3,4],
                 [82,85,83,87],
                 [0.1,0.1,0.1,0.125]
                ])

R_3_2 = ((data[1,2] + data[2,2]) / (data[1,1]) * (data[1,1] + data[2,1]) / (data[1,0])) - 1
r_4_3 = np.log(((data[1,3] + data[2,3]) / (data[1,2]))) + np.log(((data[1,2] + data[2,2]) / (data[1,1]))) + np.log((((data[1,1] + data[2,1]) / (data[1,0]))))

print(data[:,0])

# Problem 7a
mean = 0.06 * 4
std = 0.47 * 4

#Problem 10
above_100_count = 0
for _ in range(10000):
    daily_log_return = np.random.normal(0.0002, 0.03, 20)

    price = 97 * np.exp(np.sum(daily_log_return))

    if price > 100: 
        above_100_count += 1

print(f'The probability of the investment exceeding $100 after 20 trading days is {above_100_count/10000}')

# Problem 11
# t = 1
# probability = 0

# while probability < 0.9:
#     above_doubled_count = 0
#     for _ in range(1000):
#         daily_log_returns = np.random.normal(0.0005, 0.012, t)

#         price = 97 * np.exp(np.sum(daily_log_returns))

#         if price > (97 * 2):
#             above_doubled_count += 1
#     probability = above_doubled_count / 1000
#     t += 1
#     print(f'P {probability}')

# print(t)

    

















    







