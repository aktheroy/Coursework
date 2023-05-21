import json
import sys
import random as rs

input=sys.std.read()
numbers_resources = int(event['numbers_resources'])
service_type = str(event['service_type'])
price_history = int(event['price_history'])
profit_days = int(event['profit_days'])
ln_data = int(event['ln_data'])
transaction_type = str(event['transaction_type'])
data_points = str(event['data_points'])
dict_of_lists = eval(event['dict_of_lists'])

var_95 = []
var_99 = []
tarik  = []
hint   = []
result = []
change = []

for i in range(price_history, ln_data):
    if dict_of_lists["buy"][i] == 1 and transaction_type == "buy":
        # Calculate mean and standard deviation using price history
        prices = [dict_of_lists['Close'][j] for j in range(i - price_history, i)]
        returns = [(prices[j] - prices[j - 1]) / prices[j - 1] for j in range(1, price_history)]
        mean = sum(returns) / len(returns)
        std = ((sum([(x - mean) ** 2 for x in returns])) / (price_history - 1)) ** 0.5

        simulated = [rs.gauss(mean, std) for _ in range(int(data_points))]
        simulated.sort(reverse=True)

        var95 = simulated[int(len(simulated) * 0.95)]
        var99 = simulated[int(len(simulated) * 0.99)]
        tarik.append(dict_of_lists['Date'][i])

        change.append(round(dict_of_lists['Close'][i] - dict_of_lists['Close'][i - profit_days], 2))
        var_99.append(var99)
        var_95.append(var95)
        hint.append(dict_of_lists['Profit_Loss'][i])

    elif dict_of_lists["sell"][i] == 1 and transaction_type == "sell":
        prices = [dict_of_lists['Close'][j] for j in range(i - price_history, i)]
        returns = [(prices[j] - prices[j - 1]) / prices[j - 1] for j in range(1, price_history)]
        mean = sum(returns) / len(returns)
        std = ((sum((x - mean) ** 2 for x in returns)) / (price_history - 1)) ** 0.5

        simulated = [rs.gauss(mean, std) for _ in range(int(data_points))]
        simulated.sort()

        var95 = simulated[int(len(simulated) * 0.95)]
        var99 = simulated[int(len(simulated) * 0.99)]
        tarik.append(dict_of_lists['Date'][i])
        hint.append(dict_of_lists['Profit_Loss'][i])
        change.append(round(dict_of_lists['Close'][i] - dict_of_lists['Close'][i - profit_days], 2))
        var_99.append(var99)
        var_95.append(var95)

print(var_95, var_99, tarik, change)
for i in range(len(tarik)):
    result.append({"Date": tarik[i], "Var95": var_95[i], "Var99": var_99[i], "Profit_Loss": hint[i], "Change": change[i]})


