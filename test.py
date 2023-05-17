from flask import Flask, request, render_template
import math
import random
import yfinance as yf
import pandas as pd
from datetime import date, timedelta
from pandas_datareader import data as pdr
import json
import numpy as np  # import numpy library for calculating simulated values

# override yfinance with pandas – seems to be a common step
yf.pdr_override()

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.htm')


@app.route('/calculate', methods=['POST'])
def calculate():
     # Get user inputs from the form
    service_type = request.form.get('service-type')
    num_resources = request.form.get('num-resources')
    price_history = int(request.form.get('price-history')) # convert price history to integer
    data_points = int(request.form.get('data-points'))
    transaction_type = request.form.get('transaction-type')
    profit_days = int(request.form.get('profit-days')) # convert profit days to integer

     # Define start and end dates for retrieving historical data
    today = date.today()
    decade_ago = today - timedelta(days=profit_days)
    

     # Get stock data
    data = pdr.get_data_yahoo('BP.L', start=decade_ago, end=today)
    data.reset_index(inplace=True)
    

    data['buy']=0
    data['sell']=0
    


     # Loop through each row of the data starting at the third row
    for i in range(2, len(data)):
    
     # Define the minimum body size for a signal
     min_body_size = 0.01

     # Check for Three Soldiers pattern
     if (data['Close'][i] - data['Open'][i]) >= min_body_size \
        and data['Close'][i] > data['Close'][i-1] \
        and (data['Close'][i-1] - data['Open'][i-1]) >= min_body_size \
        and data['Close'][i-1] > data['Close'][i-2] \
        and (data['Close'][i-2] - data['Open'][i-2]) >= min_body_size:
        
        # Set the 'Buy' column to 1 for the current row
        data.at[data.index[i], 'buy'] = 1
        #print(f"Buy at {data.index[i]}")

     # Check for Three Crows pattern
     if (data['Open'][i] - data['Close'][i]) >= min_body_size \
        and data['Close'][i] < data['Close'][i-1] \
        and (data['Open'][i-1] - data['Close'][i-1]) >= min_body_size \
        and data['Close'][i-1] < data['Close'][i-2] \
        and (data['Open'][i-2] - data['Close'][i-2]) >= min_body_size:
        
        # Set the 'Sell' column to 1 for the current row
        data.at[data.index[i], 'sell'] = 1
        #print(f"Sell at {data.index[i]}")

     # Data now contains signals, so we can pick signals with a minimum amount
     # of historic data, and use shots for the amount of simulated values
     # to be generated based on the mean and standard deviation of the recent history



    dict_of_lists = data.to_dict(orient='list')
    dict_of_lists['Date'] = [i.strftime('%Y-%m-%d') for i in dict_of_lists['Date']]
    var_95 = []
    var_99 = []
    tarik = []
     # Create an empty lsit to store simulation results
     # # Loop over the data to find signals and perform simulations
    for i in range(price_history, len(data)):
        
        if dict_of_lists['buy'][i] == 1 and transaction_type == 'buy':
            # Calculate mean and standard deviation using price history
            prices = [dict_of_lists['Close'][j] for j in range(i-price_history, i)]
            returns = [(prices[j] - prices[j-1])/prices[j-1] for j in range(1, price_history)]
            mean = sum(returns)/len(returns)
            std = ((sum([(x - mean)**2 for x in returns]))/(price_history - 1))**0.5

           # Generate a large number of random price changes with the same characteristics
            simulated = [random.gauss(mean, std) for x in range(data_points)]
           # Sort the simulated price changes in descending order
            simulated.sort(reverse=True)

            # Pick the 95th and 99th percentiles as risk measures
            var95 = simulated[int(len(simulated)*0.95)]
            var99 = simulated[int(len(simulated)*0.99)]
            tarik.append(dict_of_lists['Date'][i])
            var_99.append(var99)
            var_95.append(var95)
           # Store the results in the list
            

        elif dict_of_lists['sell'][i] == 1 and transaction_type == 'sell':
           # Calculate mean and standard deviation using price history
            prices = [dict_of_lists['Close'][j] for j in range(i-price_history, i)]
            returns = [(prices[j] - prices[j-1])/prices[j-1] for j in range(1, price_history)]
            mean = sum(returns)/len(returns)
            std = ((sum((x - mean)**2 for x in returns))/(price_history - 1))**0.5

           # Generate a large number of random price changes with the same characteristics
            simulated = [random.gauss(mean, std) for x in range(data_points)]

            # Sort the simulated price changes in ascending order
            simulated.sort()

          # Pick the 95th and 99th percentiles as risk measures
            var95 = simulated[int(len(simulated)*0.95)]
            var99 = simulated[int(len(simulated)*0.99)]


           # Store the results in the dictionary
            var_95.append(var95)
            var_99.append(var99)
            tarik.append(dict_of_lists['Date'][i])


           # now we have tarik, var_95, var_99
        result = []
        for i in range(len(tarik)):
            result.append([tarik[i], var_95[i], var_99[i]])



       
    return render_template("result.htm",data = result)

if __name__ == '__main__':
	app.run(debug=True)
        