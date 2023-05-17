import pandas as pd
import numpy as np
import json as js
import boto3
import random as rs
import time
from concurrent.futures import ThreadPoolExecutor
import math
import yfinance as yf
import http.client
from datetime import date, timedelta
import statistics
from pandas_datareader import data as pdr
from flask import Flask, request, render_template

# Define AWS credentials and client
aws_access_key_id = "ASIATGDG72GDVQBV7DSK"
aws_secret_access_key = "Mj/xKNKAsVHfEtMqO537bCl3v3FjilSdAHtDE2P0"
aws_session_token = "FwoGZXIvYXdzECkaDLn/OQk1sXkWb1uVoyLFAVAqZHaKOs5iJphUKXFPY1/GGwH7pS2ebuw23EW4G5iVZ1GTV7UDF5rmloEIMSA/zOgX/Yo0T/EJiOk/AuQASXESRtXUF04ypq8y0FGda2nHGUJAy2+yUpdqAVGPrgYuKiAgPKLKg3mtyFpE/9leZFWgZZtcVtfrsmr+wluP9i5e0903NYd9wtjC5jne2PszyScTOrtudUQBDMQBToz682SHbfkS4I7GKb8VOSsHZfZAIxBZKeeLDFuNH+BcLVnx/Fu+FHmQKNKGkqMGMi3UXJXsUfi8v2ox1t0ohUmZ5PovniDVG5SyfuWp8Uv2qVcXHQrPiCTGvRhavwY="
region_name = "us-east-1"
lmbda_client = boto3.client("lambda", aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key, aws_session_token=aws_session_token, region_name=region_name)

# Override yfinance with pandas
yf.pdr_override()

app = Flask(__name__)

# Add exception handler
@app.errorhandler(Exception)
def handle_exception(e):
    # Log the exception
    app.logger.error(str(e))

    # Render an error page
    return render_template('error.htm', error=str(e))

# Define your routes below
@app.route('/')
def home():
    return render_template('index.htm')

global runs
runs = []
@app.route("/calculate", methods=['POST'])
def calculate():
    # Taking inputs from the page
    service_type = request.form.get('service-type')
    numbers_resources = request.form.get('num-resources')
    price_history = request.form.get('price-history')
    data_points = request.form.get('data-points')
    transaction_type = request.form.get('transaction-type')
    profit_days = request.form.get('profit-days')

    runs = [value for value in range(int(numbers_resources))]

    # Define start and end dates for retrieving historical data
    today = date.today()
    decade_ago = today - timedelta(days=1095)

    # Get stock data
    data = pdr.get_data_yahoo('BP.L', start=decade_ago, end=today)
    data.reset_index(inplace=True)

    data["buy"] = 0
    data["sell"] = 0

        # Loop through each row of the data
    for i in range(2, len(data)):
        min_body_size = 0.01
        # Three Soldiers
        if (
            (data.Close[i] - data.Open[i]) >= min_body_size
            and data.Close[i] > data.Close[i - 1]
            and (data.Close[i - 1] - data.Open[i - 1]) >= min_body_size
            and data.Close[i - 1] > data.Close[i - 2]
            and (data.Close[i - 2] - data.Open[i - 2]) >= min_body_size
        ):
            data.at[data.index[i], 'buy'] = 1

        # Three Crows
        if (
            (data.Open[i] - data.Close[i]) >= min_body_size
            and data.Close[i] < data.Close[i - 1]
            and (data.Open[i - 1] - data.Close[i - 1]) >= min_body_size
            and data.Close[i - 1] < data.Close[i - 2]
            and (data.Open[i - 2] - data.Close[i - 2]) >= min_body_size
        ):
            data.at[data.index[i], 'sell'] = 1

        # Calculate the percentage change over the specified number of days
        data['Change'] = data['Close'].pct_change(periods=int(profit_days))

        # Check if profit or loss has been made after the specified number of days
        data['Profit_Loss'] = np.where(data['Change'] > 0, 'Profit', 'Loss')
        data['Change'].fillna(0, inplace=True)

    dict_of_lists = data.to_dict(orient='list')

    dict_of_lists['Date'] = [i.strftime('%Y-%m-%d') for i in dict_of_lists['Date']]
    ln_data = len(data)

    input_dict = {
        'service_type': str(request.form.get('service-type')),
        'numbers_resources': str(request.form.get('num-resources')),
        'price_history': str(request.form.get('price-history')),
        'data_points': str(request.form.get('data-points')),
        'transaction_type': str(request.form.get('transaction-type')),
        'profit_days': str(request.form.get('profit-days')),
        'dict_of_lists': js.dumps(dict_of_lists),
        'ln_data': str(ln_data)
    }

    json_data = js.dumps(input_dict)
    print(json_data)

    # Define the Lambda function name and payload
    function_name = 'cloud_computing'
    payload = json_data

    response = lmbda_client.invoke(
        FunctionName=function_name,
        Payload=payload
    )

    # Retrieve the response
    result = response['Payload'].read().decode('utf-8')
    result = eval(result)
    print("this is result", result)

    print(type(result))
    # return render_template("result.htm", data=result)

    def getpage(id):
            # Define the Lambda function name and payload
        function_name = 'cloud_computing'
        payload = json_data

        response = lmbda_client.invoke(
            FunctionName=function_name,
            Payload=payload
        )

        # Retrieve the response
        result = response['Payload'].read().decode('utf-8')
        result = eval(result)
        return result
    def getpages(runs):
        with ThreadPoolExecutor() as executor:
            results=executor.map(getpage, runs)
        return results
    print("xxxxxxxxxxxxxxxx",list(getpages(runs)))
    print(np.shape(list(getpages(runs))))
    return render_template("result.htm", data=result)


if __name__ == '__main__':
    app.run(debug=True)
