import os
import time
import json as js
import boto3
import botocore
import yfinance as yf
from statistics import mean
from flask import Flask, render_template, request, redirect,jsonify
from pandas_datareader import data as pdr
from datetime import date, timedelta
from concurrent.futures import ThreadPoolExecutor
from io import StringIO
import pandas as pd
import numpy as np
import requests

# Load the credentials
with open("credentials.txt", "r") as file:
    exec(file.read())

lmbda_client = boto3.client("lambda", aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key, aws_session_token=aws_session_token, region_name=region_name)
s3 = boto3.resource("s3", aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key, aws_session_token=aws_session_token, region_name=region_name)
ec_2 = boto3.resource("ec2", aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key, aws_session_token=aws_session_token, region_name=region_name)

# Override yfinance with pandas
yf.pdr_override()

app = Flask(__name__)
ec2_urls = []
instance_id =[]


def get_stock_data():
    today = date.today()
    decade_ago = today - timedelta(days=1095)
    stock_data = pdr.get_data_yahoo('BP.L', start=decade_ago, end=today)
    return stock_data


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


@app.route("/calculate", methods=['POST', 'GET'])
def calculate():
    # start the timer
    start_time = time.time()
    # Taking inputs from the page
    service_type = request.form.get('service-type')
    numbers_resources = request.form.get('num-resources')
    price_history = request.form.get('price-history')
    data_points = request.form.get('data-points')
    transaction_type = request.form.get('transaction-type')
    profit_days = request.form.get('profit-days')

    runs = range(int(numbers_resources))
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

    if service_type == "ec2":
        invoke_ec2(json_data, [f'http://{i}/helloapache.py' for i in ec2_urls])

    # Define the Lambda function name and payload
    function_name = 'cloud_computing'
    payload = json_data

    response = lmbda_client.invoke(
        FunctionName=function_name,
        Payload=payload
    )

    # Retrieve the response
    lambda_result = js.loads(response['Payload'].read().decode('utf-8'))
    print(type(lambda_result))
    print("this is result", lambda_result)

    
    # return render_template("result.htm", data=result)

    def getpage(runs):
        # Define the Lambda function name and payload
        function_name = 'cloud_computing'
        payload = json_data

        response = lmbda_client.invoke(
            FunctionName=function_name,
            Payload=payload
        )

        # Retrieve the response
        lambda_result = js.loads(response['Payload'].read().decode('utf-8'))
        return lambda_result

    def getpages(runs):
        with ThreadPoolExecutor() as executor:
            results = executor.map(getpage, runs)
        return results
    
    multi_thrd = list(getpages(runs))
    print(type(multi_thrd))
    print("xxxxxxxxxxxxxxxx", multi_thrd)

    # calculate elapsed_time
    elapsed_time = time.time() - start_time

    dataframes = [pd.json_normalize(js.loads(jd)) for jd in multi_thrd]
    raw_data = pd.concat(dataframes, ignore_index=True)

    # Group raw_data by 'Date' and calculate mean for 'Var95' and 'Var99'
    grouped_var_data = raw_data.groupby('Date').agg({'Var95': 'mean', 'Var99': 'mean'}).reset_index()

    # Group raw_data by 'Date' and calculate most frequent 'Profit_Loss' and sum 'Change'
    grouped_profit_change_data = raw_data.groupby('Date').agg({'Profit_Loss': lambda x: x.value_counts().index[0], 'Change': 'sum'}).reset_index()

    # Merge these two dataframes together
    final_grouped_data = pd.merge(grouped_var_data, grouped_profit_change_data, on='Date')

    # Now final_grouped_data is the new DataFrame with 'Date', 'Var95', 'Var99', 'Profit_Loss', and 'Change'


    print("--------------------------x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x--------------------------","\n",raw_data)
    # Calculate total_change
    total_change = raw_data['Change'].sum()

    # Calculate averages

    #overall mean
    totall_var95_mean= final_grouped_data["Var95"].mean()
    totall_var99_mean = final_grouped_data["Var99"].mean()

    print("--------------------------x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x--------------------------","\n",totall_var99_mean,totall_var95_mean,total_change,"\n","--------------------------x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x--------------------------",)


    print(np.shape(raw_data))



    memory_in_gb = 128 / 1024

    # Calculate cost per second
    cost_per_second = 0.0000166667 * memory_in_gb

    # Calculate total cost
    total_cost = cost_per_second * elapsed_time

    #aws data 

    print("#########################################################################",total_cost)



    # Create new DataFrame
    df_today = pd.DataFrame([{
        'Date': today.strftime('%Y-%m-%d'),
        'Service_Type': service_type,
        'Number_of_Resources': numbers_resources,
        'Price_History': price_history,
        'Data_Points': data_points,
        'Transaction_Type': transaction_type,
        'Profit_Days': profit_days,
        'Elapsed_Time': elapsed_time,
        'Total_Change': total_change,
        'Var95_Mean': totall_var99_mean,  # Add today's average of Var95
        'Var99_Mean': totall_var99_mean,
        'Lambda_Cost': total_cost 
    }])

    # Convert DataFrame to csv - assuming CSV is the format you want
    csv_data = df_today.to_csv(index=False)

    print("========================================================================================================","\n",df_today)

    # Save to S3
    bucket_name = "cldcmptng"
    file_name = "audit_data.csv"

    # Read existing data from S3
    try:
        obj = s3.Object(bucket_name, file_name)
        previous_data = obj.get()['Body'].read().decode('utf-8')

        # Check if the file has a header
        if previous_data:
            # Don't write header, because the file already has a header
            csv_data = df_today.to_csv(index=False, header=False)
            # concatenate old data with new data
            total_csv_data = previous_data + "\n" + csv_data
        else:
            # Write header, because the file is empty
            csv_data = df_today.to_csv(index=False)
            # Start a new file with the data
            total_csv_data = csv_data

    except botocore.exceptions.ClientError as e:
        # If the file didn't exist, just write the new data
        if e.response['Error']['Code'] == "NoSuchKey":
            csv_data = df_today.to_csv(index=False)
            total_csv_data = csv_data
        else:
            raise

    s3.Bucket(bucket_name).put_object(Key=file_name, Body=total_csv_data)


    raw_data = final_grouped_data.to_dict(orient='records')
    print(">>>>>>>>>>>>>>>>>>>>>", raw_data)
    print("xxxxxxxxxxxxxxxxxx----------------------xxxxxxxxxxxxxxxxxxx----------xxxxxxxxx", "\n",csv_data)

    return render_template("result.htm", data=raw_data,elapsed_time = elapsed_time,total_change=total_change)

def invoke_ec2(payload, urls=["http://ec2-54-84-13-180.compute-1.amazonaws.com/helloapache.py"]):
    
    data = []
    
    for url in urls:
        response = requests.post(url,
                                data = payload)
        
        print("£££££££££££££££££££££££££££££££££££££££££££")
        
        print(response.content)
        data.append(eval(response.content))



    return data


@app.route("/audit")
def audit():
    bucket_name = "cldcmptng"
    file_name = "audit_data.csv"

    # Create an S3 client
    s3 = boto3.client("s3", aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key, aws_session_token=aws_session_token, region_name=region_name)

    # Get object from S3
    response = s3.get_object(Bucket=bucket_name, Key=file_name)

    # Read the CSV data from the S3 object
    data = response["Body"].read().decode("utf-8")
    data = pd.read_csv(StringIO(data))


    # Rename the columns
    new_column_names = {
        "Date": "Audit Date",
        "Service_Type": "Service Type",
        "Number_of_Resources": "Number of Resources",
        "Price_History": "Price History",
        "Data_Points": "Data Points",
        "Transaction_Type": "Transaction Type",
        "Profit_Days": "Profit Days",
        "Elapsed_Time": "Elapsed Time",
        "Total_Change": "Total Change",
        "Var95_Mean": "Var95 Mean",
        "Var99_Mean": "Var99 Mean",
        "Lambda_Cost": "Costing"
    }
    data = data.rename(columns=new_column_names)
    table = data.to_html(index=False)

    print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@",table)
    return render_template("audit.htm", table=table)



def save_data(data):
    bucket_name = "cldcmptng"
    file_name = "audit_data.csv"

    # Create an S3 client
    s3 = boto3.client("s3", aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key,
                      aws_session_token=aws_session_token, region_name=region_name)

    # Convert DataFrame to CSV
    csv_data = data.to_csv(index=False)

    # Upload CSV data to S3
    s3.put_object(Body=csv_data, Bucket=bucket_name, Key=file_name)



@app.route("/reset", methods=["POST"])
def reset():
    # Create an empty DataFrame
    empty_data = pd.DataFrame(columns=[
        "Date", "Service_Type", "Number_of_Resources", "Price_History", "Data_Points",
        "Transaction_Type", "Profit_Days", "Elapsed_Time", "Total_Change", "Var95_Mean", "Var99_Mean","Costing $"
    ])

    # Save empty DataFrame to S3
    save_data(empty_data)

    return redirect("/")



@app.route("/warmup", methods=['POST'])
def EC2():
    numbers_resources = int(request.form.get('number_of_resources'))

    # Create the EC2 instance
    instances = ec_2.create_instances(
        ImageId='ami-0554b0c22f2e34c67',  # replace with your AMI ID
        MinCount=numbers_resources,
        MaxCount= numbers_resources,
        InstanceType='t2.micro',
        SecurityGroupIds=['sg-0137c6e26c08f62de'],  # replace with your security group ID
    )

    for instance in instances:
        instance.wait_until_running()
        instance.reload()
        ec2_urls.append(instance.public_dns_name)

    print(ec2_urls)

@app.route("/terminate-button", methods=['POST'])
def terminate_ec2():
    ec_2.instances.filter(InstanceIds=instance_id).terminate()

    return jsonify({'status': 'success'}), 200



if __name__ == '__main__':
    start = time.time()

    app.run(debug=True)


