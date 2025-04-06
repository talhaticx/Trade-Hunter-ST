from dataReader import DataReader
from datetime import date, timedelta
import os

import pandas as pd

Scraper = DataReader()

start_date = date.today() - timedelta(days=120)  # Correct start date
end_date = date.today()

# Path to the date file
file_path = "DATA/date.txt"

def csvToData():
    """Convert the CSV files in the 'DATA' folder to pandas DataFrames."""
    dataframes = {}
    data_folder = 'DATA'

    for filename in os.listdir(data_folder):
        file_path = os.path.join(data_folder, filename)

        if filename.endswith(".csv"):
            df = pd.read_csv(file_path)
            ticker = filename.replace('.csv', '')
            dataframes[ticker] = df
    
    return dataframes

def scrapDataHistory(tickers):
    # Check if the date file exists and read its content
    os.makedirs('DATA/', exist_ok=True)
    
    if os.path.exists(file_path):
        with open(file_path, "r") as file:
            content = file.read()
            if content != str(end_date):
                # Fetch new data if the date doesn't match
                data = Scraper.getStockHistory(tickers, start_date, end_date)
                Scraper.save_to_csv(data)
                
                # Update the date.txt with the current date
                with open(file_path, "w") as file:
                    file.write(str(end_date))
                
                return data
            else:
                # print("CSV to data")
                # If data is up-to-date, load from CSV
                data = pd.read_csv("DATA/DATA.csv", index_col=['Ticker', 'Date'], parse_dates=['Date'])
                # print(data)
                return data
    else:
        # If the file doesn't exist, fetch and save the data
        data = Scraper.getStockHistory(tickers, start_date, end_date)
        Scraper.save_to_csv(data)
        
        # Create the date.txt file and write the current date
        with open(file_path, "w") as file:
            file.write(str(end_date))
        
        return data

def getCurrentData(tickers):
    data = Scraper.getStockDataCurrent(tickers)
    return data