from concurrent.futures import ThreadPoolExecutor, as_completed, _base
from dateutil.relativedelta import relativedelta
from pandas import DataFrame as container
from bs4 import BeautifulSoup as parser
from collections import defaultdict
from datetime import datetime, date
from typing import Union

# from rich.progress import Progress

import os

import threading
import pandas as pd
import numpy as np
import requests

from bs4 import BeautifulSoup

class DataReader:

    headers = ['TIME', 'OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME']

    def __init__(self):
        self.__history = "https://dps.psx.com.pk/historical"
        self.__symbols = "https://dps.psx.com.pk/symbols"
        self.__local = threading.local()

    @property
    def session(self):
        if not hasattr(self.__local, "session"):
            self.__local.session = requests.Session()
        return self.__local.session

    def tickers(self):
        return pd.read_json(self.__symbols)

    def get_psx_data(self, symbol: str, dates: list) -> container:
        data = futures = []
        # with Progress() as progress:
            # task = progress.add_task(f"[cyan]Fetching {symbol} Stock Data History...", total=len(dates))

        with ThreadPoolExecutor(max_workers=6) as executor:
            for date in dates:
                futures.append(executor.submit(self.download, symbol=symbol, date=date))

            for future in as_completed(futures):
                data.append(future.result())
                    # progress.update(task, advance=1)
            
            data = [instance for instance in data if isinstance(instance, container)]
        
        return self.preprocess(data)
    
    def getStockHistory(self, tickers: Union[str, list], start: date, end: date) -> container:
        tickers = [tickers] if isinstance(tickers, str) else tickers
        dates = self.daterange(start, end)

        data = [self.get_psx_data(ticker, dates)[start: end] for ticker in tickers]

        # if len(data) == 1:
        #     return data[0]

        return pd.concat(data, keys=tickers, names=["Ticker", "Date"])


    def download(self, symbol: str, date: date):
        session = self.session
        post = {"month": date.month, "year": date.year, "symbol": symbol}
        with session.post(self.__history, data=post) as response:
            data = parser(response.text, features="html.parser")
            data = self.toframe(data)
        return data

    def toframe(self, data):
        stocks = defaultdict(list)
        rows = data.select("tr")

        for row in rows:
            cols = [col.getText() for col in row.select("td")]
        
            for key, value in zip(self.headers, cols):
                if key == "TIME":
                    value = datetime.strptime(value, "%b %d, %Y")
                stocks[key].append(value)

        return pd.DataFrame(stocks, columns=self.headers).set_index("TIME")

    def daterange(self, start: date, end: date) -> list:
        period = end - start
        number_of_months = period.days // 30
        current_date = datetime(start.year, start.month, 1)
        dates = [current_date]

        for month in range(number_of_months):
            prev_date = dates[-1]
            dates.append(prev_date + relativedelta(months=1))

        dates = dates if len(dates) else [start]
        return dates

    def preprocess(self, data: list) -> pd.DataFrame:
        # concatenate each frame to a single dataframe
        data = pd.concat(data)
        # sort the data by date
        data = data.sort_index()
        # change indexes from all uppercase to title
        data = data.rename(columns=str.title)
        # change index label Title to Date
        data.index.name = "Date"
        # remove non-numeric characters from volume column 
        data.Volume = data.Volume.str.replace(",", "")
        # coerce each column type to float
        for column in data.columns:
            data[column] = data[column].str.replace(",", "").astype(np.float64)
        return data
    


    def save_to_csv(self, data: pd.DataFrame, filename: str = "DATA/DATA.csv",comb: bool = True, sep: bool = False) -> None:
        """
        Save the DataFrame to a CSV file.
        If sep is True, save separate files for each ticker.
        
        :param data: The DataFrame to be saved.
        :param filename: Name of the combined file.
        :param sep: Whether to save separate files for each ticker.
        :param comb: Whether to save combined filed for each ticker .
        """
        if sep:
            # Ensure the directory exists
            os.makedirs('DATA', exist_ok=True)

            # Loop through each unique ticker and save it as a separate CSV file
            for ticker in data.index.get_level_values('Ticker').unique():
                ticker_data = data.xs(ticker, level='Ticker')
                ticker_filename = f"DATA/{ticker}.csv"
                ticker_data.to_csv(ticker_filename)
        if comb:
            # Save as a single file if sep is False
            data.to_csv(filename)
            
    def getStockDataCurrent(self, stockList, v=True):
        """
        Get Current Data
        """
        DATA = []
        # with Progress() as progress:
        #     # Define the task

        #     task = progress.add_task(f"[cyan]Fetching Stock Data...", total= len(stockList))
        for stockSymbol in stockList:
            url = f'https://dps.psx.com.pk/company/{stockSymbol}'  # URL based on PSX structure
            response = requests.get(url)
            soup = BeautifulSoup(response.content, 'html.parser')

            # Extract stock details based on the HTML structure you provided
            try:
                stats = soup.find_all('div', class_='stats_item')

                # Extract values from each stat item
                open_price = stats[0].find('div', class_='stats_value').text.strip()  # Open
                high_price = stats[1].find('div', class_='stats_value').text.strip()  # High
                low_price = stats[2].find('div', class_='stats_value').text.strip()  # Low
                volume = stats[3].find('div', class_='stats_value').text.strip()  # Volume
                current = soup.find('div', class_='quote__close').text.strip()

                DATA.append({
                    'open': open_price,
                    'high': high_price,
                    'low': low_price,
                    'volume': volume,
                    'current': current[3:]
                })
                
                # print(current)
            except Exception as e:
                # print(f"Error retrieving data for {stockSymbol}: {e}")
                DATA.append(None)

                # Update the progress bar
                # progress.update(task, advance=1)
        
        return DATA

data_reader = DataReader()

if __name__ == "__main__":
    # data = data_reader.getStockHistory(["MARI", "GAL"], date(2025, 1, 1), date(2025, 4, 5))
    # data_reader.save_to_csv(data, comb=True, sep=True)
    # print(data)
    data = data_reader.getStockDataCurrent(["MARI"])
    # print(data)