from requests import get
from json import loads
import pandas as pd


URL_ENDPOINT = 'https://min-api.cryptocompare.com/data/v2/histoday'
SYMBOLS = ['btc', 'eth', 'xrp', 'usdt', 'usdc',
           'busd', 'bnb', 'tusd', 'doge', 'shib']


class CryptoCompareConnector:

  @staticmethod
  def data_histoday(symbol: str) -> pd.DataFrame:
    '''Retrieves data from the [Daily Pair OHLCV service](https://min-api.cryptocompare.com/documentation?key=Historical&cat=dataHistoday) offered by CryptoCompare:

    - `time`: Timestamp of the specific day for which the data is provided.
    - `high`: Highest price of the CC during the specified day.
    - `low`: Lowest price of the CC during the specified day.
    - `open`: Opening price of the CC at the beginning of the specified day.
    - `volumefrom`: Trading volume of the CC, indicating the quantity of the CC traded "from" during the specified day.
    - `volumeto`: Total value or monetary volume of the CC traded "to" during the specified day. This value is usually measured in the currency in which the trading volume is denominated (e.g., USD).
    - `close`: Closing price of the CC at the end of the specified day.
    - `conversionType`: Type of conversion used for the price data. In this case, it is mentioned as "direct," indicating that the prices provided are direct market prices without any additional conversions or adjustments.
    - `conversionSymbol`: Currency code used for conversion, if any. For example, if the prices were converted to a different currency, the conversion symbol would specify the target currency code.

    Args:
      symbol (str): The cryptocurrency symbol of interest [ Min length - 1] [ Max length - 30]

    Returns:
      pd.DataFrame: ...
    '''
    if symbol not in SYMBOLS:
      raise ValueError(f'expected one of {SYMBOLS}, got {symbol} instead')
    res = get(URL_ENDPOINT, {'fsym': symbol, 'tsym': 'usd', 'limit': 1500})
    response_text_json = loads(res.text)
    df = pd.DataFrame(response_text_json['Data']['Data'])
    # df['time'] = pd.to_datetime(df['time'], unit='s') # converting the time column to a human-readable format
    return df.drop(['conversionType', 'conversionSymbol'], axis=1)

  @staticmethod
  def
