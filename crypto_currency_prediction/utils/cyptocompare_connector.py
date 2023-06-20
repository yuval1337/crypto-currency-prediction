from requests import get
from json import loads
import pandas as pd


SYMBOLS = ['btc', 'eth', 'xrp', 'usdt', 'usdc',
           'busd', 'bnb', 'tusd', 'doge', 'shib']
CURRENCIES = ['usd', 'eur']
MAX_LIMIT = 2000


class CryptoCompareConnector:
  @staticmethod
  def data_histoday(symbol: str, to: str, limit: int = MAX_LIMIT) -> pd.DataFrame:
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
    '''
    URL_ENDPOINT = 'https://min-api.cryptocompare.com/data/v2/histoday'
    CryptoCompareConnector.check_args(symbol, to)
    # send an HTTP request, get the response
    res = get(URL_ENDPOINT, {'fsym': symbol, 'tsym': to, 'limit': limit})
    # deserialize the HTTP response
    response_text_json = loads(res.text)
    # create a DataFrame from the JSON
    df = pd.DataFrame(response_text_json['Data']['Data'])
    # drop unnecessary columns from the DataFrame
    df = df.drop(['time', 'conversionType', 'conversionSymbol'], axis=1)
    return df

  @staticmethod
  def check_args(symbol: str, to: str) -> None:
    if symbol not in SYMBOLS:
      raise ValueError(f'expected one of {SYMBOLS}, got {symbol} instead')
    if to not in CURRENCIES:
      raise ValueError(f'expected one of {CURRENCIES}, got {to} instead')
