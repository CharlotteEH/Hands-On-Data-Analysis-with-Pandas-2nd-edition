# 1-wide_vs_long.ipynb

import matplotlib.pyplot as plt
import pandas as pd

wide_df = pd.read_csv('C:/Users/charlotte.henstock/PycharmProjects/pythonProject2/ch_03/data/wide_data.csv',
                      parse_dates=['date'])
long_df = pd.read_csv(
    'C:/Users/charlotte.henstock/PycharmProjects/pythonProject2/ch_03/data/long_data.csv',
    usecols=['date', 'datatype', 'value'],
    parse_dates=['date']
)[['date', 'datatype', 'value']]

print(wide_df.head(6))

print(wide_df.describe(include='all', datetime_is_numeric=True))

wide_df.plot(
    x='date', y=['TMAX', 'TMIN', 'TOBS'], figsize=(15, 5),
    title='Temperature in NYC in October 2018'
).set_ylabel('Temperature in Celsius')
# plt.show()

print(long_df.head(6))
print(long_df.describe(include='all', datetime_is_numeric=True))

import seaborn as sns

sns.set(rc={'figure.figsize': (15, 5)}, style='white')

ax = sns.lineplot(
    data=long_df, x='date', y='value', hue='datatype'
)
ax.set_ylabel('Temperature in Celsius')
ax.set_title('Temperature in NYC in October 2018')
# plt.show()0

sns.set(
    rc={'figure.figsize': (20, 10)}, style='white', font_scale=2
)

g = sns.FacetGrid(long_df, col='datatype', height=10)
g = g.map(plt.plot, 'date', 'value')
g.set_titles(size=25)
g.set_xticklabels(rotation=45)
# plt.show()

# 2-using_the_weather_api.ipynb

import requests


def make_request(endpoint, payload=None):
    """Make a request to a specific endpoint on the weather API passing headers and optional payload.

        Parameters:
            -endpoint: The endpoint of the API you want to make a GET request to.
            -payload: A dictionary of data to pass along with the request.

        Returns:
            A response object.
        """
    return requests.get(
        f'https://www.ncdc.noaa.gov/cdo-web/api/v2/{endpoint}',
        headers={
            'token': 'DMGnIKTbkxBiFFPsxSYHSEGqEoePalpo'
        },
        params=payload
    )


response = make_request('datasets', {'startdate': '2018-10-01'})
print(response.status_code)
response.ok

payload = response.json()
payload.keys()
payload['metadata']

payload['results'][0].keys()
[(data['id'], data['name']) for data in payload['results']]

response = make_request(
    'datacategories', payload={'datasetid': 'GHCND'}
)
response.status_code
response.json()['results']

response = make_request(
    'datatypes',
    payload={
        'datacategoryid': 'TEMP',
        'limit': 100
    }
)
response.status_code

[(datatype['id'], datatype['name']) for datatype in response.json()['results']][-5:]

response = make_request(
    'locationcategories',
    payload={'datasetid': 'GHCND'}
)
response.status_code

import pprint

pprint.pprint(response.json())


def get_item(name, what, endpoint, start=1, end=None):
    """
    Grab the JSON payload for a given field by name using binary search.

    Parameters:
        - name: The item to look for.
        - what: Dictionary specifying what the item in `name` is.
        - endpoint: Where to look for the item.
        - start: The position to start at. We don't need to touch this, but the
                 function will manipulate this with recursion.
        - end: The last position of the items. Used to find the midpoint, but
               like `start` this is not something we need to worry about.

    Returns:
        Dictionary of the information for the item if found otherwise
        an empty dictionary.
    """
    # find the midpoint which we use to cut the data in half each time
    mid = (start + (end or 1)) // 2

    # lowercase the name so this is not case-sensitive
    name = name.lower()

    # define the payload we will send with each request
    payload = {
        'datasetid': 'GHCND',
        'sortfield': 'name',
        'offset': mid,  # we will change the offset each time
        'limit': 1  # we only want one value back
    }

    # make our request adding any additional filter parameters from `what`
    response = make_request(endpoint, {**payload, **what})

    if response.ok:
        payload = response.json()

        # if response is ok, grab the end index from the response metadata the first time through
        end = end or payload['metadata']['resultset']['count']

        # grab the lowercase version of the current name
        current_name = payload['results'][0]['name'].lower()

        # if what we are searching for is in the current name, we have found our item
        if name in current_name:
            return payload['results'][0]  # return the found item
        else:
            if start >= end:
                # if our start index is greater than or equal to our end, we couldn't find it
                return {}
            elif name < current_name:
                # our name comes before the current name in the alphabet, so we search further to the left
                return get_item(name, what, endpoint, start, mid - 1)
            elif name > current_name:
                # our name comes after the current name in the alphabet, so we search further to the right
                return get_item(name, what, endpoint, mid + 1, end)
    else:
        # response wasn't ok, use code to determine why
        print(f'Response not OK, status: {response.status_code}')

nyc=get_item('New York',{'locationcategoryid':'CITY'}, 'locations')
nyc

central_park=get_item('NY City Central Park',{'locationid':nyc['id']}, 'stations')
central_park


response=make_request(
    'data',
    {
        'datasetid': 'GHCND',
        'stationid': central_park['id'],
        'locationid':nyc['id'],
        'startdate':'2018-10-01',
        'enddate':'2018-10-31',
        'datatypeid':['TAVG','TMAX','TMIN'],
        'units':'metric',
        'limit':1000
    }
)
response.status_code

import pandas as pd
df=pd.DataFrame(response.json()['results'])
df.head()

df.datatype.unique()


if get_item(
    'NY City Central Park', {'locationid':nyc['id'],'datatype':'TAVG'}, 'stations'
):
    print('Found!')

laguardia=get_item(
    'LaGuardia',{'locationid':nyc['id']},'stations'
)
laguardia

response=make_request(
    'data',
    {
        'datasetid':'GHCND',
        'stationid':laguardia['id'],
        'locationid':nyc['id'],
        'startdate':'2018-10-01',
        'enddate':'2018-10-31',
        'datatypeid':['TAVG','TMAX','TMIN'],
        'units':'metric',
        'limit':1000
    }
)
response.status_code

df=pd.DataFrame(response.json()['results'])
df.head()

df.datatype.value_counts()
df.to_csv('C:/Users/charlotte.henstock/PycharmProjects/pythonProject2/ch_03/data/nyc_temperatures.csv',index=False)

# 3-cleaning_data.ipynb

df.columns
df.rename(
    columns={
        'value':'temp_C',
        'attributes':'flags'
    }, inplace=True
)

df.columns
df.rename(str.upper,axis='columns').columns
df.columns
df.dtypes

df.loc[:,'date']=pd.to_datetime(df.date)
df.dtypes

df.date.describe(datetime_is_numeric=True)
pd.date_range(start='2018-10-25',periods=2,freq='D').tz_localize('EST')

eastern=pd.read_csv(
    'C:/Users/charlotte.henstock/PycharmProjects/pythonProject2/ch_03/data/nyc_temperatures.csv', index_col='date',parse_dates=True
).tz_localize('EST')
eastern.head()

eastern.tz_convert('UTC').head()
eastern.tz_localize(None).to_period('M').index

eastern.tz_localize(None).to_period('M').to_timestamp().index

df=pd.read_csv('C:/Users/charlotte.henstock/PycharmProjects/pythonProject2/ch_03/data/nyc_temperatures.csv').rename(
    columns={
        'value':'temp_C',
        'attributes':'flags'
    }
)

new_df = df.assign(
    date = pd.to_datetime(df.date),
    temp_F=(df.temp_C * 9/5)+32
)
new_df.dtypes

new_df.head()

df=df.assign(
    date=lambda x:pd.to_datetime(x.date),
    temp_C_whole=lambda x: x.temp_C.astype('int'),
    temp_F=lambda x:(x.temp_C*9/5)+32,
    temp_F_whole=lambda x:x.temp_F.astype('int')
)
df.head()

df_with_categories=df.assign(
    station=df.station.astype('category'),
    datatype=df.datatype.astype('category')
)
df_with_categories.dtypes

df_with_categories.describe(include='category')

pd.Categorical(
    ['med','med','low','high'],
    categories=['low','med','high'],
    ordered=True
)

df[df.datatype=='TMAX'].sort_values(by='temp_C',ascending=False).head(10)

df[df.datatype=='TMAX'].sort_values(by=['temp_C','date'],ascending=[False, True],ignore_index=True).head(10)

df[df.datatype=='TAVG'].nlargest(n=10,columns='temp_C')

df.nsmallest(n=5,columns=['temp_C','date'])

df.sample(5,random_state=0).index
df.sample(5,random_state=-0).sort_index().index

df.sort_index(axis=1).head()
df.sort_index(axis=1).head().loc[:,'temp_C':'temp_F_whole']

df.equals(df.sort_values(by='temp_C'))
df.equals(df.sort_values(by='temp_C').sort_index())

df.set_index('date',inplace=True)
df.head()
df['2018-10-11':'2018-10-12']

df['2018-10-11':'2018-10-12'].reset_index()

sp=pd.read_csv(
    'C:/Users/charlotte.henstock/PycharmProjects/pythonProject2/ch_03/data/sp500.csv',index_col='date',parse_dates=True
).drop(columns=['adj_close'])

sp.head(10).assign(
    day_of_week=lambda x: x.index.day_name()
)

bitcoin=pd.read_csv(
    'C:/Users/charlotte.henstock/PycharmProjects/pythonProject2/ch_03/data/bitcoin.csv',index_col='date',parse_dates=True
).drop(columns=['market_cap'])
portfolio=pd.concat([sp,bitcoin],sort=False).groupby(level='date').sum()
portfolio.head(10).assign(
    day_of_week=lambda x: x.index.day_name()
)

import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter

ax=portfolio['2017-Q4':'2018-Q2'].plot(
    y='close',figsize=(15,5),legend=False,
    title='Bitcoin +S&P 500 value without accounting for different indices'
)

ax.set_ylabel('price')
ax.yaxis.set_major_formatter(StrMethodFormatter('${x:,.0f}'))
for spine in ['top','right']:
    ax.spines[spine].set_visible(False)

plt.show()

sp.reindex(bitcoin.index).head(10).assign(
    day_of_week=lambda x: x.index.day_name()
)

sp.reindex(bitcoin.index, method='ffill').head(10)\
    .assign(day_of_week = lambda x: x.index.day_name())

sp.reindex(bitcoin.index)\
    .compare(sp.reindex(bitcoin.index,method='ffill'))\
    .head(10).assign(day_of_week=lambda x: x.index.day_name())

import numpy as np

sp_reindexed = sp.reindex(bitcoin.index).assign(
    volume=lambda x: x.volume.fillna(0),
    close=lambda x: x.close.fillna(method='ffill'),
    open=lambda x: np.where(x.open.isnull(), x.close, x.open),
    high=lambda x: np.where(x.high.isnull(),x.close, x.high),
    low=lambda x: np.where(x.low.isnull(),x.close, x.low)
)
sp_reindexed.head(10).assign(
    day_of_week=lambda x: x.index.day_name()
)