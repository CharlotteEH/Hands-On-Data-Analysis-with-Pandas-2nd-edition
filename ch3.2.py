# 4-reshaping_data.ipynb

import pandas as pd

long_df=pd.read_csv(
    'C:/Users/charlotte.henstock/PycharmProjects/pythonProject2/ch_03/data/long_data.csv', usecols=['date','datatype','value']
).rename(
    columns={'value':'temp_C'}
).assign(
    date=lambda x: pd.to_datetime(x.date),
    temp_F=lambda x: (x.temp_C * 9/5) +32
)
long_df.head()

long_df.set_index('date').head(6).T

pivoted_df=long_df.pivot(
    index='date',columns='datatype',values='temp_C'
)
pivoted_df.head()

pivoted_df.describe()

pivoted_df=long_df.pivot(
    index='date',columns='datatype',values=['temp_C','temp_F']
)
pivoted_df.head()

pivoted_df['temp_F']['TMIN'].head()

multi_index_df=long_df.set_index(['date','datatype'])
multi_index_df.head().index

unstacked_df=multi_index_df.unstack()
unstacked_df.head()

extra_data=long_df.append([{
    'datatype':'TAVG',
    'date':'2018-10-01',
    'temp_C': 10,
    'temp_F': 50
}]).set_index(['date','datatype']).sort_index()
extra_data['2018-10-01':'2018-10-02']

extra_data.unstack().head()
extra_data.unstack(fill_value=-40).head()

wide_df=pd.read_csv('C:/Users/charlotte.henstock/PycharmProjects/pythonProject2/ch_03/data/wide_data.csv')
wide_df.head()

melted_df=wide_df.melt(
    id_vars='date',
    value_vars=['TMAX','TMIN','TOBS'],
    value_name='temp_C',
    var_name='measurement'
)
melted_df.head()

wide_df.set_index('date',inplace=True)
wide_df.head()

stacked_series=wide_df.stack()
stacked_series.head()

stacked_df=stacked_series.to_frame('values')
stacked_df.head()

stacked_df.head().index
stacked_df.index.names

stacked_df.index.set_names(['date','datatype'],inplace=True)
stacked_df.index.names

df=pd.read_csv('C:/Users/charlotte.henstock/PycharmProjects/pythonProject2/ch_03/data/dirty_data.csv')

df.head()
df.describe()
df.info()

contain_nulls=df[
    df.SNOW.isna() | df.SNWD.isna() | df.TOBS.isna()
    | df.WESF.isna() | df.inclement_weather.isna()
]
contain_nulls.shape[0]

contain_nulls.head(10)
df[df.inclement_weather=='NaN'].shape[0]

import numpy as np
df[df.inclement_weather==np.nan].shape[0]

df[df.inclement_weather.isna()].shape[0]
df[df.SNWD.isin([-np.inf,np.inf])].shape[0]

def get_inf_count(df):
    """Find the number of inf/-inf values per column in the dataframe"""
    return {
        col: df[df[col].isin([np.inf, -np.inf])].shape[0] for col in df.columns
    }

get_inf_count(df)

pd.DataFrame({
    'np.inf Snow Depth': df[df.SNWD == np.inf].SNOW.describe(),
    '-np.inf Snow Depth': df[df.SNWD == -np.inf].SNOW.describe()
}).T

df.describe(include='object')
df.value_counts()

df[df.duplicated()].shape[0]

df[df.duplicated(keep=False)].shape[0]
df[df.duplicated(['date','station'])].shape[0]

df[df.duplicated()].head()
df[df.WESF.notna()].station.unique()

df.date=pd.to_datetime(df.date)
station_qm_wesf=df[df.station=='?'].drop_duplicates('date').set_index('date').WESF
df.sort_values('station', ascending=False, inplace=True)

df_deduped=df.drop_duplicates('date')
df_deduped = df_deduped.drop(columns='station').set_index('date').sort_index()

df_deduped=df_deduped.assign(
    WESF=lambda x: x.WESF.combine_first(station_qm_wesf)
)
df_deduped.shape

df_deduped.head()
df_deduped.dropna().shape
df_deduped.dropna(how='all').shape

df_deduped.dropna(
    how='all',subset=['inclement_weather','SNOW','SNWD']
).shape

df_deduped.dropna(axis='columns',thresh=df_deduped.shape[0]*0.75).columns

df_deduped.loc[:,'WESF'].fillna(0,inplace=True)
df_deduped.head()

df_deduped=df_deduped.assign(
    TMAX=lambda x: x.TMAX.replace(5505,np.nan),
    TMIN=lambda x: x.TMIN.replace(-40, np.nan)
)

df_deduped.assign(
    TMAX=lambda x: x.TMAX.fillna(method='ffill'),
    TMIN=lambda x: x.TMIN.fillna(method='ffill')
).head()

df_deduped.assign(
    TMAX=lambda x: x.TMAX.fillna(method='ffill'),
    TMIN=lambda x: x.TMIN.fillna(method='ffill')
).head()

df_deduped.assign(
    SNWD=lambda x: x.SNWD.clip(0,x.SNOW)
).head()

df_deduped.assign(
    TMAX=lambda x: x.TMAX.fillna(x.TMAX.median()),
    TMIN=lambda x: x.TMIN.fillna(x.TMIN.median()),
    TOBS=lambda x: x.TOBS.fillna((x.TMAX + x.TMIN)/2)
).head()

df_deduped.apply(
    lambda x: x.fillna(x.rolling(7, min_periods=0).median())
).head(10)

df_deduped\
    .reindex(pd.date_range('2018-01-01','2018-12-31',freq='D'))\
    .apply(lambda x: x.interpolate())\
    .head(10)

# exercises

#1. combine the csvs to one faang dataset
import pandas as pd

faang = pd.DataFrame()
for ticker in ['fb', 'aapl', 'amzn', 'nflx', 'goog']:
    df = pd.read_csv(f'C:/Users/charlotte.henstock/PycharmProjects/pythonProject2/ch_03/exercises/{ticker}.csv')
    # make the ticker the first column
    df.insert(0, 'ticker', ticker.upper())
    faang=pd.concat([faang,df], axis=0)

faang.to_csv('faang.csv', index=False)

# use type conversion to cast the values of the date column into dataframes and the
# volume volume into integers.  then sort by date and ticker

faang = faang.assign(
    date=lambda x: pd.to_datetime(x.date),
    volume=lambda x:x.volume.astype(int)
).sort_values(
    ['date','ticker']
)

faang.head()
# seven rows in faang with the lowest values for volume

faang.nsmallest(7,'volume')

# melt the data to make it completely long

melted_faang = faang.melt(
    id_vars=['ticker','date'],
    value_vars=['open','high','low','close','volume']
)
melted_faang.head()

faang.set_index('date', inplace=True)
faang['2018-07-26':'2018-07-30']
faang.dtypes

# clean and pivot the data so it is in a wide format

covid = pd.read_csv('C:/Users/charlotte.henstock/PycharmProjects/pythonProject2/ch_03/exercises/covid19_cases.csv').assign(
    date=lambda x: pd.to_datetime(x.dateRep, format='%d/%m/%Y')
).set_index('date').replace(
    'United_States_of_America', 'USA'
).replace('United_Kingdom', 'UK').sort_index()

covid[
    covid.countriesAndTerritories.isin([
        'Argentina', 'Brazil', 'China', 'Colombia', 'India', 'Italy',
        'Mexico', 'Peru', 'Russia', 'Spain', 'Turkey', 'UK', 'USA'
    ])
].reset_index().pivot(index='date', columns='countriesAndTerritories', values='cases').fillna(0)

# find the 20 countries with the largest covid cases

pd.read_csv('C:/Users/charlotte.henstock/PycharmProjects/pythonProject2/ch_03/exercises/covid19_total_cases.csv', index_col='index')\
    .T.nlargest(20, 'cases').sort_values('cases', ascending=False)

