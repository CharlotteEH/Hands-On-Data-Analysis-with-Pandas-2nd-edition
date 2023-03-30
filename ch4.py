import pandas as pd

weather = pd.read_csv("C:/Users/charlotte.henstock/PycharmProjects/pythonProject2/ch_04/data/nyc_weather_2018.csv")
weather.head()

snow_data = weather.query('datatype == "SNOW" and value>0 and station.str.contains("US1NY")')
snow_data.head()

import sqlite3

with sqlite3.connect('C:/Users/charlotte.henstock/PycharmProjects/pythonProject2/ch_04/data/weather.db') as connection:
    snow_data_from_db = pd.read_sql(
        'SELECT * FROM weather WHERE datatype == "SNOW" AND value>0 and station LIKE "%US1NY%"',
        connection
    )

snow_data.reset_index().drop(columns='index').equals(snow_data_from_db)

station_info = pd.read_csv('C:/Users/charlotte.henstock/PycharmProjects/pythonProject2/ch_04/data/weather_stations.csv')
station_info.head()

station_info.id.describe()
weather.station.describe()

station_info.shape[0],weather.shape[0]

def get_row_count(*dfs):
    return [df.shape[0] for df in dfs]
get_row_count(station_info, weather)

inner_join = weather.merge(station_info,
                           left_on='station',
                           right_on='id')
inner_join.sample(5, random_state=0)

weather.merge(station_info.rename(dict(id='station'),axis=1), on='station').sample(5, random_state=0)

left_join = station_info.merge(weather, left_on='id', right_on='station', how='left')
right_join = weather.merge(station_info, left_on='station', right_on='id', how='right')
right_join[right_join.datatype.isna()].head()

left_join.sort_index(axis=1).sort_values(['date', 'station'], ignore_index=True).equals(
    right_join.sort_index(axis=1).sort_values(['date', 'station'], ignore_index=True)
)

get_row_count(inner_join, left_join, right_join)

outer_join = weather.merge(
    station_info[station_info.id.str.contains('US1NY')],
    left_on='station', right_on='id', how='outer', indicator=True
)

pd.concat([
    outer_join.query(f'_merge == "{kind}"').sample(2, random_state=0)
    for kind in outer_join._merge.unique()
]).sort_index()

import sqlite3

with sqlite3.connect('C:/Users/charlotte.henstock/PycharmProjects/pythonProject2/ch_04/data/weather.db') as connection:
    inner_join_from_db = pd.read_sql(
        'SELECT * FROM weather JOIN stations ON weather.station == stations.id',
        connection
    )
inner_join_from_db.shape == inner_join.shape

dirty_data = pd.read_csv(
    'C:/Users/charlotte.henstock/PycharmProjects/pythonProject2/ch_04/data/dirty_data.csv', index_col='date'
).drop_duplicates().drop(columns='SNWD')
dirty_data.head()

valid_station = dirty_data.query('station != "?"').drop(columns=['WESF', 'station'])
station_with_wesf = dirty_data.query('station == "?"').drop(columns=['station','TOBS','TMIN','TMAX'])

valid_station.merge(
    station_with_wesf, how='left', left_index=True, right_index=True
).query('WESF>0').head()

valid_station.merge(
    station_with_wesf, how='left', left_index=True, right_index=True,
    suffixes=('', '_?')
).query('WESF > 0').head()

valid_station.join(station_with_wesf, how='left', rsuffix='_?').query('WESF > 0').head()

weather.set_index('station', inplace=True)
station_info.set_index('id', inplace=True)

weather.index.intersection(station_info.index)

weather.index.difference(station_info.index)
station_info.index.difference(weather.index)

ny_in_name = station_info[station_info.index.str.contains('US1NY')]

ny_in_name.index.difference(weather.index).shape[0]\
+ weather.index.difference(ny_in_name.index).shape[0]\
== weather.index.symmetric_difference(ny_in_name.index).shape[0]

weather.index.unique().union(station_info.index)

# 2-dataframe_operations.ipynb

import numpy as np
import pandas as pd

weather = pd.read_csv('C:/Users/charlotte.henstock/PycharmProjects/pythonProject2/ch_04/data/nyc_weather_2018.csv', parse_dates=['date'])
weather.head()

fb = pd.read_csv('C:/Users/charlotte.henstock/PycharmProjects/pythonProject2/ch_04/data/fb_2018.csv', index_col='date', parse_dates=True)
fb.head()

fb.assign(
    abs_z_score_volume=lambda x: \
    x.volume.sub(x.volume.mean()).div(x.volume.std()).abs()
).query('abs_z_score_volume > 3')

fb.assign(
    volume_pct_change=fb.volume.pct_change(),
    pct_change_rank=lambda x: \
    x.volume_pct_change.abs().rank(ascending=False)
).nsmallest(5, 'pct_change_rank')

fb['2018-01-11':'2018-01-12']
(fb>215).any()

(fb>215).all()

(fb.volume.value_counts()>1).sum()
(fb.volume.value_counts()>1).any()

volume_binned=pd.cut(fb.volume, bins=3, labels=['low','med', 'high'])
volume_binned.value_counts()

fb[volume_binned == 'high'].sort_values('volume', ascending=False)

fb['2018-07-25':'2018-07-26']

fb['2018-03-16':'2018-03-20']

volume_qbinned = pd.qcut(
    fb.volume, q=4, labels=['q1','q2','q3','q4']
)
volume_qbinned.value_counts()

# from visual_aids.misc_viz import low_med_high_bins_viz
#
# low_med_high_bins_viz(
#     fb,'volume', ylabel='volume traded',
#     title='Daily Volume Traded of Facebook Stock in 2018 (with bins)'
# )

central_park_weather = weather\
    .query('station == "GHCND:USW00094728"')\
    .pivot(index='date', columns='datatype', values='value')

oct_weather_z_scores = central_park_weather\
    .loc['2018-10', ['TMIN', 'TMAX', 'PRCP']]\
    .apply(lambda x: x.sub(x.mean()).div(x.std()))
oct_weather_z_scores.describe().T

oct_weather_z_scores.query('PRCP > 3').PRCP

central_park_weather.loc['2018-10', 'PRCP'].describe()

fb.apply(
    lambda x: np.vectorize(lambda y: len(str(np.ceil(y))))(x)
).astype('int64').equals(
    fb.applymap(lambda x: len(str(np.ceil(x))))
)

import time

import numpy as np
import pandas as pd

np.random.seed(0)

vectorized_results = {}
iteritems_results = {}

for size in [10, 100, 1000, 10000, 100000, 500000, 1000000, 5000000, 10000000]:
    # set of numbers to use
    test = pd.Series(np.random.uniform(size=size))

    # time the vectorized operation
    start = time.time()
    x = test + 10
    end = time.time()
    vectorized_results[size] = end - start

    # time the operation with `iteritems()`
    start = time.time()
    x = []
    for i, v in test.iteritems():
        x.append(v + 10)
    x = pd.Series(x)
    end = time.time()
    iteritems_results[size] = end - start

results = pd.DataFrame(
    [pd.Series(vectorized_results, name='vectorized'), pd.Series(iteritems_results, name='iteritems')]
).T

# plotting
ax = results.plot(title='Time Complexity', color=['blue', 'red'], legend=False)

# formatting
ax.set(xlabel='item size (rows)', ylabel='time (s)')
ax.text(0.5e7, iteritems_results[0.5e7] * .9, 'iteritems()', rotation=34, color='red', fontsize=12, ha='center',
        va='bottom')
ax.text(0.5e7, vectorized_results[0.5e7], 'vectorized', color='blue', fontsize=12, ha='center', va='bottom')
for spine in ['top', 'right']:
    ax.spines[spine].set_visible(False)

central_park_weather.loc['2018-10'].assign(
    rolling_PRCP=lambda x: x.PRCP.rolling('3D').sum()
)[['PRCP', 'rolling_PRCP']].head(7).T

central_park_weather.loc['2018-10'].rolling('3D').mean().head(7).iloc[:,:6]

central_park_weather['2018-10-01':'2018-10-07'].rolling('3D').agg(
    {'TMAX':'max', 'TMIN':'min', 'AWND':'mean', 'PRCP':'sum'}
).join(
    central_park_weather[['TMAX','TMIN','AWND','PRCP']],
    lsuffix='_rolling'
).sort_index(axis=1)

central_park_weather.loc['2018-06'].assign(
    TOTAL_PRCP=lambda x: x.PRCP.cumsum(),
    AVG_PRCP=lambda x: x.PRCP.expanding().mean()
).head(10)[['PRCP','TOTAL_PRCP','AVG_PRCP']].T

central_park_weather\
    ['2018-10-01':'2018-10-07'].expanding().agg({
    'TMAX': np.max, 'TMIN':np.min,
    'AWND': np.mean, 'PRCP':np.sum
}).join(
    central_park_weather[['TMAX', 'TMIN', 'AWND', 'PRCP']],
    lsuffix='_expanding'
).sort_index(axis=1)

central_park_weather.assign(
AVG=lambda x: x.TMAX.rolling('30D').mean(),
EWMA=lambda x: x.TMAX.ewm(span=30).mean()
).loc['2018-09-29':'2018-10-08', ['TMAX', 'EWMA', 'AVG']].T

# def get_info(df):
#     return '%d rows, %d cols and max closing z-score: %d'
#     %(*df.shape, df.close.max())
#
# get_info(fb.loc['2018-Q1']\
#          .apply(lambda x: (x-x.mean())/x.stf())
#
# fb.loc['2018-Q1'].apply(lambda x: (x-x.mean())/x.std())\
#     .pipe(get_info()
#
# fb.pipe(pd.DataFrame.rolling, '20D').mean().equals(
#     fb.rolling('20D').mean()
# )



from window_calc import window_calc
#window_calc??

window_calc(fb, pd.DataFrame.expanding, np.median).head()

window_calc(fb, pd.DataFrame.ewm, 'mean', span=3).head()

window_calc(
    central_park_weather.loc['2018-10'],
    pd.DataFrame.rolling,
    {'TMAX':'max', 'TMIN':'min',
     'AWND':'mean','PRCP':'sum'},
    '3D'
).head()


fb_reindexed = fb\
    .reindex(pd.date_range('2018-01-01', '2018-12-31', freq='D'))\
    .assign(
    volume=lambda x: x.volume.fillna(0),
    close=lambda x: x.close.fillna(method='ffill'),
    open=lambda x: x.open.combine_first(x.close),
    high=lambda x: x.high.combine_first(x.close),
    low=lambda x: x.low.combine_first(x.close)
)
fb_reindexed.assign(day=lambda x: x.index.day_name()).head(10)

from pandas.api.indexers import VariableOffsetWindowIndexer

indexer = VariableOffsetWindowIndexer(
        index=fb_reindexed.index, offset=pd.offsets.BDay(3)
)
fb_reindexed.assign(window_start_day=0).rolling(indexer).agg({
    'window_start_day': lambda x: x.index.min().timestamp(),
    'open': 'mean','high': 'max', 'low':'min',
    'close':'mean', 'volume':'sum'
}).join(
    fb_reindexed, lsuffix='_rolling'
).sort_index(axis=1).assign(
    day=lambda x: x.index.day_name(),
window_start_day=lambda x: pd.to_datetime(x.window_start_day, unit='s')
).head(10)

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.max_colwidth", None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.width', None)
central_park_weather.loc['2018-06'].assign(
    TOTAL_PRCP=lambda x: x.PRCP.cumsum(),
    AVG_PRCP=lambda x: x.PRCP.expanding().mean()
).head(10)[['PRCP', 'TOTAL_PRCP', 'AVG_PRCP']].T

central_park_weather['2018-10-01':'2018-10-07'].expanding().agg(
    {'TMAX':np.max, 'TMIN': np.min, 'AWND':np.mean, 'PRCP':np.sum}
).join(
    central_park_weather[['TMAX', 'TMIN', 'AWND', 'PRCP']],
    lsuffix='_expanding'
).sort_index(axis=1)

central_park_weather.assign(
    AVG=lambda x: x.TMAX.rolling('30D').mean(),
    EWMA=lambda x: x.TMAX.ewm(span=30).mean()
).loc['2018-09-29':'2018-10-08',['TMAX','EWMA','AVG']].T

fb.pipe(pd.DataFrame.rolling, '20D').mean().equals(fb.rolling('20D').mean())

# 3-aggregations.ipynb

import numpy as np
import pandas as pd

fb = pd.read_csv('C:/Users/charlotte.henstock/PycharmProjects/pythonProject2/ch_04/data/fb_2018.csv', index_col='date', parse_dates=True).assign(
    trading_volume=lambda x: pd.cut(x.volume, bins=3, labels=['low', 'med', 'high'])
)
fb.head()

weather = pd.read_csv('C:/Users/charlotte.henstock/PycharmProjects/pythonProject2/ch_04/data/weather_by_station.csv', index_col='date', parse_dates=True)
weather.head()

pd.set_option('display.float_format', lambda x: '%.2f' %x)

fb.agg({
    'open': np.mean,
    'high': np.max,
    'low': np.min,
    'close': np.mean,
    'volume':np.sum
})

weather.query('station == "GHCND:USW00094728"')\
    .pivot(columns='datatype', values='value')[['SNOW', 'PRCP']]\
    .sum()

fb.agg({
    'open': 'mean',
    'high': ['min', 'max'],
    'low': ['min', 'max'],
    'close': 'mean'
})

fb.groupby('trading_volume').mean()

fb.groupby('trading_volume')['close'].agg(['min', 'max', 'mean'])

fb_agg = fb.groupby('trading_volume').agg({
    'open': 'mean',
    'high': ['min', 'max'],
    'low': ['min', 'max'],
    'close': 'mean'
})
fb_agg

fb_agg.columns

fb_agg.columns = ['_'.join(col_agg) for col_agg in fb_agg.columns]
fb_agg.head()

weather.loc['2018-10'].query('datatype == "PRCP"')\
    .groupby(level=0).mean().head().squeeze()

weather.query('datatype == "PRCP"').groupby(
    ['station_name', pd.Grouper(freq='Q')]
).sum().unstack().sample(5, random_state=1)

weather.groupby('station_name').filter(
    lambda x: x.name.endswith('NY US')
).query('datatype=="SNOW"').groupby('station_name').sum().squeeze()

weather.query('datatype=="PRCP"')\
    .groupby(level=0).mean()\
    .groupby(pd.Grouper(freq='M')).sum().value.nlargest()

weather.query('datatype == "PRCP"')\
    .rename(dict(value='prcp'), axis=1)\
    .groupby(level=0).mean()\
    .groupby(pd.Grouper(freq='M'))\
    .transform(np.sum)['2018-01-28':'2018-02-03']

weather\
    .query('datatype=="PRCP"')\
    .rename(dict(value='prcp'), axis=1)\
    .groupby(level=0).mean()\
    .assign(
        total_prcp_in_month=lambda x:\
    x.groupby(pd.Grouper(freq='M')).transform(np.sum),
    pct_monthly_prcp=lambda x:\
    x.prcp.div(x.total_prcp_in_month)
)\
    .nlargest(5, 'pct_monthly_prcp')

fb[['open','high','low','close']]\
    .transform(lambda x: (x - x.mean()).div(x.std()))\
    .head()

fb.pivot_table(columns='trading_volume')
fb.pivot_table(index='trading_volume')

weather.reset_index().pivot_table(
    index=['date','station','station_name'],
    columns='datatype',
    values='value',
    aggfunc='median'
).reset_index().tail()

pd.crosstab(
    index=fb.trading_volume,
    columns=fb.index.month,
    colnames=['month']
)

pd.crosstab(
    index=fb.trading_volume,
    columns=fb.index.month,
    colnames=['month'],
    normalize='columns'
)

pd.crosstab(
    index=fb.trading_volume,
    columns=fb.index.month,
    colnames=['month'],
    values=fb.close,
    aggfunc=np.mean
)

snow_data = weather.query('datatype == "SNOW"')
pd.crosstab(
    index=snow_data.station_name,
    columns=snow_data.index.month,
    colnames=['month'],
    values=snow_data.value,
    aggfunc=lambda x: (x>0).sum(),
    margins=True,
    margins_name='total observation of snow')

# 4-time_series.ipynb

import numpy as np
import pandas as pd

fb = pd.read_csv('C:/Users/charlotte.henstock/PycharmProjects/pythonProject2/ch_04/data/fb_2018.csv', index_col='date', parse_dates=True).assign(
    trading_volume=lambda x: pd.cut(x.volume, bins=3, labels=['low', 'med', 'high'])
)
fb.head()

fb['2018-10-11':'2018-10-15']

fb.loc['2018-q1'].equals(fb['2018-01':'2018-03'])
fb.first('1W')
fb.last('1W')

fb_reindexed = fb.reindex(pd.date_range('2018-01-01', '2018-12-31', freq='D'))
fb_reindexed.first('1D').isna().squeeze().all()

fb_reindexed.loc['2018-Q1'].first_valid_index()
fb_reindexed.loc['2018-Q1'].last_valid_index()

fb_reindexed.asof('2018-03-31')

stock_data_per_minute = pd.read_csv(
    'C:/Users/charlotte.henstock/PycharmProjects/pythonProject2/ch_04/data/fb_week_of_may_20_per_minute.csv', index_col='date', parse_dates=True,
    date_parser=lambda x: pd.to_datetime(x, format='%Y-%m-%d %H-%M')
)
stock_data_per_minute.head()

stock_data_per_minute.groupby(pd.Grouper(freq='1D')).agg({
    'open': 'first',
    'high': 'max',
    'low': 'min',
    'close': 'last',
    'volume': 'sum'
})

stock_data_per_minute.at_time('9:30')

stock_data_per_minute.between_time('15:59', '16:00')

shares_traded_in_first_30_min = stock_data_per_minute\
    .between_time('9:30', '10:00')\
    .groupby(pd.Grouper(freq='1D'))\
    .filter(lambda x: (x.volume>0).all())\
    .volume.mean()

shares_traded_in_last_30_min = stock_data_per_minute\
    .between_time('15:30', '16:00')\
    .groupby(pd.Grouper(freq='1D'))\
    .filter(lambda x: (x.volume > 0).all())\
    .volume.mean()

shares_traded_in_first_30_min - shares_traded_in_last_30_min

pd.DataFrame(
    dict(before=stock_data_per_minute.index, after=stock_data_per_minute.index.normalize())
).head()

stock_data_per_minute.index.to_series().dt.normalize().head()

fb.assign(
    prior_close=lambda x: x.close.shift(),
    after_hours_change_in_price=lambda x: x.open - x.prior_close,
    abs_change=lambda x: x.after_hours_change_in_price.abs()
).nlargest(5, 'abs_change')

pd.date_range('2018-01-01', freq='D', periods=5) + pd.Timedelta('9 hours 30 minutes')

(
    fb.drop(columns='trading_volume')
    - fb.drop(columns='trading_volume').shift()
).equals(
    fb.drop(columns='trading_volume').diff()
)

fb.drop(columns='trading_volume').diff().head()

fb.drop(columns='trading_volume').diff(-3).head()

from visual_aids.misc_viz import resampling_example
resampling_example()

array([<AxesSubplot:title={'center':'raw data'}, xlabel='date', ylabel='events'>,
       <AxesSubplot:title={'center':'daily totals'}, xlabel='date', ylabel='events'>],
      dtype=object)

stock_data_per_minute.head()

stock_data_per_minute.resample('1D').agg({
    'open': 'first',
    'high': 'max',
    'low': 'min',
    'close': 'last',
    'volume': 'sum'
})

fb.resample('Q').mean()

fb.drop(columns='trading_volume').resample('Q').apply(
    lambda x: x.last('1D').values - x.first('1D').values
)

melted_stock_data = pd.read_csv('C:/Users/charlotte.henstock/PycharmProjects/pythonProject2/ch_04/data/melted_stock_data.csv', index_col='date', parse_dates=True)
melted_stock_data.head()

melted_stock_data.resample('1D').ohlc()['price']

fb.resample('6H').asfreq().head()
fb.resample('6H').pad().head()

fb.resample('6H').fillna('nearest').head()

fb.resample('6H').asfreq().assign(
    volume=lambda x: x.volume.fillna(0),
    close=lambda x: x.close.fillna(method='ffill'),
    open=lambda x: x.open.combine_first(x.close),
    high=lambda x: x.high.combine_first(x.close),
    low=lambda x: x.low.combine_first(x.close)
).head()

import sqlite3

with sqlite3.connect('C:/Users/charlotte.henstock/PycharmProjects/pythonProject2/ch_04/data/stocks.db') as connection:
    fb_prices = pd.read_sql(
        'SELECT * FROM fb_prices', connection,
        index_col='date', parse_dates=['date']
    )
aapl_prices = pd.read_sql(
    'SELECT * FROM aapl_prices', connection,
    index_col='date', parse_dates=['date']
)

fb_prices.index.second.unique()
aapl_prices.index.second.unique()

pd.merge_asof(
    fb_prices, aapl_prices,
    left_index=True, right_index=True,
    direction='nearest', tolerance=pd.Timedelta(30, unit='s')
).head()

pd.merge_ordered(
    fb_prices.reset_index(), aapl_prices.reset_index()
).set_index('date').head()

pd.merge_ordered(
    fb_prices.reset_index(), aapl_prices.reset_index(),
    fill_method='ffill'
).set_index('date').head()

pd.merge_ordered(
    fb_prices.reset_index(), aapl_prices.reset_index(),
    fill_method='fillna()'
).set_index('date').head()

# exercises


quakes = pd.read_csv('C:/Users/charlotte.henstock/PycharmProjects/pythonProject2/ch_04/exercises/earthquakes.csv')
faang = pd.read_csv('C:/Users/charlotte.henstock/PycharmProjects/pythonProject2/ch_04/exercises/faang.csv', index_col='date', parse_dates=True)

# 1. Earthquakes in Japan with magnitude mb equal and over 4.9

quakes.query(
    "parsed_place == 'Japan' and magType == 'mb' and mag>=4.9"
)[['mag', 'magType', 'place']]