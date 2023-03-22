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