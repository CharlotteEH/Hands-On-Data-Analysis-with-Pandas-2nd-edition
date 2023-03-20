### 2-creating_dataframes.ipynb

import datetime as dt
import numpy as np
import pandas as pd

np.random.seed(0)  # set a seed for reproducibility
pd.Series(np.random.rand(5), name='random')

print(pd.Series(np.linspace(0, 10, num=5)).to_frame())

np.random.seed(0)
print(pd.DataFrame(
    {
        'random': np.random.rand(5),
        'text': ['hot', 'warm', 'cool', 'cold', None],
        'truth': [np.random.choice([True, False]) for _ in range(5)]
    },
    index=pd.date_range(
        end=dt.date(2019, 4, 21),
        freq='1D',
        periods=5,
        name='date'
    )
))

print(pd.DataFrame([
    {'mag': 5.2, 'place': 'California'},
    {'mag': 1.2, 'place': 'Alaska'},
    {'mag': 0.2, 'place': 'California'},
]))

list_of_tuples = [(n, n ** 2, n ** 3) for n in range(5)]
print(list_of_tuples)

print(pd.DataFrame(
    list_of_tuples,
    columns=['n', 'n_squared', 'n_cubed']
))

print(pd.DataFrame(
    np.array([
        [1, 1, 1],
        [2, 4, 8],
        [3, 9, 27],
        [4, 16, 64]
    ]), columns=['n', 'n_squared', 'n_cubed']
))

import os

with open('C:/Users/charlotte.henstock/PycharmProjects/pythonProject2/ch_02/data/earthquakes.csv', 'rb') as file:
    file.seek(0, os.SEEK_END)
    while file.read(1) != b'\n':
        file.seek(-2, os.SEEK_CUR)
    print(file.readline().decode())

with open('C:/Users/charlotte.henstock/PycharmProjects/pythonProject2/ch_02/data/earthquakes.csv', 'r') as file:
    print(len(file.readline().split(',')))

df = pd.read_csv("C:/Users/charlotte.henstock/PycharmProjects/pythonProject2/ch_02/data/earthquakes.csv")

df = pd.read_csv(
    'https://github.com/stefmolin/'
    'Hands-On-Data-Analysis-with-Pandas-2nd-edition'
    '/blob/master/ch_02/data/earthquakes.csv?raw=True'
)

df.to_csv('output.csv', index=False)

import sqlite3

with sqlite3.connect('C:/Users/charlotte.henstock/PycharmProjects/pythonProject2/ch_02/data/quakes.db') as connection:
    pd.read_csv('C:/Users/charlotte.henstock/PycharmProjects/pythonProject2/ch_02/data/tsunamis.csv').to_sql(
        'tsunamis', connection, index=False, if_exists='replace'
    )

with sqlite3.connect('C:/Users/charlotte.henstock/PycharmProjects/pythonProject2/ch_02/data/quakes.db') as connection:
    tsunamis = pd.read_sql('SELECT * FROM tsunamis', connection)
    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)
    pd.set_option("display.max_colwidth", None)
print(tsunamis.head())

## 3-making_dataframes_from_api_requests.ipynb

import datetime as dt
import pandas as pd
import requests

yesterday = dt.date.today() - dt.timedelta(days=1)
api = 'https://earthquake.usgs.gov/fdsnws/event/1/query'
payload = {
    'format': 'geojson',
    'starttime': yesterday - dt.timedelta(days=30),
    'endtime': yesterday
}
response = requests.get(api, params=payload)
# check
response.status_code

earthquake_json = response.json()
earthquake_json.keys()

earthquake_json['metadata']

type(earthquake_json['features'])
earthquake_json['features'][0]

earthquake_properties_data = [
    quake['properties'] for quake in earthquake_json['features']
]
df = pd.DataFrame(earthquake_properties_data)
df.head()

## 4-inspecting_dataframes.ipynb

import numpy as np
import pandas as pd

df = pd.read_csv('C:/Users/charlotte.henstock/PycharmProjects/pythonProject2/ch_02/data/earthquakes.csv')
df.empty
df.shape
df.columns
df.head()
df.tail()

pd.set_option('display.max_columns', None)

df.dtypes
df.info()
df.describe()

df.describe(percentiles=[0.05, 0.95])
df.describe(include='all')
df.felt.describe()

df.alert.unique()
df.alert.value_counts()

## 5-subsetting_data_ipynb
df.mag
df['mag']
df.get('event', False)
df[['mag', 'title']]

df[
    ['title', 'time']
    + [col for col in df.columns if col.startswith('mag')]
    ]

[col for col in df.columns if col.startswith('mag')]
['title', 'time'] \
+ [col for col in df.columns if col.startswith('mag')]

df[
    ['title', 'time']
    + [col for col in df.columns if col.startswith('mag')]
    ]

df[100:103]
df[['title', 'time']][100:103]

df[100:103][['title', 'time']].equals(
    df[['title', 'time']][100:103]
)

df[110:113]['title'] = df[110:113]['title'].str.lower()
df[110:113]['title']

df.loc[110:112, 'title'] = df.loc[110:112, 'title'].str.lower()
df.loc[110:112, 'title']

df.loc[:, 'title']
df.loc[10:15, ['title', 'mag']]

df.iloc[10:15, [19, 8]]

df.iloc[10:15, 6:10]
df.iloc[10:15, 6:10].equals(
    df.loc[10:14, 'gap':'magType']
)

df.at[10, 'mag']
df.iat[10, 8]
df.mag > 2
df[df.mag >= 7.0]
df.loc[df.mag >= 7.0,
['alert', 'mag', 'magType', 'title', 'tsunami', 'type']
]

df.loc[
    (df.tsunami == 1) & (df.alert == 'red'),
    ['alert', 'mag', 'magType', 'title', 'tsunami', 'type']
]

df.loc[
    (df.tsunami == 1) | (df.alert == 'red'),
    ['alert', 'mag', 'magType', 'title', 'tsunami', 'type']
]

df.loc[
    (df.place.str.contains('Alaska')) & (df.alert.notnull()),
    ['alert', 'mag', 'magType', 'title', 'tsunami', 'type']]

df.loc[
    df.mag.between(6.5, 7.5),
    ['alert', 'mag', 'magType', 'title', 'tsunami', 'type']
]

df.loc[
    (df.place.str.contains(r'CA|California$')) & (df.mag > 3.8),
    ['alert', 'mag', 'magType', 'title', 'tsunami', 'type']
]

df.loc[
    df.magType.isin(['mw', 'mwb']),
    ['alert', 'mag', 'magType', 'title', 'tsunami', 'type']
]

[df.mag.idxmin(), df.mag.idxmax()]
df.loc[
    [df.mag.idxmin(), df.mag.idxmax()],
    ['alert', 'mag', 'magType', 'title', 'tsunami', 'type']
]
df_to_modify = df.copy()

df.filter(items=['mag', 'magType']).head()
df.filter(like='mag').head()
df.filter(regex=r'^t').head()
df.set_index('place').filter(like='Japan', axis=0).filter(items=['mag', 'magType', 'title']).head()
df.set_index('place').title.filter(like='Japan').head()

##6-adding_and_removing_data.ipynb
df['source'] = 'USGS API'
df.head()

df['mag_negative'] = df.mag < 0
df.head()

df.place.str.extract(r',(.*$)')[0].sort_values().unique()

df['parsed_place'] = df.place.str.replace(
    r'.* of', '', regex=True
).str.replace(
    'the ', ''
).str.replace(
    r'CA$', 'California', regex=True
).str.replace(
    r'NV$', 'Nevada', regex=True
).str.replace(
    r'MX', 'Mexico', regex=True
).str.replace(
    r' region$', '', regex=True
).str.replace(
    'northern ', ''
).str.replace(
    'Fiji Islands', 'Fiji'
).str.replace(
    r'^.*, ', '', regex=True
).str.strip()

df.parsed_place.sort_values().unique()

df.assign(
    in_ca=df.parsed_place.str.endswith('California'),
    in_alaska=df.parsed_place.str.endswith('Alaska')
).sample(5, random_state=0)

df.assign(
    in_ca=df.parsed_place == 'California',
    in_alaska=df.parsed_place == 'Alaska',
    neither=lambda x: ~x.in_ca & ~x.in_alaska
).sample(5, random_state=0)

tsunami = df[df.tsunami == 1]
no_tsunami = df[df.tsunami == 0]
tsunami.shape, no_tsunami.shape

tsunami.append(no_tsunami).shape
additional_columns = pd.read_csv(
    'C:/Users/charlotte.henstock/PycharmProjects/pythonProject2/ch_02/data/earthquakes.csv',
    usecols=['tz', 'felt', 'ids']
)
pd.concat([df.head(2), additional_columns.head(2)], axis=1)

additional_columns = pd.read_csv(
    'C:/Users/charlotte.henstock/PycharmProjects/pythonProject2/ch_02/data/earthquakes.csv',
    usecols=['tz', 'felt', 'ids', 'time'], index_col='time')
pd.concat([df.head(2), additional_columns.head(2)], axis=1)

pd.concat(
    [tsunami.head(2), no_tsunami.head(2).assign(type='earthquake')], join='inner'
)

pd.concat(
    [tsunami.head(2), no_tsunami.head(2).assign(type='earthquake')], join='inner', ignore_index=True
)

del df['source']
df.columns

try:
    del df['source']
except KeyError:
    print('not there any more')

mag_negative = df.pop('mag_negative')
df.columns

mag_negative.value_counts()
df[mag_negative].head

df.drop([0, 1]).head(2)

cols_to_drop = [
    col for col in df.columns
    if col not in ['alert', 'mag', 'title', 'time', 'tsunami']
]
df.drop(columns=cols_to_drop).head()

df.drop(columns=cols_to_drop).equals(
    df.drop(cols_to_drop, axis=1)
)

df.drop(columns=cols_to_drop, inplace=True)
df.head()

# Exercises

# 95th percentile of earthquake magnitude mb in japan
df = pd.read_csv('C:/Users/charlotte.henstock/PycharmProjects/pythonProject2/ch_02/data/parsed.csv')
df[
    (df.parsed_place=='Japan')&(df.magType=='mb')
].mag.quantile(0.95)

# percentage of earthquakes in indonesia coupled with tsunamis
f"{df[df.parsed_place=='Indonesia'].tsunami.value_counts(normalize=True).iloc[1,]:.2%}"

# summary statistics for earthquakes in Nevada
df[df.parsed_place=='Nevada'].describe()

# add a column for whether it happened on the ring of fire
df['ring_of_fire']=df.parsed_place.str.contains(r'|'.join([
    'Alaska','Antarctic', 'Bolivia', 'California', 'Canada',
    'Chile', 'Costa Rica', 'Ecuador', 'Fiji', 'Guatemala',
    'Indonesia', 'Japan', 'Kermadec Islands', '^Mexico',
    'New Zealand', 'Peru', 'Philippines', 'Russia',
    'Taiwan', 'Tonga', 'Washington'
]))

df.columns
df[['parsed_place','ring_of_fire']]

# number of earthquakes in the ring of fire and outside

df.ring_of_fire.value_counts()


#tsunami count along the ring of fire
df.loc[df.ring_of_fire,'tsunami'].sum()