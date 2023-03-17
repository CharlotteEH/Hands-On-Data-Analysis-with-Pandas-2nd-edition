import datetime as dt
import numpy as np
import pandas as pd

np.random.seed(0) # set a seed for reproducibility
pd.Series(np.random.rand(5), name='random')

print(pd.Series(np.linspace(0,10,num=5)).to_frame())

np.random.seed(0)
print(pd.DataFrame(
    {
        'random': np.random.rand(5),
        'text':['hot','warm','cool','cold',None],
        'truth':[np.random.choice([True,False]) for _ in range(5)]
    },
    index=pd.date_range(
        end=dt.date(2019,4,21),
        freq='1D',
        periods=5,
        name='date'
    )
))


print(pd.DataFrame([
    {'mag':5.2,'place':'California'},
    {'mag':1.2,'place':'Alaska'},
    {'mag':0.2,'place':'California'},
]))

list_of_tuples=[(n, n**2, n**3) for n in range (5)]
print(list_of_tuples)

print(pd.DataFrame(
    list_of_tuples,
    columns=['n','n_squared', 'n_cubed']
))

print(pd.DataFrame(
    np.array([
        [1,1,1],
        [2,4,8],
        [3,9,27],
        [4,16,64]
    ]),columns=['n','n_squared','n_cubed']
))



import os
with open('C:/Users/charlotte.henstock/PycharmProjects/pythonProject2/ch_02/data/earthquakes.csv', 'rb') as file:
    file.seek(0, os.SEEK_END)
    while file.read(1) !=b'\n':
        file.seek(-2, os.SEEK_CUR)
    print(file.readline().decode())

with open ('C:/Users/charlotte.henstock/PycharmProjects/pythonProject2/ch_02/data/earthquakes.csv', 'r' ) as file:
    print(len(file.readline().split(',')))

df = pd.read_csv("C:/Users/charlotte.henstock/PycharmProjects/pythonProject2/ch_02/data/earthquakes.csv")

df=pd.read_csv(
    'https://github.com/stefmolin/'
    'Hands-On-Data-Analysis-with-Pandas-2nd-edition'
    '/blob/master/ch_02/data/earthquakes.csv?raw=True'
)

df.to_csv('output.csv', index=False)

import sqlite3
with sqlite3.connect('C:/Users/charlotte.henstock/PycharmProjects/pythonProject2/ch_02/data/quakes.db') as connection:
    pd.read_csv('C:/Users/charlotte.henstock/PycharmProjects/pythonProject2/ch_02/data/tsunamis.csv').to_sql(
        'tsunamis',connection, index=False, if_exists='replace'
    )

with sqlite3.connect('C:/Users/charlotte.henstock/PycharmProjects/pythonProject2/ch_02/data/quakes.db') as connection:
    tsunamis=pd.read_sql('SELECT * FROM tsunamis', connection)
    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)
    pd.set_option("display.max_colwidth", None)
print(tsunamis.head())