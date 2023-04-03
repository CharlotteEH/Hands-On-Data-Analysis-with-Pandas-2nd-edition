import matplotlib.pyplot as plt
import pandas as pd

fb = pd.read_csv(
    'C:/Users/charlotte.henstock/PycharmProjects/pythonProject2/ch_05/data/fb_stock_prices_2018.csv', index_col='date', parse_dates=True
)

plt.plot(fb.index, fb.open)
plt.show()

plt.plot('high', 'low', 'or', data=fb.head(20))
plt.show()

quakes = pd.read_csv('C:/Users/charlotte.henstock/PycharmProjects/pythonProject2/ch_05/data/earthquakes.csv')
plt.hist(quakes.query('magType == "ml"').mag)
plt.show()

x = quakes.query('magType == "ml"').mag
fig, axes = plt.subplots(1, 2, figsize=(10, 3))
for ax, bins in zip(axes, [7, 35]):
    ax.hist(x, bins=bins)
    ax.set_title(f'bins param: {bins}')

fig = plt.figure()
fig, axes = plt.subplots(1, 2)

fig = plt.figure(figsize=(3, 3))
outside = fig.add_axes([0.1, 0.1, 0.9, 0.9])
inside = fig.add_axes([0.7, 0.7, 0.25, 0.25])

fig = plt.figure(figsize=(8, 8))
gs = fig.add_gridspec(3, 3)
top_left = fig.add_subplot(gs[0, 0])
mid_left = fig.add_subplot(gs[1, 0])
top_right = fig.add_subplot(gs[:2, 1:])
bottom = fig.add_subplot(gs[2, :])

fig.savefig('empty.png')

plt.close('all')

fig = plt.figure(figsize=(10, 4))
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

import random
import matplotlib as mpl

rcparams_list = list(mpl.rcParams.keys())
random.seed(20) # make this repeatable
random.shuffle(rcparams_list)
sorted(rcparams_list[:20])

mpl.rcParams['figure.figsize']

mpl.rcParams['figure.figsize'] = (300, 10)
mpl.rcParams['figure.figsize']

mpl.rcdefaults()
mpl.rcParams['figure.figsize']

plt.rc('figure', figsize=(20, 20))
plt.rcdefaults()

# 2-plotting_with_pandas.ipynb

fb = pd.read_csv(
    'C:/Users/charlotte.henstock/PycharmProjects/pythonProject2/ch_05/data/fb_stock_prices_2018.csv', index_col='date', parse_dates=True
)
quakes = pd.read_csv('C:/Users/charlotte.henstock/PycharmProjects/pythonProject2/ch_05/data/earthquakes.csv')
covid = quakes = pd.read_csv('C:/Users/charlotte.henstock/PycharmProjects/pythonProject2/ch_05/data/covid19_cases.csv').assign(
    date=lambda x: pd.to_datetime(x.dateRep, format='%d/%m/%Y')
).set_index('date').replace(
    'United_States_of_America', 'USA'
).sort_index()['2020-01-18':'2020-09-18']

fb.plot(
    kind='line',
    y='open',
    figsize=(10, 5),
    style='-b',
    legend=False,
    title='Evolution of Facebook Open Price'
)
plt.show()

fb.plot(
    kind='line',
    y='open',
    figsize=(10, 5),
    color='blue',
    linestyle='solid',
    legend=False,
    title='Evolution of Facebook Open Price'
)
plt.show()

fb.first('1W').plot(
    y=['open', 'high', 'low', 'close'],
    style=['o-b', '--r', ':k', '.-g'],
    title='Facebook OHLC Prices during the first week of Trading 2018'
).autoscale()
plt.show()

fb.plot(
    kind='line',
    subplots=True,
    layout=(3, 2),
    figsize=(15, 10),
    title='Facebook Stock 2018'
)
plt.show()

import pandas as pd

covid = quakes = pd.read_csv('C:/Users/charlotte.henstock/PycharmProjects/pythonProject2/ch_05/data/covid19_cases.csv').assign(
    date=lambda x: pd.to_datetime(x.dateRep, format='%d/%m/%Y')
).set_index('date').replace(
    'United_States_of_America', 'USA'
).sort_index()['2020-01-18':'2020-09-18']

new_cases_rolling_average = covid.pivot_table(
    index=covid.index,
    columns='countriesAndTerritories',
    values='cases'
).rolling(7).mean()


import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

new_cases_rolling_average[['China']].plot(ax=axes[0], style='-.c')
new_cases_rolling_average[['Italy', 'Spain']].plot(
    ax=axes[1], style=['-', '--'],
    title='7-day rolling average of new COVID-19 cases\n(source: ECDC)'
)
new_cases_rolling_average[['Brazil', 'India', 'USA']]\
    .plot(ax=axes[2], style=['--', ':', '-'])

plot_cols = ['Brazil', 'India', 'Italy & Spain', 'USA', 'Other']
grouped = ['Italy', 'Spain']
other_cols = [
    col for col in new_cases_rolling_average.columns
    if col not in plot_cols
]

new_cases_rolling_average.sort_index(axis=1).assign(
    **{
        'Italy & Spain': lambda x: x[grouped].sum(axis=1),
        'Other': lambda x: x[other_cols].drop(columns=grouped).sum(axis=1)
    }
)[plot_cols].plot(
    kind='area', figsize=(15, 5),
    title='7-day rolling average of new COVID-19 cases\n(source: ECDC)'
)
plt.show()

fig, axes = plt.subplots(1, 3, figsize=(15, 3))
cumulative_covid_cases = covid.groupby(
    ['countriesAndTerritories', pd.Grouper(freq='1D')]
).cases.sum().unstack(0).apply('cumsum')

cumulative_covid_cases[['China']].plot(ax=axes[0], style='-.c')
cumulative_covid_cases[['Italy', 'Spain']].plot(
    ax=axes[1], style=['-', '--'],
    title='Cumulative covid cases\n(source: ECDC)'
)
cumulative_covid_cases[['Brazil', 'India', 'USA']]\
    .plot(ax=axes[2], style=['--',':','-'])
plt.show()

fb.assign(
    max_abs_change=fb.high - fb.low
).plot(
    kind='scatter', x='volume', y='max_abs_change',
    title='Facebook Daily High - Low vs. Volume Traded'
)
plt.show()

fb.assign(
    max_abs_change=fb.high - fb.low
).plot(
    kind='scatter', x='volume', y='max_abs_change',
    title='Facebook Daily High - Low vs. log(Volume Traded)',
    logx=True
)
plt.show()

fb.assign(
    max_abs_change=fb.high - fb.low
).plot(
    kind='scatter', x='volume', y='max_abs_change',
    title='Facebook daily high - low vs log(volume traded)',
    logx=True, alpha=0.25
)
plt.show()

import numpy as np
fb.assign(
    log_volume=np.log(fb.volume),
    max_abs_change=fb.high - fb.low
).plot(
    kind='hexbin',
    x='log_volume',
    y='max_abs_change',
    title='Facebook Daily High - Low vs log(volume traded)',
    colormap='gray_r',
    gridsize=20,
    sharex=False
)
plt.show()

fig, ax = plt.subplots(figsize=(20, 10))

# calculate the correlation matrix
fb_corr = fb.assign(
    log_volume=np.log(fb.volume),
    max_abs_change=fb.high - fb.low
).corr()

# create the heatmap and colorbar
im = ax.matshow(fb_corr, cmap='seismic')
im.set_clim(-1, 1)
fig.colorbar(im)

# label the ticks with the column names
labels = [col.lower() for col in fb_corr.columns]
ax.set_xticks(ax.get_xticks()[1:-1]) # to handle bug in matplotlib
ax.set_xticklabels(labels, rotation=45)
ax.set_yticks(ax.get_yticks()[1:-1]) # to handle bug in matplotlib
ax.set_yticklabels(labels)

# include the value of the correlation coefficient in the boxes
for (i, j), coef in np.ndenumerate(fb_corr):
    ax.text(
        i, j, fr'$\rho$ = {coef:.2f}', # raw (r), format (f) string
        ha='center', va='center',
        color='white', fontsize=14
    )
plt.show()

fb_corr.loc['max_abs_change', ['volume', 'log_volume']]

fb.volume.plot(
    kind='hist',
    title='Histogram of Daily Volume Traded in Facebook Stock'
)
plt.xlabel('Volume traded')
plt.show()

quakes = pd.read_csv('C:/Users/charlotte.henstock/PycharmProjects/pythonProject2/ch_05/data/earthquakes.csv')
fig, axes = plt.subplots(figsize=(8, 5))

for magType in quakes.magType.unique():
    data = quakes.query(f'magType == "{magType}"').mag
    if not data.empty:
        data.plot(
            kind='hist', ax=axes, alpha=0.4,
            label=magType, legend=True,
            title='Comparing histograms of earthquake magnitude by magType'
        )

plt.xlabel('magnitude')
plt.show()

fb.high.plot(
    kind='kde',
    title='KDE of Daily High Price for Facebook Stock'
)
plt.xlabel('Price($')
plt.show()

ax = fb.high.plot(kind='hist', density=True, alpha=0.5)
fb.high.plot(
    ax=ax, kind='kde', color='blue',
    title='distibution of facebook stock\'s daily high price in 2018'
)
plt.xlabel('Price($)')
plt.show()

from statsmodels.distributions.empirical_distribution import ECDF

ecdf = ECDF(quakes.query('magType=="ml"').mag)
plt.plot(ecdf.x, ecdf.y)

plt.xlabel('mag')
plt.ylabel('cumulative probability')

plt.title('ECDF of earthquake magnitude with magType ml')
plt.show()

from statsmodels.distributions.empirical_distribution import ECDF

ecdf = ECDF(quakes.query('magType == "ml"').mag)
plt.plot(ecdf.x, ecdf.y)

plt.xlabel('mag')
plt.ylabel('cumulative probability')

plt.plot(
    [3, 3], [0, 0.98], '--k',
    [-1.5, 3], [0.98, 0.98], '--k', alpha=0.4
)

plt.ylim(0, None)
plt.xlim(-1.25, None)
plt.title('P(mag<=3)=98%')
plt.show()

fb.iloc[:, :4].plot(kind='box', title='Facebook OHLC Prices Box Plot')
plt.ylabel('price ($')
plt.show()

fb.iloc[:, :4].plot(kind='box', title='Facebook OHLC Prices Box Plot', notch=True)
plt.ylabel('price ($)')
plt.show()

fb.assign(
    volume_bin=pd.cut(fb.volume, 3, labels=['low', 'med', 'high'])
).groupby('volume_bin').boxplot(
    column=['open', 'high', 'low', 'close'],
    layout=(1, 3), figsize=(12, 3)
)
plt.suptitle('Facebook OHLC Box Plots by Volume Trader', y=1.1)
plt.show()

quakes[['mag', 'magType']].groupby('magType').boxplot(
    figsize=(15, 8), subplots=False
)
plt.title('Earthquake Magnitude Box Plots by magType')
plt.ylabel('magnitude')
plt.show()

quakes.parsed_place.value_counts().iloc[14::-1,].plot(
    kind='barh', figsize=(10, 5),
    title='Top 15 Places for Earthquakes '
    '(September 18, 2018 - October 13, 2018)'
)
plt.xlabel('earthquakes')
plt.show()

quakes.groupby('parsed_place').tsunami.sum().sort_values().iloc[-10:,].plot(
    kind='barh', figsize=(10, 5),
    title='Top 10 Places for Tsunamis '
    '(September 18, 2018 - October 13, 2018)'
)
plt.xlabel('tsunamis')
plt.show()

indonesia_quakes = quakes.query('parsed_place == "Indonesia"').assign(
    time=lambda x: pd.to_datetime(x.time, unit='ms'),
    earthquake=1
).set_index('time').resample('1D').sum()

indonesia_quakes.index = indonesia_quakes.index.strftime('%b\n%d')

indonesia_quakes.plot(
    y=['earthquake', 'tsunami'], kind='bar', figsize=(15, 3),
    rot=0, label =['earthquakes', 'tsunamis'],
    title='Earthquakes and Tsunamis in Indonesia '
    '(September 18, 2018 - October 13, 2018)'
)

plt.xlabel('date')
plt.ylabel('count')
plt.show()

quakes.groupby(['parsed_place', 'tsunami']).mag.count()\
    .unstack().apply(lambda x: x/x.sum(), axis=1)\
    .rename(columns={0: 'no', 1: 'yes'})\
    .sort_values('yes', ascending=False)[7::-1]\
    .plot.barh(
        title='Frequency of a tsunami accompanying an earthquake'
)

plt.legend(title='tsunami?', bbox_to_anchor=(1, 0.65))

plt.xlabel('percentage of earthquakes')
plt.ylabel('')
plt.tight_layout
plt.show()

quakes.magType.value_counts().plot(
    kind='bar', title='Earthquakes Recorded per magType', rot=0
)

plt.xlabel('magType')
plt.ylabel('earthquakes')
plt.show()

pivot = quakes.assign(
    mag_bin=lambda x: np.floor(x.mag)
).pivot_table(
    index='mag_bin', columns='magType', values='mag', aggfunc='count'
)
pivot.plot.bar(
    stacked=True, rot=0, ylabel='earthquakes',
    title='Earthquakes by integer magnitude and magType'
)
plt.show()

normalized_pivot = pivot.fillna(0).apply(lambda x: x/x.sum(), axis=1)
ax=normalized_pivot.plot.bar(
    stacked=True, rot=0, figsize=(10, 5),
    title='Percentage of earthquakes by integer magnitude for each magType'
)
ax.legend(bbox_to_anchor=(1, 0.8))
plt.ylabel('percentage')
plt.show()

quakes.groupby(['parsed_place', 'tsunami']).mag.count()\
    .unstack().apply(lambda x: x/x.sum(), axis=1)\
    .rename(columns={0: 'no', 1: 'yes'})\
    .sort_values('yes', ascending=False)[7::-1]\
    .plot.barh(
        title='Frequency of a tsunami accomoanying an earthquake',
        stacked=True
)

plt.legend(title='tsunami?', bbox_to_anchor=(1, 0.65))

plt.xlabel('percentage of earthquakes')
plt.ylabel('')
plt.show()

# 3-pandas_plotting_module.ipynb

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

fb = pd.read_csv(
    'C:/Users/charlotte.henstock/PycharmProjects/pythonProject2/ch_05/data/fb_stock_prices_2018.csv', index_col='date', parse_dates=True
)

from pandas.plotting import scatter_matrix
scatter_matrix(fb, figsize=(10, 10))
plt.show()

scatter_matrix(fb, figsize=(10, 10), diagonal='kde')
plt.show()

from pandas.plotting import lag_plot
np.random.seed(0)
lag_plot(pd.Series(np.random.random(size=200)))
plt.show()

lag_plot(fb.close)
plt.show()

lag_plot(fb.close, lag=5)
plt.show()

from pandas.plotting import autocorrelation_plot
np.random.seed(0)
autocorrelation_plot(pd.Series(np.random.random(size=200)))
plt.show()

autocorrelation_plot(fb.close)
plt.show()

from pandas.plotting import bootstrap_plot
fig = bootstrap_plot(fb.volume, fig=plt.figure(figsize=(10, 6)))
plt.show()

# exercises.

# 20 rolling minimum of facebook close price


fb = pd.read_csv(
    'C:/Users/charlotte.henstock/PycharmProjects/pythonProject2/ch_05/data/fb_stock_prices_2018.csv', index_col='date', parse_dates=True
)
quakes = pd.read_csv(
'C:/Users/charlotte.henstock/PycharmProjects/pythonProject2/ch_05/data/earthquakes.csv'
)
covid = pd.read_csv(
'C:/Users/charlotte.henstock/PycharmProjects/pythonProject2/ch_05/data/covid19_cases.csv').assign(
    date=lambda x: pd.to_datetime(x.dateRep, format='%d/%m/%Y')
).set_index('date').replace(
    'United_States_of_America', 'USA'
).sort_index()['2020-01-18':'2020-09-18']

fb.close.rolling('20D').min().plot(
    title='Rolling 20D Minimum Closing Price of Facebook Stock'
)
plt.show()

# histogram and kde of the change from open to close in the price of facebook stock

differential = fb.open - fb.close
ax = differential.plot(kind='hist', density='True', alpha=0.3)
differential.plot(
    kind='kde', color='blue', ax=ax,
    title='Facenbook Stock price\'s daily change from open to close'
)


# box plots of each magtype for indonesia

quakes.query('parsed_place == "Indonesia"')[['mag', 'magType']]\
    .groupby('magType').boxplot(layout=(1, 4), figsize=(15, 3))
plt.show()

# line plot of the difference between teh weekly max high price and min low price

fb.resample('1W').agg(
    dict(high='max', low='min')
).assign(
    max_change_weekly=lambda x: x.high - x.low
).max_change_weekly.plot(
    title='Difference between Weekly maximum high price\n'
    'and weekly minimum low price of facebook stock'
)
plt.show()

# 14 day moving average of the daily change in new covid cases in
#Brazil, china, india, spain, usa

#diff() to calculate day on day changes
# rolling to calculate 14 day average

# three plots.  china, italy & spain, brazil, india, & usa

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

new_cases_rolling_average = covid.pivot_table(
    index=covid.index, columns=['countriesAndTerritories'], values='cases'
).apply(lambda x: x.diff().rolling(14).mean())

new_cases_rolling_average[['China']].plot(ax=axes[0], color='red')
new_cases_rolling_average[['Italy', 'Spain']].plot(
    ax=axes[1], color=['magenta', 'cyan'],
    title='14-day rolling average of change dailt covid cases\n(source: ECDC)'
)
new_cases_rolling_average[['Brazil', 'India', 'USA']].plot(ax=axes[2], color=['pink', 'purple', 'cyan'])
plt.show()

# use matplotlib and pandas create sublplots that show the effect after hours trading has on facebooks stock price

# the daily differece between that days opening and the previous days closing

# bar plot showing the effect this had monthly using resample

# color the bars green and red

# show the 3 letter for the month

series =(fb.open - fb.close.shift())
monthly_effect = series.resample('1M').sum()

fig, axes = plt.subplots(1, 2, figsize=(10, 3))

series.plot(
    ax=axes[0],
    title='After hours trading\n(Open Price - prior day\'s Close)'
)

monthly_effect.index = monthly_effect.index.strftime('%b')
monthly_effect.plot(
    ax=axes[1],
    kind='bar',
    title='After hours trading monthly effect',
    color=np.where(monthly_effect>=0, 'g', 'r'),
    rot=0
) 
plt.show()