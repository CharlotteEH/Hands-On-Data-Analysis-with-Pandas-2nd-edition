# 1-introduction_to_seaborn

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

fb = pd.read_csv(
    'C:/Users/charlotte.henstock/PycharmProjects/pythonProject2/ch_06/data/fb_stock_prices_2018.csv', index_col='date', parse_dates=True
)
quakes = pd.read_csv(
    'C:/Users/charlotte.henstock/PycharmProjects/pythonProject2/ch_06/data/earthquakes.csv'
)

quakes.assign(
    time=lambda x: pd.to_datetime(x.time, unit='ms')
).set_index('time').loc['2018-09-28'].query(
    'parsed_place == "Indonesia" and tsunami and mag == 7.5'
)

sns.stripplot(
    x='magType',
    y='mag',
    hue='tsunami',
    palette=["darkgoldenrod", "deepskyblue"],
    data=quakes.query('parsed_place == "Indonesia"')
)
plt.show()

sns.swarmplot(
    x='magType',
    y='mag',
    hue='tsunami',
    palette=["darkgoldenrod", "deepskyblue"],
    data=quakes.query('parsed_place == "Indonesia"'),
    size = 3.5
)
plt.show()

sns.boxenplot(
    x='magType', y='mag', data=quakes[['magType', 'mag']]
)
plt.title('comparing earthquake magnitude by magType')
plt.show()

fig, axes = plt.subplots(figsize = (10, 5))
sns.violinplot(
    x='magType', y='mag', data=quakes[['magType', 'mag']],
    ax=axes, scale='width'
)
plt.title('comparing earthquake magnitude by magtype')
plt.show()

sns.heatmap(
    fb.sort_index().assign(
        log_volume=np.log(fb.volume),
        max_abs_change=fb.high - fb.low
    ).corr(),
    annot=True, center=0, vmin=-1, vmax=1, cmap="PiYG"
)
plt.show()

sns.pairplot(fb)
plt.show()

sns.pairplot(
    fb.assign(quarter=lambda x: x.index.quarter),
    diag_kind='kde',
    hue='quarter'
)
plt.show()

sns.jointplot(
    x='log_volume',
    y='max_abs_change',
    data=fb.assign(
        log_volume=np.log(fb.volume),
        max_abs_change=fb.high - fb.low
    )
)
plt.show()

sns.jointplot(
    x='log_volume',
    y='max_abs_change',
    kind='hex',
    data=fb.assign(
        log_volume=np.log(fb.volume),
        max_abs_change=fb.high - fb.low
    )
)
plt.show()

sns.jointplot(
    x='log_volume',
    y='max_abs_change',
    kind='reg',
    data=fb.assign(
        log_volume=np.log(fb.volume),
        max_abs_change=fb.high - fb.low
    )
)
plt.show()

sns.jointplot(
    x='log_volume',
    y='max_abs_change',
    kind='kde',
    data=fb.assign(
        log_volume=np.log(fb.volume),
        max_abs_change=fb.high - fb.low
    )
)
plt.show()

sns.jointplot(
    x='log_volume',
    y='max_abs_change',
    kind='resid',
    data=fb.assign(
        log_volume=np.log(fb.volume),
        max_abs_change=fb.high - fb.low
    )
)
# update y-axis label (discussed in the next notebook)
plt.ylabel('residuals')
plt.show()

fb_reg_data = fb.assign(
    log_volume=np.log(fb.volume),
    max_abs_change=fb.high - fb.low
).iloc[:,-2:]

import itertools

iterator = itertools.repeat("I'm an iterator", 1)

for i in iterator:
    print(f'-->{i}')
print('This printed once because the iterator has been exhausted')
for i in iterator:
    print(f'-->{i}')

iterable = list(itertools.repeat("I'm an iterable", 1))

for i in iterable:
    print(f'-->{i}')
print('This prints again because it\'s an iterable:')
for i in iterable:
    print(f'-->{i}')

from viz import reg_resid_plots
reg_resid_plots??

reg_resid_plots(fb_reg_data)

sns.lmplot(
    x='log_volume',
    y='max_abs_change',
    data=fb.assign(
        log_volume=np.log(fb.volume),
        max_abs_change=fb.high - fb.low,
        quarter=lambda x: x.index.quarter
    ),
    col='quarter'
)

g=sns.FacetGrid(
    quakes.query(
        'parsed_place.isin(["Indonesia", "Papua New Guinea"] )'
        'and magType == "mb"'
    ),
    row='tsunami',
    col='parsed_place',
    height=4
)
g = g.map(sns.histplot, 'mag', kde=True)

# 2-formatting_plots.ipynb

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

fb = pd.read_csv(
    'C:/Users/charlotte.henstock/PycharmProjects/pythonProject2/ch_06/data/fb_stock_prices_2018.csv', index_col='date', parse_dates=True
)
covid = pd.read_csv('C:/Users/charlotte.henstock/PycharmProjects/pythonProject2/ch_06/data/covid19_cases.csv').assign(
    date=lambda x: pd.to_datetime(x.dateRep, format='%d/%m/%Y')
).set_index('date').replace(
    'United_States_of_America', 'USA'
).sort_index()['2020-01-18':'2020-09-18']

fb.close.plot()
plt.title('FB Closing Price')
plt.xlabel('date')
plt.ylabel('price ($)')
plt.show()

fb.iloc[:,:4].plot(subplots=True, layout=(2, 2), figsize=(12, 5))
plt.title('Facebook 2018 Stock Data')
plt.ylabel('price ($)')
plt.show()

axes = fb.iloc[:, :4].plot(subplots=True, layout=(2, 2), figsize=(12, 5))
plt.suptitle('Facebook 2018 Stock Data')
for ax in axes.flatten():
    ax.set_ylabel('price ($)')
plt.show()

fb.assign(
    ma=lambda x: x.close.rolling(20).mean()
).plot(
    y=['close', 'ma'],
    title='FB closing price in 2018',#
    label=['closing price', '20D moving average'],
    style=['-', '--']
)
plt.legend(loc='lower left')
plt.ylabel('price ($)')
plt.show()

new_cases = covid.reset_index().pivot(
    index='date', columns='countriesAndTerritories', values='cases'
).fillna(0)

pct_new_cases = new_cases.apply(lambda x: x/new_cases.apply('sum', axis=1), axis=0)[
    ['Italy', 'China', 'Spain', 'USA', 'India', 'Brazil']
].sort_index(axis=1).fillna(0)

ax = pct_new_cases.plot(
    figsize=(12, 7), style=['-']*3+['--', ':', '-.'],
    title='Percentage of the world\'s new covid cases\n(source: ecdc)'
)

ax.legend(title='Country', framealpha=0.5, ncol=2)
ax.set_xlabel('')
ax.set_ylabel('percentage of the world\'s covid cases')

for spine in ['top', 'right']:
    ax.spines[spine].set_visible(False)
plt.show()


from matplotlib.ticker import PercentFormatter

ax = pct_new_cases.plot(
    figsize=(12, 7), style=['-']*3+['--', ':', '-.'],
    title='Percentage of the world\'s new covid cases\n (source: ecdc)'
)

tick_locs=covid.index[covid.index.day==18].unique()
tick_labels=[loc.strftime('%b %d\n%Y') for loc in tick_locs]
plt.xticks(tick_locs, tick_labels)

ax.legend(title='', framealpha=0.5, ncol=2)
ax.set_xlabel('')
ax.set_ylabel('percentage of the world\'s covid cases')
ax.set_ylim(0, None)
ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))

for spine in ['top', 'right']:
    ax.spines[spine].set_visible(False)

plt.show()

from matplotlib.ticker import EngFormatter

ax = covid.query('continentExp !="Other"').groupby([
    'continentExp', pd.Grouper(freq='1D')
]).cases.sum().unstack(0).apply('cumsum').plot(
    style=['-', '-', '--', ':', '-.'],
    title='Cumulative covid cases per continent\n(source- ecdc)'
)

ax.legend(title='',loc='center left')
ax.set(xlabel='', ylabel='total covid cases')
ax.yaxis.set_major_formatter(EngFormatter())

for spine in ['top', 'right']:
    ax.spines[spine].set_visible(False)
plt.show()

ax = new_cases.New_Zealand['2020-04-18':'2020-09-18'].plot(
    title='Daily new covid cases in New Zealand\n(source: ECDC)'
)
ax.set(xlabel='', ylabel='new COVID-19 cases')
for spine in ['top', 'right']:
    ax.spines[spine].set_visible(False)

from matplotlib.ticker import MultipleLocator

ax = new_cases.New_Zealand['2020-04-18':'2020-09-18'].plot(
    title='Daily new COVID-19 cases in New Zealand\n(source: ECDC)'
)
ax.set(xlabel='', ylabel='new COVID-19 cases')
ax.yaxis.set_major_locator(MultipleLocator(base=3))

for spine in ['top', 'right']:
    ax.spines[spine].set_visible(False)


# 3-customizing_visualizations.ipynb

import matplotlib.pyplot as plt
import pandas as pd

fb = pd.read_csv(
    'C:/Users/charlotte.henstock/PycharmProjects/pythonProject2/ch_06/data/fb_stock_prices_2018.csv', index_col='date', parse_dates=True
)
quakes = pd.read_csv('C:/Users/charlotte.henstock/PycharmProjects/pythonProject2/ch_06/data/earthquakes.csv')

from stock_analysis import StockAnalyzer
fb_analyzer = StockAnalyzer(fb)

support, resistance = (
    getattr(fb_analyzer, stat)(level=3) for stat in ['support', 'resistance']
)
support, resistance

fb.close['2018-12'].plot(title='FB Closing Price December 2018')
plt.axhline(
    y=resistance, color='r', linestyle='--',
    label=f'resistance (${resistance:,.2f})'
)
plt.axhline(
    y=support, color='g', linestyle='--',
    label=f'support (${support:,.2f}'
)
plt.ylabel('price ($)')
plt.legend()

from viz import std_from_mean_kde
std_from_mean_kde??

ax = std_from_mean_kde(
    quakes.query(
        'magType== "mb" and parsed_place =="Indonesia"'
    ).mag
)
ax.set_title('mb magnitude distribution in Indonesia')
ax.set_xlabel('mb earthquake magnitude')
plt.show()


ax = fb.close.plot(title='FB Closing Price')
ax.axhspan(support, resistance, alpha=0.2)
plt.ylabel('Price ($)')
plt.show()

fb_q4 = fb.loc['2018-Q4']
plt.fill_between(fb_q4.index, fb_q4.high, fb_q4.low)
plt.xticks(['2018-10-01', '2018-11-01', '2018-12-01', '2019-01-01'])
plt.xlabel('date')
plt.ylabel('price ($)')
plt.title('FB differential between high and low price q4 2018')

fb_q4 = fb.loc['2018-q4']
plt.fill_between(fb_q4.index, fb_q4.high, fb_q4.low)
plt.xticks(['2018-10-01', '2018-11-01', '2018-12-01', '2019-01-01'])
plt.xlabel('date')
plt.ylabel('price ($)')
plt.title('FB differential between high and low pirce q4 2018')

fb_q4 = fb.loc['2018-Q4']
plt.fill_between(
    fb_q4.index, fb_q4.high, fb_q4.low,
    where=fb_q4.index.month == 12,
    color='khaki', label='December differential'
)
plt.plot(fb_q4.index, fb_q4.high, '--', label='daily high')
plt.plot(fb_q4.index, fb_q4.low, '--', label='daily low')
plt.xticks(['2018-10-01', '2018-11-01', '2018-12-01', '2019-01-01'])
plt.xlabel('date')
plt.ylabel('price ($)')
plt.legend()
plt.title('FB differential between high and low price Q4 2018')
plt.show()


ax = fb.close.plot(title='FB Closing Price 2018', figsize=(15, 3))
ax.set_ylabel('price ($)')

ax.axhspan(support, resistance, alpha=0.2)

plt.annotate(
    f'support\n(${support:,.2f})',
    xy=('2018-12-31', support),
    xytext=('2019-01-21', support),
    arrowprops={'arrowstyle': '->'}
)
plt.annotate(
    f'resistance\n(${resistance:,.2f})',
    xy=('2018-12-23', resistance)
)

for spine in ['top', 'right']:
    ax.spines[spine].set_visible(False)

close_price = fb.loc['2018-07-25', 'close']
open_price = fb.loc['2018-07-26', 'open']
pct_drop = (open_price - close_price) / close_price
fb.close.plot(title='FB cling price 2018', alpha=0.5)
plt.annotate(
    f'{pct_drop:.2%}', va='center',
    xy=('2018-07-27', (open_price+close_price)/2),
    xytext=('2018-08-20', (open_price+close_price)/2),
    arrowprops=dict(arrowstyle='-[,widthB=4.0, lengthB=0.2')
)
plt.ylabel('price ($)')

close_price= fb.loc['2018-07-25', 'close']
open_price = fb.loc['2018-07-26', 'open']
pct_drop=(open_price - close_price) / close_price

fb.close.plot(title='FB Closing Price 2018', alpha=0.5)
plt.annotate(
    f'{pct_drop:.2%}', va='center',
    xy=('2018-07-27', (open_price+close_price)/2),
    xytext=('2018-08-20', (open_price+close_price)/2),
    arrowprops=dict(arrowstyle='-[,widthB=3.0, lengthB=0.2'),
    color='red',
    fontsize='14',
    fontweight='medium'
)
plt.ylabel('price ($)')

fb.plot(
    y='open',
    figsize=(5, 3),
    color='#8000FF',
    legend=False,
    title='Evolution of FB opening price in 2018'
)
plt.ylabel('price ($)')


fb.plot(
    y='open',
    figsize=(5, 3),
    color=(128 / 255, 0, 1),
    legend=False,
    title='Evolution of FB Opening Price in 2018'
)
plt.ylabel('price ($)')

from matplotlib import cm
cm.datad.keys()

ax=fb.assign(
    rolling_min=lambda x: x.low.rolling(20).min(),
    rolling_max=lambda x: x.high.rolling(20).max()
).plot(
    y=['rolling_max', 'rolling_min'],
    colormap='coolwarm_r',
    label=['20D rolling max', '20D rolling min'],
    style=[':', '--'],
    figsize=(12, 3),
    title='FB closing price in 2018 oscillating between '
    '20-day rolling minimum and maximum price'
)
ax.plot(fb.close, 'purple', alpha=0.25, label='closing price')
plt.legend()
plt.ylabel('price ($)')

cm.get_cmap('ocean')(.5)


import color_utils
my_colors = ['#800080', '#FFA500', '#FFFF00']
rgbs = color_utils.hex_to_rgb_color_list(my_colors)
my_cmap = color_utils.blended_cmap(rgbs)
color_utils.draw_cmap(my_cmap, orientation='horizontal')

import seaborn as sns
sns.palplot(sns.color_palette("BuGn_r"))

diverging_cmap=sns.choose_diverging_palette()
from matplotlib.colors import ListedColormap
color_utils.draw_cmap(ListedColormap(diverging_cmap), orientation='horizontal')

import itertools
colors=itertools.cycle(['#ffffff', '#f0f0f0', '#000000'])
colors
next(colors)

from matplotlib.colors import ListedColormap
red_black = ListedColormap(['red', 'black'], N=2000)
[red_black(i) for i in range(3)]


def color_generator():
    for year in range(1992, 200019):  # integers in [1992, 200019)
        if year % 100 == 0 and year % 400 != 0:
            # special case (divisible by 100 but not 400)
            color = '#f0f0f0'
        elif year % 4 == 0:
            # leap year (divisible by 4)
            color = '#000000'
        else:
            color = '#ffffff'
        yield color


year_colors = color_generator()
year_colors

next(year_colors)

























