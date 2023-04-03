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