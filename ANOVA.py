from cgitb import text
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.formula.api import ols
import seaborn as sns
import numpy as np
from statsmodels.stats.multicomp import pairwise_tukeyhsd, MultiComparison

# Hyphothesis: How age will affect totalKg in powerlifting for 4 choosen countries: Poland, Germany, UK, USA

data = pd.read_csv("./openpowerlifting.csv")
plt.style.use('fivethirtyeight')

# Data cleaning
data = data[data.Sex != "F"]  # only boys
data = data[data.Equipment == "Raw"]  # only no equipment
data = data[data.Tested.notna()]  # only tested
data = data[data.Country.notna()]  # only with country
data = data[data.Age.notna()]
data = data[data.TotalKg.notna()]

data = data[data.Age > 18.0]
data = data[data.Age < 25.0]

# Columns droping
data.drop(['MeetState', 'MeetName', 'Date',
          'MeetCountry'], axis=1, inplace=True)

data.to_excel("dlaPiotra.xlsx")


# f, ax = plt.subplots(figsize=(11, 9))
# plt.title('Total weight in kg distributions between sample')
# plt.ylabel('pdf')
# sns.distplot(data.TotalKg)
# plt.show()

# f, ax = plt.subplots(figsize=(11, 9))
# sns.distplot(data[data.Country == 'Poland'].TotalKg, ax=ax, label='Poland')
# sns.distplot(data[data.Country == 'Germany'].TotalKg, ax=ax, label='Germany')
# sns.distplot(data[data.Country == 'UK'].TotalKg, ax=ax, label='UK')
# sns.distplot(data[data.Country == 'USA'].TotalKg, ax=ax, label='USA')
# plt.title('Total weight in kg distributions between each test country')
# plt.legend()
# plt.show()


mod = ols('Age ~ TotalKg', data=data[data.Country == 'Poland']).fit()
aov_table1 = sm.stats.anova_lm(mod, typ=2)
print('ANOVA table for Poland')
print('----------------------')
print(aov_table1)
print()

mod = ols('Age ~ TotalKg', data=data[data.Country == 'Germany']).fit()
aov_table2 = sm.stats.anova_lm(mod, typ=2)
print('ANOVA table for Germany')
print('----------------------')
print(aov_table2)
print()

mod = ols('Age ~ TotalKg', data=data[data.Country == 'UK']).fit()
aov_table3 = sm.stats.anova_lm(mod, typ=2)
print('ANOVA table for UK')
print('----------------------')
print(aov_table3)
print()

mod = ols('Age ~ TotalKg', data=data[data.Country == 'USA']).fit()
aov_table4 = sm.stats.anova_lm(mod, typ=2)
print('ANOVA table for USA')
print('----------------------')
print(aov_table4)
print()

data = data[data.Country == 'USA']
mc = MultiComparison(data['TotalKg'], data['Age'])
tukey_result = mc.tukeyhsd(alpha=0.05)

print(tukey_result)
print('Unique age groups: {}'.format(mc.groupsunique))


with pd.ExcelWriter("ANOVA.xlsx") as writer:
    aov_table1.to_excel(writer, sheet_name="ANOVA_Poland")
    aov_table2.to_excel(writer, sheet_name="ANOVA_Germany")
    aov_table3.to_excel(writer, sheet_name="ANOVA_UK")
    aov_table4.to_excel(writer, sheet_name="ANOVA_USA")
