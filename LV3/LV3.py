'''
ZADATAK 1:
import pandas as pd

mtcars = pd.read_csv("C:/Users/Bruno/Desktop/LV3/mtcars.csv")

most_consumption = mtcars.sort_values("mpg", ascending=True).head(5)
print("5 automobila s najvećom potrošnjom:\n", most_consumption[['car', 'mpg']])

eight_cylinders = mtcars[mtcars['cyl'] == 8].sort_values("mpg").head(3)
print("3 auta s 8 cilindara i najmanjom potrošnjom:\n", eight_cylinders[['car', 'mpg']])

six_cylinders_avg = mtcars[mtcars['cyl'] == 6]["mpg"].mean()
print("Srednja potrošnja automobila sa 6 cilindara:", six_cylinders_avg)

four_cylinders_filtered = mtcars[(mtcars['cyl'] == 4) & ((mtcars['wt'] * 1000 > 2000) & (mtcars['wt'] * 1000 <= 2200))]
four_cylinders_avg = four_cylinders_filtered["mpg"].mean()
print("Srednja potrošnja automobila s 4 cilindra mase 2000-2200 lbs.:", four_cylinders_avg)

manual_count = len(mtcars[mtcars['am'] == 1])
auto_count = len(mtcars[mtcars['am'] == 0])
print(f"Broj automobila s ručnim mjenjačem: {manual_count}, s automatskim mjenjačem: {auto_count}")

auto_powerful = len(mtcars[(mtcars['am'] == 0) & (mtcars['hp'] > 100)])
print("Broj automobila s automatskim mjenjačem i snagom preko 100 konja:", auto_powerful)
'''


''''
ZADATAK 2:
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

mtcars = pd.read_csv('C:/Users/Bruno/Desktop/LV3/mtcars.csv')


mtcars['cyl'] = mtcars['cyl'].astype(str) + ' cilindara'
mtcars['am'] = mtcars['am'].apply(lambda x: 'Ručno' if x == 1 else 'Automatski')

plt.figure(figsize=(10, 6))
sns.barplot(x='cyl', y='mpg', data=mtcars, ci=None, palette='pastel')
plt.title('Potrošnja automobila s 4, 6 i 8 cilindara')
plt.ylabel('Milja po galonu (mpg)')
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(x='cyl', y='wt', data=mtcars, palette='pastel')
plt.title('Distribucija težine automobila po broju cilindara')
plt.ylabel('Težina (1000 lbs)')
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(x='am', y='mpg', data=mtcars, palette='pastel')
plt.title('Potrošnja automobila s ručnim i automatskim mjenjačem')
plt.ylabel('Milja po galonu (mpg)')
plt.show()

plt.figure(figsize=(10, 6))
sns.scatterplot(x='hp', y='qsec', hue='am', data=mtcars, palette='pastel', s=100)
plt.title('Odnos ubrzanja i snage za ručne i automatske mjenjače')
plt.xlabel('Snaga (hp)')
plt.ylabel('Ubrzanje (kvartalna milja u sek.)')
plt.legend(title='Mjenjač')
plt.show()
'''