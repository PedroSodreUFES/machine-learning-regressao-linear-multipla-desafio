import joblib
import matplotlib.pyplot as plt
import pandas as pd
import pingouin as pg
import seaborn as sns
from scipy.stats import kstest, shapiro, zscore
from statsmodels.stats.diagnostic import lilliefors, het_goldfeldquandt
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# 1: Carregar dados e checar valores faltantes.
# Não há valores nulos ou faltantes.
# Coluna id será inútil.
print("Carregar dados")
df_aluguel = pd.read_csv("./dataset/dataset_aluguel.csv")
print(df_aluguel.info())
print("\nRemovendo coluna id...\n")
df_aluguel.drop(columns=['id'], inplace=True)
df_aluguel.columns = [
    'metros_quadrados',
    'n_quartos',
    'idade',
    'garagem',
    'periferia',
    'suburbio',
    'aluguel'
]
print(df_aluguel.describe())

# 2: Análise Exploratória dos Dados
# Não há outliers nos valores
print("\nAnálise Exploratória.")
df_aluguel_eda = df_aluguel.copy()

sns.pairplot(data=df_aluguel_eda, y_vars=['aluguel'])
plt.savefig("./dataviz/pairplot.png")
plt.close()

sns.boxplot(data=df_aluguel_eda, x='n_quartos', y='aluguel')
plt.xlabel("Número de Quartos")
plt.ylabel("Valor do Aluguel")
plt.savefig('./dataviz/n-quartos-x-aluguel-boxplot.png')
plt.close()

sns.boxplot(data=df_aluguel_eda, x='periferia', y='aluguel')
plt.xlabel("Está na Periferia")
plt.ylabel("Valor aluguel")
plt.savefig('./dataviz/periferia-x-aluguel-boxplot.png')
plt.close()

sns.scatterplot(data=df_aluguel_eda, x='metros_quadrados', y='aluguel')
plt.xlabel('Metros Quadrados')
plt.ylabel("Valor do Aluguel")
plt.savefig("./dataviz/metros-quadrados-x-aluguel-scatter.png")
plt.close()

sns.boxplot(data=df_aluguel_eda, y='aluguel')
plt.savefig("./dataviz/aluguel-hist.png")
plt.close()

sns.heatmap(
    data=df_aluguel_eda.corr("pearson"), 
    vmin=-1,
    vmax=1,
    annot=True,
    cmap='crest'    
)
plt.savefig("./dataviz/pearson-corr-heatmap.png")
plt.close()

sns.heatmap(
    data=df_aluguel_eda.corr("spearman"), 
    vmin=-1,
    vmax=1,
    annot=True,
    cmap='crest'    
)
plt.savefig("./dataviz/spearman-corr-heatmap.png")
plt.close()

sns.heatmap(
    data=df_aluguel_eda.corr("pearson")[['aluguel']].sort_values(by='aluguel', ascending=False), 
    vmin=-1,
    vmax=1,
    annot=True,
    cmap='crest'    
)
plt.savefig("./dataviz/aluguel-pearson-corr-heatmap.png")
plt.close()

sns.heatmap(
    data=df_aluguel_eda.corr("spearman")[['aluguel']].sort_values(by='aluguel', ascending=False), 
    vmin=-1,
    vmax=1,
    annot=True,
    cmap='crest'    
)
plt.savefig("./dataviz/aluguel-spearman-corr-heatmap.png")
plt.close()
