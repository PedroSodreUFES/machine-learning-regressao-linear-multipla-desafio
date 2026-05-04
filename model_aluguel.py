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

# 1: Carregar dados e checar valores faltantes
# Não há valores nulos ou faltantes
print("Carregar dados")
df_aluguel = pd.read_csv("./dataset/dataset_aluguel.csv")
print(df_aluguel.info())

# 2: Análise Exploratória dos Dados
df_aluguel_eda = df_aluguel.copy()

sns.pairplot(data=df_aluguel_eda, y_vars=['valor_aluguel'])
plt.savefig("./dataviz/pairplot.png")
plt.close()
