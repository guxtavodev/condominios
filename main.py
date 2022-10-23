import pandas as pd
import matplotlib

dados = pd.read_csv("quinto_andar_transformed_data.csv")
# Visualizando as primeiras linhas
print(dados.head())
# Informações da tabela
print(dados.info())
#Quantidade de valores nulos
print(dados.isnull().sum())
# Tenho que achar soluções para que o valor não seja nulo
#Filtrando valores vazios
print(dados[dados.condominio.isnull()])

# Substituindo valores nulos
print(dados.loc[dados.condominio.isnull(), "condominio"])
dados.loc[dados.condominio.isnull(), "condominio"] = 0
print(dados.isnull().sum())

# Excluindo dados sem IPTU
print(dados[dados.iptu.isnull()])
dados = dados.dropna(axis=0)
print(dados.info())
print(dados.isnull().sum())

# Eliminando coluna url que é desnecessário
dados = dados.drop("url", axis=1)

## Analise exploratoria
print(dados.describe())

#  Calculando a média de aluguel
media = dados.aluguel.sum()/dados.aluguel.count()
print(media)
print(dados.aluguel.median()) # Isso ele pega os valores com mais frequência, ex.: Nisso, significa que existe bastante aluguel barato

# Histograma entre os valores dos alugueis
print(dados.aluguel.hist())

## Verificando as propriedades de maior valor
print(dados.sort_values("aluguel", ascending=False).head(10))

# Vendo se existe valores duplicados(nao pode ter)
print(dados[dados.duplicated()])

# Eliminando dados duplicados
dados = dados.drop_duplicates(keep="last")
print(dados.sort_values("aluguel", ascending=False).head(10))

# Vendo se existe relação entre aluguel e metragem
print(dados.plot.scatter(x="aluguel", y="metragem"))

# Vendo se metro próximo influencia no valor do aluguel
import seaborn as sns
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(15,12))
sns.boxplot(x="metro_prox", y="aluguel", data=dados)

# Será  que o bairro influencia o valor do aluguel?
sns.boxplot(x="aluguel", y="bairro", data=dados, orient="h")

# Quantos registros tem em cada bairro?
print(dados.bairro.value_counts())

# X é variavel preditora(que vamos usar pra fazer a previsao)
# Y é o que queremos prever

from sklearn.model_selection import train_test_split

dados = dados.drop(["bairro","seguro_incendio", "taxa_serviço", "total"], axis=1)

X = dados.drop("aluguel", axis=1)
y = dados.aluguel

# Separando bases
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

print("=== X_train")
print(X_train)

print("=== X_test")
print(y_train)

# Treinando o modelo
from sklearn.linear_model import LinearRegression
reg = LinearRegression().fit(X_train, y_train)

print("Score")

print(reg.score(X_train, y_train))

## Testando..
from sklearn.metrics import mean_absolute_error

pr = reg.predict(X_test)
print(mean_absolute_error(y_test, pr))

from joblib import dump
dump(reg, "rege.joblib")

pro = pd.read_csv("producao.csv", sep=";")
print(pro.head())
pro = pro.drop("bairro", axis=1)
X = pro.drop("aluguel", axis=1)
y = pro.aluguel

from joblib import load

rr = load("rege.joblib")

y_pred = rr.predict(X)

pro["aluguel"] = y_pred
print(pro.head())

import openpyxl

pro.to_excel("resultado.xlsx")
