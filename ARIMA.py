import numpy as np
from numpy import array
from numpy import random
import pandas as pd
import matplotlib.pyplot as plt
from keras import backend as K
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm
from statsmodels.tsa.arima_model import ARIMA
from pandas.plotting import register_matplotlib_converters
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_squared_error
from scipy.special import boxcox, inv_boxcox
import pmdarima as pm

register_matplotlib_converters()
#Iniciar una nueva sesion en Keras y borrar rastros de cualquier sesion anterior
K.clear_session()
path1 = r'New CSVs\full dataset.csv'
path2 = r'New CSVs\ciclo 2019-0.csv'
##COLUMNAS FILTRADAS, ESTAN COMO VALORES CRUDOS (RAW)
#[Hora, Trafico Volumen Suma, Trafico Volumen In, Trafico Volumen Out, Trafico Velocidad Suma, Trafico Velocidad In, Trafico Velocidad Out]
data=pd.read_csv(path1,usecols=(0,3,5,7,9,11,13))
data2=pd.read_csv(path2,usecols=(0,3,5,7,9,11,13))

#PASO 1
#Depurar el dataset
#Eliminar las dos ultimas filas sin valores numericos
data = data.drop(data.index[[(data.shape[0]-2),(data.shape[0]-1)]])
data2 = data2.drop(data2.index[[(data2.shape[0]-2),(data2.shape[0]-1)]])

#Eliminar la primera fila con entradas vacias
data = data.iloc[1:].fillna(data.mean())
data2 = data2.iloc[1:].fillna(data2.mean())
print(data.mean())

def agregarTemp(df):
    idf = []
    hora= []
    dia= []
    semana=[]
    numFilas=df.shape[0]
    for x in range(numFilas):
        idf.append(x+1)
        hora.append((x%24)+1)
        dia.append((int(x/24)%7)+1)
        semana.append(int(x/(24*7))+1)        
    df=df.assign(ID=idf)
    df=df.assign(Hr=hora)
    df=df.assign(Dia=dia)
    df=df.assign(Semana=semana)
    return df
    
data = agregarTemp(data)
data2 = agregarTemp(data2)

#print(data[['ID','Hr','Dia','Semana']])
##Estandarizacion de la data
#los valores crudos estan en bytes, requieren conversion
def conversionGB(dataset):
    #columnas volumen, convertir a Gigabytes(GB)
    volumenTotal = (((dataset[dataset.columns[1]]/1024)/1024)/1024)
    dataset[dataset.columns[1]] = volumenTotal
    volumenIn = (((dataset[dataset.columns[3]]/1024)/1024)/1024)
    dataset[dataset.columns[3]] = volumenIn
    volumenOut = (((dataset[dataset.columns[5]]/1024)/1024)/1024)
    dataset[dataset.columns[5]] = volumenOut   
    return dataset

data = conversionGB(data)
data2 = conversionGB(data2)

uniq, ind = np.unique(data['Semana'].values, return_index=True)
plt.figure(1)
plt.title("Datos originales")
plt.xlabel("Semana")
plt.ylabel("GB")
plt.xticks(data.index[ind][::3], uniq[::3])
plt.plot(data[data.columns[1]])
plt.show()

VolTotDiff1 = data.iloc[:,1].diff().fillna(data.iloc[:,1])

#plt.figure(2)
#plt.title("Diff1")
#plt.xlabel("Semana")
#plt.ylabel("GB")
#plt.xticks(data.index[ind], uniq)
#plt.plot(VolTotDiff1)
#plt.show()

#Medir estacionariedad de datos

#filter = data[data.columns[1]].values <= 0
#data.loc[filter] = 0.00001

X=data[data.columns[1]].values
#X=boxcox(X,0)

print('Estacionariedad de data original')
split = int(len(X)/2)
X1, X2 = X[0:split], X[split:]
med1, med2 = X1.mean(), X2.mean()
var1, var2 = X1.var(), X2.var()
print('media1=%f, media2=%f' % (med1, med2))
print('varianza1=%f, varianza2=%f' % (var1, var2))
print()

result = adfuller(data[data.columns[1]].values)
print('Valor ADF: {}'.format(result[0]))
print('p-value: {}'.format(result[1]))
print('Valores críticos:')
for key, value in result[4].items():
    print('\t{}: {}'.format(key, value))
print('--------------------------------------------')
print('Estacionariedad de data diferenciada en orden 1')
X1= VolTotDiff1.values
split1 = int(len(X1)/2)
X11, X21 = X1[0:split1], X1[split1:]
med11, med21 = X11.mean(), X21.mean()
var11, var21 = X11.var(), X21.var()
print('media1=%f, media2=%f' % (med11, med21))
print('varianza1=%f, varianza2=%f' % (var11, var21))
print()

result1 = adfuller(VolTotDiff1.values)
print('Valor ADF: {}'.format(result1[0]))
print('p-value: {}'.format(result1[1]))
print('Valores críticos:')
for key, value in result1[4].items():
    print('\t{}: {}'.format(key, value))
    

plot_pacf(data[data.columns[1]].values, zero=False)
plt.figure(1)
plt.show()

plot_acf(data[data.columns[1]].values, zero=False)
plt.figure(2)
plt.show()

orden= (1,0,7)
orden_sarima=(1,0,7,12)

modelo = sm.tsa.statespace.SARIMAX(X, order=orden, seasonal_orden=orden_sarima)
model_fit = modelo.fit(disp=0)
print(model_fit.summary())

resultados = model_fit.predict(dynamic=False)
#resultados = inv_boxcox(resultados, 0)
np_res= np.array(resultados)
np_res[np_res<=0]=0
mse=mean_squared_error(data[data.columns[1]].values,np_res)/100    
print(np_res)

uniq, ind = np.unique(data['Semana'].values, return_index=True)

plt.figure(3)
plt.title("Datos originales")
plt.xlabel("Semana")
plt.ylabel("GB")
plt.xticks(data.index[ind][::3], uniq[::3])
plt.plot(data[data.columns[1]], label='Original', color='blue')
plt.plot(np_res, label='Predicciones', color='red')
plt.show()

print("Error cuadrático medio: ", mse)
