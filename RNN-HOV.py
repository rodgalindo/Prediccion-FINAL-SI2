import numpy as np
from numpy import array
import pandas as pd
import matplotlib.pyplot as plt
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Activation
from sklearn.model_selection import TimeSeriesSplit
from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)

#Iniciar una nueva sesion en Keras y borrar rastros de cualquier sesion anterior
K.clear_session()
path1 = r'New CSVs\train.csv'
path2 = r'New CSVs\test.csv'
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
    volumenTotal = (((dataset[dataset.columns[1]]/1000)/1000)/1000)
    dataset[dataset.columns[1]] = volumenTotal
    volumenIn = (((dataset[dataset.columns[3]]/1000)/1000)/1000)
    dataset[dataset.columns[3]] = volumenIn
    volumenOut = (((dataset[dataset.columns[5]]/1000)/1000)/1000)
    dataset[dataset.columns[5]] = volumenOut
    #columnas velocidad, convertir a Megabits por segundo (Mbps)
    velocidadTotal = (((dataset[dataset.columns[2]]*8)/1024)/1024)
    dataset[dataset.columns[2]] = velocidadTotal
    velocidadIn = (((dataset[dataset.columns[4]]*8)/1024)/1024)
    dataset[dataset.columns[4]] = velocidadIn
    velocidadOut = (((dataset[dataset.columns[6]]*8)/1024)/1024)
    dataset[dataset.columns[6]] = velocidadOut    
    return dataset

data = conversionGB(data)
data2 = conversionGB(data2)
print(data2[data2.columns[1]].shape)

print(data.drop("Fecha Hora",axis=1))
#PASO 2
#Primero se tiene que dividir el dataset en conjunto de entrenamiento
#y prueba. 
#Entrenamiento = 70% de registros
#Validacion = 30% de registros
def dividirData(val, lim):    
    train = val.iloc[0:lim]
    train = train.values
    validacion = val.iloc[lim:]
    validacion = validacion.values    
    return train, validacion
#X=variables independientes (In,out)
#Y=variable dependiente (Total)
limite = int(7*data.shape[0]/10)

a=data[data.columns[3:6:2]]
print(a)
b=data[data.columns[8:10:1]]
#b=data[data.columns[8:11:1]]
a=a.join(b)
#print(a)
c=data[data.columns[1]]
d=data[data.columns[4:7:2]]
e=data[data.columns[2]]
#Obtener subconjuntos de entrada
Y_train, Y_prue = dividirData(c,limite)
X_train, X_prue = dividirData(a,limite)
#print(data[data.columns[1]])
#dimensiones para modelo LSTM: 
#(cantidad de registros ,unidades de salto de tiempo (1), caracteristicas)
X_train = X_train.reshape((X_train.shape[0],1,X_train.shape[1]))
X_prue = X_prue.reshape((X_prue.shape[0],1,X_prue.shape[1]))
print(X_train.shape, Y_train.shape, X_prue.shape, Y_prue.shape)

#PASO 3
##ENTRENAMIENTO Y VALIDACION DEL VOLUMEN
#definir el modelo 
#GRIDSEARCH MANUAL
#MODIFICAR HIPER PARAMETROS
neuronas=50
epocas=150
batchsize=25

def soft_acc(y_true, y_pred):
    return K.mean(K.equal(K.round(y_true), K.round(y_pred)))

modelo = Sequential()
modelo.add(LSTM(neuronas,activation='relu',return_sequences=True,input_shape=(X_train.shape[1], X_train.shape[2])))
modelo.add(LSTM(neuronas,activation='relu',return_sequences=True))
modelo.add(LSTM(neuronas,activation='relu',return_sequences=True))
modelo.add(LSTM(neuronas,activation='relu',return_sequences=True))
modelo.add(LSTM(neuronas,activation='relu'))
modelo.add(Dense(1,activation='linear'))
modelo.compile(loss='mse', optimizer='adam',  metrics=[soft_acc])
#entrenar el modelo
history=modelo.fit(X_train,Y_train,epochs=epocas,batch_size=batchsize, validation_data=(X_prue,Y_prue),
                      verbose=0,shuffle=False)
#plt.figure(1)
#plt.title("Perdidas - Volumen")
#plt.plot(history.history['loss'], label='Entrenamiento')
#plt.plot(history.history['val_loss'], label='Validacion')
#plt.legend(loc='best')
#plt.show()

##PRUEBA CON MES FEBRERO 2019
#creacion de data de prueba
dataTest = []
dataTest.append(data2[data2.columns[3]])
dataTest.append(data2[data2.columns[5]])
dataTest.append(data2[data2.columns[8]])
dataTest.append(data2[data2.columns[9]])
#dataTest.append(data2[data2.columns[10]])
dataTest = array(dataTest)
dataTest = dataTest.transpose()
#convertir en arreglo tridimensional para realizar una prediccion
dataTest = dataTest.reshape(dataTest.shape[0],1,dataTest.shape[1])
#prediccion del modelo
prediccion=modelo.predict(dataTest)
print(dataTest.shape)
score=modelo.evaluate(dataTest,data2[data2.columns[1]],batch_size=batchsize)
#revertir cambio de dimensiones
dataTest = dataTest.reshape(dataTest.shape[0], dataTest.shape[2])
print(prediccion)
plt.figure(1)
plt.title("Volumen de trafico - Predicción")
plt.xlabel('Hora')
plt.ylabel('GB')
plt.plot(prediccion,label='Predicción',color='green')
plt.plot(data2[data2.columns[1]],label='Original',color='red')
plt.legend(loc='best')
plt.show()
plt.figure(2)
plt.title("Volumen de trafico - Original")
plt.xlabel('Hora')
plt.ylabel('GB')
plt.plot(data2[data2.columns[1]],label='Original',color='red')
#plt.plot(prediccion,label='Predicción',color='blue', linestyle='--')
plt.legend(loc='best')
plt.show()

print("Neuronas: ",neuronas,", Epocas: ",epocas,", Batch size:" , batchsize)
print("-------------------------------------------------------")
print("Score y Accuracy de Prueba Volumen con Hold-out Validation: ", score)