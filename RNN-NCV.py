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
#print(data)
print(data2[data2.columns[1]].shape)
#numero de splits: spt
spt=5
it=0
tscv = TimeSeriesSplit(n_splits=spt-1)
def splitData(dataset, split):
    arraySplit = []
    for i in range(split):
        #tipo dataFrame
        dFrames = dataset[int(dataset.shape[0]*(i/split)):int(dataset.shape[0]*((i+1)/split))]
        #array = array.values
        arraySplit.append(dFrames)
    return arraySplit
dataSplit = splitData(data,spt)
#FORWARD CHAINING CROSS VALIDATION
#Ejemplo: Arreglo = [1 2 3 4]
#Splits = 3
#Iteraciones:
#1. Train = [1] Test [2]
#2. Train = [1 2] Test [3]
#3. Train = [1 2 3] Test [4]
def soft_acc(y_true, y_pred):
    return K.mean(K.equal(K.round(y_true), K.round(y_pred)))

modeloVol = Sequential()
#modeloVel = Sequential()
capas=5


# HIPERPARAMETROS
# MODIFICAR PARA GRIDSEARCH
neuronas=100
epocas=450
batchsize=100

for train, test in tscv.split(data): 
    #usar la misma semilla para obtener resultados consistentes
    np.random.seed(0)
    it=it+1
    a=data[data.columns[3:6:2]]
    b=data[data.columns[8:10:1]]
    #b=data[data.columns[8:11:1]]
    a=a.join(b)
    print(a)
    c=data[data.columns[1]]
    d=data[data.columns[4:7:2]]
    e=data[data.columns[2]]
    #Obtener subconjuntos de entrenamiento y prueba para volumen y velocidad
    #Estan como dataFrames, convertir en Numpy arrays
    X_trainVol, X_testVol=a.iloc[train].values, a.iloc[test].values
    Y_trainVol, Y_testVol=c.iloc[train].values, c.iloc[test].values
    #print(X_trainVol, X_testVol)
    #Redimensionar variables independientes X en arreglos tridimensionales:
    #[#Filas, unidad de saltos de tiempo: 1 (hora), #Columnas]
    X_trainVol = X_trainVol.reshape((X_trainVol.shape[0],1,X_trainVol.shape[1]))
    X_testVol = X_testVol.reshape((X_testVol.shape[0],1,X_testVol.shape[1]))
    print(X_trainVol.shape, Y_trainVol.shape, X_testVol.shape, Y_testVol.shape)
    
    modeloVol.add(LSTM(neuronas,activation='relu',return_sequences=True,
                       input_shape=(X_trainVol.shape[1], X_trainVol.shape[2])))
    modeloVol.add(LSTM(neuronas,activation='relu',return_sequences=True))
    modeloVol.add(LSTM(neuronas,activation='relu',return_sequences=True))
    modeloVol.add(LSTM(neuronas,activation='relu',return_sequences=True))
    modeloVol.add(LSTM(neuronas,activation='relu'))
    modeloVol.add(Dense(1,activation='linear'))
    modeloVol.compile(loss='mse', optimizer='adam', metrics=[soft_acc])
    #entrenar el modelo
    history=modeloVol.fit(X_trainVol,Y_trainVol,epochs=epocas,batch_size=batchsize,
                          validation_data=(X_testVol,Y_testVol),verbose=0,shuffle=False)
    if it < spt-1:
        modeloVol.pop()
        modeloVol.pop()
        modeloVol.pop()
        modeloVol.pop()
        modeloVol.pop()
        modeloVol.compile(loss='mse', optimizer='adam')           
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
prediccion=modeloVol.predict(dataTest)
print(dataTest.shape)
score=modeloVol.evaluate(dataTest,data2[data2.columns[1]],batch_size=batchsize)
#revertir cambio de dimensiones
dataTest = dataTest.reshape(dataTest.shape[0], dataTest.shape[2])
uniq, ind = np.unique(data['Semana'].values, return_index=True)
plt.figure(1)
plt.title("Volumen de trafico - Predicción")
plt.xlabel('Semana')
plt.xticks(data.index[ind][::3], uniq[::3])
plt.ylabel('GB')
plt.plot(prediccion,label='Predicción',color='green')
plt.plot(data2[data2.columns[1]],label='Original',color='red', linestyle='--')
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
print("Score y Accuracy de Prueba Volumen con Cross-validation: ", score)