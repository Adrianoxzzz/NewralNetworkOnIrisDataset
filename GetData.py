from ucimlrepo import fetch_ucirepo 
import numpy as np
# fetch dataset 
iris = fetch_ucirepo(id=53) 
  
# data (as pandas dataframes) 
X = iris.data.features 
y = iris.data.targets 
  
# metadata 

inputData =np.array(X)
OutputData = np.array(y)


# Definir el mapeo de números de etiquetas a nombres de especies
label_to_name = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}

# Convertir etiquetas numéricas a nombres de especies
y_test_names,y_train_names = np.split(OutputData,[int(0.8*len(OutputData))])

# Definir la proporción de entrenamiento
train_ratio = 0.8
train_size = int(train_ratio * len(inputData))

# Dividir los datos en entrenamiento y prueba
train_data, test_data = np.split(inputData, [train_size])

print("Ytest",y_test_names,"Ytrain",y_train_names,"Xtrain",train_data,"Xtest",test_data)

# variable information 
#print(iris.variables) 
