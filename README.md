# Explicación del código
### Importar librerias necesarias
```inserta código
import pandas as pd  # importa la libreria pandas y la nombra como pd
import matplotlib.pyplot as plt  # importa la libreria para hacer graficas
import numpy as np
from numpy.linalg import inv
import math
import utm
```
### Cargar los archivos de la ruta real y los datos medidos
```
datosReales = pd.read_csv('Ruta2Real.csv',
                          header=0)  # importa el archivo .csv e indica que el encabezado esta en la fila 0
datosGPS = pd.read_csv('Ruta2.csv',
                       header=0)  # importa el archivo .csv e indica que el encabezado esta en la fila 0
```
### Colocar los datos de latitud y longitud en arreglos
```lonReal = datosReales['lon']
latReal = datosReales['lat']
lonMed = datosGPS['lon']
latMed = datosGPS['lat']
velocidad = datosGPS['speed'] * 1000 / 3600
grapmedx = []
grapmedy = []
graprealx = []
graprealy = []
```
# Definir funciones
### Transformar de coordenadas geográficas a UTM(en metros)
```
def GEOTOUTM():
    for i in np.arange(0, len(lonMed)):
        # U = (datosGPS['speed'].ix[i])/11170000
        # U = np.array([[0, 0, 0, 0]])
        # (xmedida, ymedida, angulo) = haversine(latMed.ix[0], lonMed.ix[0], latMed.ix[i], lonMed.ix[i])
        (x, y, zonanumber, zonaletter) = utm.from_latlon(latMed.ix[i], lonMed.ix[i])
        grapmedx.append(x)
        grapmedy.append(y)

    for i in np.arange(0, len(lonReal)):
        # U = (datosGPS['speed'].ix[i])/11170000
        # U = np.array([[0, 0, 0, 0]])
        # (xmedida, ymedida, angulo) = haversine(latMed.ix[0], lonMed.ix[0], latMed.ix[i], lonMed.ix[i])
        (x, y, zonanumber, zonaletter) = utm.from_latlon(latReal.ix[i], lonReal.ix[i])
        graprealx.append(x)
        graprealy.append(y)
```
### Graficar los datos
```
def GRAFICAS():
    plt.plot(grapmedx, grapmedy, '.', label='Mediciones')
    plt.plot(realx, realy, '-', label='Ruta Real')
    plt.plot(predicho[:, 0], predicho[:, 1], '.-', label='Estimacion')
    plt.xlabel('X (m)')  # nombra el eje x
    plt.ylabel('Y (m)')  # nombra el eje y
    plt.title('Position')  # pone titulo a la grafica
    plt.legend()
    plt.show()

    plt.clf()
    plt.plot(velocidad, '.', label='Mediciones')
    plt.plot(vel, predichoV[:, 2], '.-', label='Estimacion')
    plt.ylabel('V (m/s)')  # nombra el eje x
    plt.title('Velocity')  # pone titulo a la grafica
    plt.legend()
    plt.show()

```
## Filtro de kalman 
### Etapa de predicción del KF
```
def kf_predict(X, P, A, Q, B, U):
    X = np.dot(A, X) + np.dot(B, U)
    P = np.dot(np.dot(A, P), A.T) + Q
    return X, P
```
### Etapa de corrección del KF
 ```
 def kf_update(X, P, Y, H, R):
    V = Y - np.dot(H, X)
    S = R + np.dot(np.dot(H, P), H.T)
    K = np.dot(np.dot(P, H.T), inv(S))
    X = X + np.dot(K, V)
    P = P - np.dot(np.dot(K, S), K.T)
    return X, P
 ```

# Aplicación 
* La variable Niter indica el número de iteraciones que se realizan entre cada medición
* Ecuaciónes ingresadas:  

![1](http://latex.codecogs.com/gif.latex?x_%7Bk%7D%20%3D%20x_%7Bk-1%7D%20&plus;%20v%5Ccdot%20%5CDelta%20t%20&plus;%20%5Cfrac%7B1%7D%7B2%7Da%5Ccdot%20%5CDelta%20t%5E%7B2%7D)  

![2](http://latex.codecogs.com/gif.latex?y_%7Bk%7D%20%3D%20y_%7Bk-1%7D%20&plus;%20v%5Ccdot%20%5CDelta%20t%20&plus;%20%5Cfrac%7B1%7D%7B2%7Da%5Ccdot%20%5CDelta%20t%5E%7B2%7D)

```
GEOTOUTM()  # convierte de coordenadas geograficas a UTM(coordenadas X e Y)

aux = 0
Niter = 20
dt = 1.0 / Niter

A = np.array([[1, dt, 0, 0], [0, 1, 0, 0], [0, 0, 1, dt], [0, 0, 0, 1]])
X = np.array([[graprealx[0]], [0.0], [graprealy[0]], [0.0]])
H = np.array([[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0]]).reshape(4, 4)
#Q = np.array([[0.15, 0.5, 0.15, 0.5], [0.15, 0.5, 0.15, 0.5], [0.15, 0.5, 0.15, 0.5], [0.15, 0.5, 0.15, 0.5]])
B = np.array([[(dt ** 2) / 2.0], [dt], [(dt ** 2) / 2.0], [dt]])
Q = np.dot(B, B.T)
#Q = 0.15
P = np.diag((0.0, 0.0, 0.0, 0.0))  # Se inicia en cero porque conosco los valores reales
R = np.eye(4)

grapmedx = map(int, grapmedx)
grapmedy = map(int, grapmedy)
realx = np.linspace(graprealx[0], graprealx[graprealx.__len__() - 1], len(lonMed)* int(Niter))
realy = np.linspace(graprealy[0], graprealy[graprealy.__len__() - 1], len(lonMed)* int(Niter))

Y = np.array([[grapmedx[0]], [0], [grapmedy[0]], [0]])
predicho = np.zeros((len(lonMed) * int(Niter), 2))
predichoV = np.zeros((len(lonMed) * int(Niter), 3))
vel = np.linspace(0, len(lonMed), len(lonMed) * int(Niter))
aux2 = 0


for i in np.arange(0, len(lonMed)):
    U = ((velocidad.ix[0] - velocidad.ix[i]) / dt)
    #U = 0
    #U = (datosGPS['speed'].ix[i]/dt)

    X = np.array([[realx[aux2]], [0.0], [realy[aux2]], [0.0]])
    (X, P) = kf_predict(X, P, A, Q, B, U)
    Y = np.array([[grapmedx[i]], [0], [grapmedy[i]], [0]])
    (X, P) = kf_update(X, P, Y, H, R)
    aux = aux + 1
    for j in np.arange(0, Niter):
        X = np.array([[realx[aux2]], [0.0], [realy[aux2]], [0.0]])
        (X, P) = kf_predict(X, P, A, Q, B, U)
        (X, P) = kf_update(X, P, Y, H, R)
        predicho[aux2, 0] = X[0, 0]
        predicho[aux2, 1] = X[2, 0]
        predichoV[aux2, 0] = X[1, 0]
        predichoV[aux2, 1] = X[3, 0]
        predichoV[aux2, 2] = math.sqrt((predichoV[aux2, 0])**2+(predichoV[aux2, 1])**2)
        aux2 = aux2 + 1
```

GRAFICAS()

# NOTAS
* El archivo prueba2_1.py contiene el código de python en su version 2.7
* Los archivos con extención .csv son los archivos de las trayectorias realizadas
* El algoritmo hace una estimación de la posición cada 0.05 segundos(Niter = 20)
