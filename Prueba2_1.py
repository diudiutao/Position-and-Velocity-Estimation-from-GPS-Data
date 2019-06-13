import numpy as np
# importar librerias necesarias
import pandas as pd  # importa la libreria pandas y la nombra como pd
import matplotlib.pyplot as plt  # importa la libreria para hacer graficas
import numpy as np
from numpy.linalg import det
from numpy.linalg import inv
import math
import utm

datosReales = pd.read_csv('Ruta2Real.csv',
                          header=0)  # importa el archivo .csv e indica que el encabezado esta en la fila 0
datosGPS = pd.read_csv('Ruta2.csv',
                       header=0)  # importa el archivo .csv e indica que el encabezado esta en la fila 0

lonReal = datosReales['lon']
latReal = datosReales['lat']
lonMed = datosGPS['lon']
latMed = datosGPS['lat']
velocidad = datosGPS['speed'] * 1000 / 3600
grapmedx = []
grapmedy = []
graprealx = []
graprealy = []


def BEARING(lat1, lon1, lat2, lon2):
    lat1 = math.radians(lat1)
    lat2 = math.radians(lat2)
    lon1 = math.radians(lon1)
    lon2 = math.radians(lon2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    R = 6372.795477598
    a = (math.sin(dlat / 2)) ** 2 + math.cos(lat1) * math.cos(lat2) * (math.sin(dlon / 2)) ** 2
    distancia = 2 * R * math.asin(math.sqrt(a))
    distancia = distancia * 1000
    x = math.sin(dlon) * math.cos(lat2)
    y = (math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon))
    track_angle = math.atan2(x, y)
    track_angle = math.degrees(track_angle)
    compass_bearing = (track_angle + 360) % 360
    posX = distancia * math.cos(track_angle)
    posY = distancia * math.sin(track_angle)
    return compass_bearing


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




def kf_predict(X, P, A, Q, B, U):
    X = np.dot(A, X) + np.dot(B, U)
    P = np.dot(np.dot(A, P), A.T) + Q
    return X, P


def kf_update(X, P, Y, H, R):
    V = Y - np.dot(H, X)
    S = R + np.dot(np.dot(H, P), H.T)
    K = np.dot(np.dot(P, H.T), inv(S))
    X = X + np.dot(K, V)
    P = P - np.dot(np.dot(K, S), K.T)
    return X, P


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

GRAFICAS()
