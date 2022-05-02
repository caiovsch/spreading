import numpy as np
from matplotlib import pyplot as plt
from numba import njit

''''''''''''''''''''''''''''''''''''''''''
''' ABRIR OS DADOS DO PROCESSAMENTO '''

caminho = './resultados/'
perfil = 'esferico'

malha = np.load(f'{caminho}malha.npz')
r = malha['r']
z = malha['z']
t = malha['t']
N_R = malha['N_R']
N_Z = malha['N_Z']

frames = np.load(f'{caminho}frames.npz')
frames_h = frames['frames_h']
frames_raio = frames['frames_raio']
indices_tempo_selecionados = frames['indices_tempo_selecionados']

dr = r[1] - r[0]
dz = z[1] - z[0]
dt = t[1] - t[0]


@njit
def calcular_delh_delr(h, n, dr):
    delh_delr = np.zeros_like(h)
    for i in range(n+1):
        if i == 0:
            delh_delr[i] = 0
        if 0 < i < n:
            delh_delr[i] = (h[i + 1] - h[i - 1])/2/dr
        if i == n:
            delh_delr[i] = 0

    return delh_delr


@njit
def calcular_del2h_delr2(delh_delr, n, dr):
    del2h_delr2 = np.zeros_like(delh_delr)
    for i in range(n+1):
        if i == 0:
            del2h_delr2[i] = (delh_delr[i+1] - delh_delr[i])/dr
        if 0 < i < n:
            del2h_delr2[i] = (delh_delr[i + 1] - delh_delr[i - 1])/2/dr
        if i == n:
            del2h_delr2[i] = (delh_delr[i] - delh_delr[i-1])/dr

    return del2h_delr2


def calcular_ur(h, delh_delr, z, n, p):
    ur = np.zeros((n+1, p+1))
    for i in range(n+1):
        for j in range(p+1):

            if z[j] < h[i]:
                ur[i, j] = - delh_delr[i] * z[j]/2 * (2*h[i] - z[j])
            else:
                ur[i, j] = float('NaN')

    return ur


def calcular_uz(h, delh_delr, del2h_delr2, r, z, n, p):
    uz = np.zeros((n + 1, p + 1))
    for i in range(n+1):
        for j in range(p+1):

            if z[j] < h[i]:
                if r[i] == 0:
                    uz[i, j] = z[j]**2/2 * (del2h_delr2[i]*(h[i]-z[j]/3) + delh_delr[i]**2)
                else:
                    uz[i, j] = z[j]**2/2 * (1/r[i]*delh_delr[i]*(h[i]-z[j]/3) + del2h_delr2[i]*(h[i]-z[j]/3) + delh_delr[i]**2)
            else:
                uz[i, j] = float('NaN')
    return uz


rr, zz = np.meshgrid(r, z)

frames_ur = np.zeros((len(frames_h), N_R + 1, N_Z + 1))
frames_uz = np.zeros_like(frames_ur)

for i in range(len(frames_ur)):
    print(i)
    delh_delr = calcular_delh_delr(frames_h[i], N_R, dr)
    del2h_delr2 = calcular_del2h_delr2(delh_delr, N_R, dr)
    frames_ur[i] = calcular_ur(frames_h[i], delh_delr, z, N_R, N_Z)
    frames_uz[i] = calcular_uz(frames_h[i], delh_delr, del2h_delr2, r, z, N_R, N_Z)

''''''''''''''''''''''''''''''''''''
''' SALVAR DADOS PARA O POS-PROCESAMENTO '''
caminho = "./resultados/"

np.savez(f'{caminho}vel.npz',
         frames_ur=frames_ur,
         frames_uz=frames_uz
         )
