import numpy as np
from numba import njit

''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
''' DIFERENCAS FINITAS '''


@njit
def calcular_exp(h, r, n, dr, lamda):
    h_novo = np.zeros_like(h)

    h_novo[0] = h[0] + 2*lamda * (h[1]**4 - h[0]**4)
    for i in range(1, n):
        h_novo[i] = h[i] + lamda * (h[i+1]**4 - 2*h[i]**4 + h[i-1]**4 + dr/2/r[i] * (h[i+1]**4 - h[i-1]**4))
    h_novo[n] = h[n] + 2*lamda * (h[n-1]**4 - h[n]**4)

    return h_novo


@njit
def calcular_imp(h, r, n, dr, lamda):
    omega = 1.0
    h_novo = np.copy(h)

    while True:
        h_iter = np.copy(h_novo)

        res = 1/(1+8*lamda*h[0]**3) * (h[0] + 2*lamda*(4*h[1]**3*h_novo[1]-3*h[1]**4) + 6*lamda*h[0]**4 - (1+8*lamda*h[0]**3)*h_novo[0])
        h_novo[0] = h_novo[0] + omega * res
        for i in range(1, n):
            res = 1/(1+8*lamda*h[i]**3) * (h[i] + lamda * ((4*h[i-1]**3*h_novo[i-1]-3*h[i-1]**4)*(1-dr/2/r[i]) + 6*h[i]**4 + (4*h[i+1]**3*h_novo[i+1]-3*h[i+1]**4)*(1+dr/2/r[i])) - (1+8*lamda*h[i]**3)*h_novo[i])
            h_novo[i] = h_novo[i] + omega*res
        res = 1/(1+8*lamda*h[n]**3) * (h[n] + 2*lamda*(4*h[n-1]**3*h_novo[n-1]-3*h[n-1]**4) + 6*lamda*h[n]**4 - (1+8*lamda*h[n]**3)*h_novo[n])
        h_novo[n] = h_novo[n] + omega * res

        criterio_parada = np.max(np.abs(h_novo - h_iter))
        if criterio_parada < 1e-7:
            break

    return h_novo


@njit
def calcular_ck(h, r, n, dr, lamda):
    omega = 1.0
    h_novo = np.copy(h)

    while True:
        h_iter = np.copy(h_novo)

        res = 1/(1+4*lamda*h[0]**3) * (h[0] + lamda*(4*h[1]**3*h_novo[1]-3*h[1]**4) + 3*lamda*h[0]**4 + lamda*(h[1]**4-h[0]**4) - (1+4*lamda*h[0]**3)*h_novo[0])
        h_novo[0] = h_novo[0] + omega * res
        for i in range(1, n):
            res = 1/(1+4*lamda*h[i]**3) * (h[i] + lamda/2 * ((4*h[i-1]**3*h_novo[i-1]-3*h[i-1]**4)*(1-dr/2/r[i]) + 6*h[i]**4 + (4*h[i+1]**3*h_novo[i+1]-3*h[i+1]**4)*(1+dr/2/r[i])) + lamda/2 * (h[i+1]**4 - 2*h[i]**4 + h[i-1]**4 + dr/2/r[i] * (h[i+1]**4 - h[i-1]**4)) - (1+4*lamda*h[i]**3)*h_novo[i])
            h_novo[i] = h_novo[i] + omega*res
        res = 1/(1+4*lamda*h[n]**3) * (h[n] + lamda*(4*h[n-1]**3*h_novo[n-1]-3*h[n-1]**4) + 3*lamda*h[n]**4 + lamda*(h[n-1]**4-h[n]**4) - (1+4*lamda*h[n]**3)*h_novo[n])
        h_novo[n] = h_novo[n] + omega * res

        criterio_parada = np.max(np.abs(h_novo - h_iter))
        if criterio_parada < 1e-7:
            break

    return h_novo


@njit
def raio_espalhamento(r, h, espessura_pre_molhada):
    raio = -1
    for i in range(0, len(r)):
        if h[i] < 1.01 * espessura_pre_molhada:
            raio = r[i-1]
            break
    return raio


@njit
def calcular_volume(r, h, dr):
    volume = 0
    for i in range(len(r)-1):
        volume += 2*np.pi * (r[i] + r[i+1])/2 * dr*(h[i]+h[i+1])/2
    return volume


''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
''' PERFIS INICIAIS '''


def sigmoide(r, espessura_pre_molhada):
    h0 = 1/(1+np.exp(5*r-5)) + espessura_pre_molhada
    return h0


def esferico(raio, espessura_pre_molhada, r, n):
    h0 = np.zeros_like(r)
    for i in range(n + 1):
        if r[i] <= raio:
            h0[i] = np.sqrt(raio**2-r[i]**2) + espessura_pre_molhada
        else:
            h0[i] = espessura_pre_molhada
    return h0


def cilindrico(raio, altura, espessura_pre_molhada, r, n):
    h0 = np.zeros_like(r)
    for i in range(n + 1):
        if r[i] <= raio:
            h0[i] = altura + espessura_pre_molhada
        else:
            h0[i] = espessura_pre_molhada
    return h0
