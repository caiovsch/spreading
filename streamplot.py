import numpy as np
from matplotlib import pyplot as plt
from numba import njit

TEMPO_PLOT = np.array([0, 100, 1000, 8000])
FRAME = 20

''''''''''''''''''''''''''''''''''''''''''
''' ABRIR OS DADOS DO PROCESSAMENTO '''

caminho = './resultados/'

malha = np.load(f'{caminho}malha.npz')
r = malha['r']
z = malha['z']
print(len(r))
print(len(z))
t = malha['t']
N_R = malha['N_R']
N_Z = malha['N_Z']

frames = np.load(f'{caminho}frames.npz')
frames_h = frames['frames_h']
frames_raio = frames['frames_raio']
indices_tempo_selecionados = frames['indices_tempo_selecionados']

vel = np.load(f'{caminho}vel.npz')
frames_ur = vel['frames_ur']
frames_uz = vel['frames_uz']

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
            delh_delr[i] = (h[i + 1] - h[i]) / dr
            # delh_delr[i] = (h[i + 1] - h[i - 1])/2/dr
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

delh_delr = calcular_delh_delr(frames_h[FRAME], N_R, dr)
del2h_delr2 = calcular_del2h_delr2(delh_delr, N_R, dr)
ur = calcular_ur(frames_h[FRAME], delh_delr, z, N_R, N_Z)
uz = calcular_uz(frames_h[FRAME], delh_delr, del2h_delr2, r, z, N_R, N_Z)

''' GRAFICOS '''
plt.style.use(['science', 'notebook', 'grid'])
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

''' STREAMPLOT '''
fig = plt.subplots(figsize=(7, 3.75))
# indice_r_stream = [0, 1, 5, 10, 20, 35, 50, 70, 90, 110]  # para frame=0
indice_r_stream = [0, 2, 5, 10, 20, 35, 50, 70, 90, 110, 135, 160, 190]
r_stream = []
h_stream = []
for i in indice_r_stream:
    h = frames_h[FRAME]
    r_stream.append(dr*i)
    h_stream.append(0.99*h[i])
print(r_stream)
print(h_stream)
# r_stream = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
stream_points = np.column_stack([r_stream, h_stream])
plt.streamplot(rr, zz, np.transpose(ur), np.transpose(uz), start_points=stream_points, density=20,
               linewidth=1.25, color='blue')
plt.plot(r, frames_h[FRAME], 'k-', linewidth=1.5)


plt.axis('scaled')
plt.xlabel('$R$', fontsize=11)
plt.ylabel('$Z$', fontsize=11)
plt.xticks(fontsize=11), plt.yticks(fontsize=11)

plt.xlim([-0.02, 1.5])
plt.ylim([-0.01, 1.1])

plt.savefig(f'{caminho}streamline10.png', format='png', dpi=700, bbox_inches='tight')
plt.show()

tempo = indices_tempo_selecionados[FRAME]*dt
print(tempo)