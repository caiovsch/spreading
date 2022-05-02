import numpy as np
from matplotlib import pyplot as plt
from numba import njit

TEMPO_PLOT = np.array([0, 100, 1000, 8000])
FRAME = 0

''''''''''''''''''''''''''''''''''''''''''
''' ABRIR OS DADOS DO PROCESSAMENTO '''

caminho = './resultados/'
perfil = 'esferico'

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
plt.style.use(['science', 'notebook'])
plt.rc('text', usetex=True)
plt.rc('font', family='serif')


def config_basicas_plot():
    plt.axis('scaled')
    plt.xlabel('$R$', fontsize=11)
    plt.ylabel('$Z$', fontsize=11)
    plt.xticks(fontsize=11), plt.yticks(fontsize=11)


# ''' DERIVADAS DE H '''
# plt.plot(r, delh_delr, 'b-*', label='1a derivada')
# plt.plot(r, del2h_delr2, 'r-d', label='2aderivada')
#
# plt.xlabel('$R$', fontsize=11)
# plt.xticks(fontsize=11), plt.yticks(fontsize=11)
# plt.legend(fontsize=11)
#
# plt.show()
#
# ''' COMPONENTE R DA VELOCIDADE (UR) '''
# fig4 = plt.subplots(figsize=(7.5, 3.75))
# plt.contourf(rr, zz, np.transpose(ur), levels=20, cmap=plt.cm.jet)
# plt.plot(r, frames_h[FRAME], 'k-', linewidth=1.25)
#
# colorbar = plt.colorbar()
# colorbar.ax.tick_params(labelsize=11)
#
# config_basicas_plot()
#
# plt.show()
#
# ''' COMPONENTE Z DA VELOCIDADE (UZ) '''
# fig5 = plt.subplots(figsize=(7.5, 3.75))
# rr, zz = np.meshgrid(r, z)
# plt.contourf(rr, zz, np.transpose(uz), levels=20, cmap=plt.cm.jet)
# plt.plot(r, frames_h[FRAME], 'k-', linewidth=1.25)
#
# colorbar = plt.colorbar()
# colorbar.ax.tick_params(labelsize=11)
#
# config_basicas_plot()
#
# plt.show()
#
''' STREAMPLOT '''
indice_r_stream = [0, 1, 5, 10, 20, 30, 40, 50, 60, 80]
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
plt.streamplot(rr, zz, np.transpose(ur), np.transpose(uz), start_points=stream_points, density=8,
               linewidth=1.25, color='blue')
plt.plot(r, frames_h[FRAME], 'k-', linewidth=1.3)

plt.xlim([0, 1.5])
plt.ylim([-0.01, 1.01])

config_basicas_plot()
plt.show()

''' PAINEL '''


def painel(frames_ur, r, z, k_plot, dt, caminho, nome_arq, titulo):

    plt.clf()
    plt.subplots(figsize=(7, 3))

    rr, zz = np.meshgrid(r, z)
    i = 1

    for k in k_plot:
        indice = np.where(indices_tempo_selecionados == k)[0][0]

        plt.subplot(2, 2, i)
        plt.contourf(rr, zz, 1e5*np.transpose(frames_ur[indice]), levels=20, cmap=plt.cm.jet)
        plt.title(f'$T = {k * dt:.0f}$', fontsize=11)
        plt.xlabel('$R$', fontsize=11)
        plt.ylabel('$Z$', fontsize=11)
        plt.axis('scaled')
        plt.xticks(fontsize=11), plt.yticks(fontsize=11)

        plt.plot(r, frames_h[indice], 'k-', linewidth=0.6)

        if i == 2 or i == 4:
            plt.tick_params(
                axis='y',  # seleciona o eixo y
                labelleft=False)  # retira os labels
            plt.ylabel(None)

        if i <= 2:
            plt.tick_params(
                axis='x',
                labelbottom=False)
            plt.xlabel(None)

        i += 1

    plt.subplots_adjust(wspace=0.1, hspace=-0.1)

    # ajustando a barra de cores
    plt.subplots_adjust(right=0.88)
    cbar_ax = plt.axes([0.9, 0.2, 0.02, 0.6])
    cbar = plt.colorbar(cax=cbar_ax)
    cbar.set_label(titulo, fontsize=11)
    cbar.ax.tick_params(labelsize=8)

    plt.savefig(f'{caminho}{nome_arq}.png', format='png', dpi=700, bbox_inches='tight')

    plt.show()

    return plt


# k_plot = TEMPO_PLOT / dt
# painel(frames_ur, r, z, k_plot, dt, caminho, 'frames_ur', '$10^5 U_R$')
# painel(frames_uz, r, z, k_plot, dt, caminho, 'frames_uz', '$10^5 U_Z$')






# ''' TESTE '''
# def colorir(n, p, r, z, h, rr, zz):
#     cores = np.zeros((n+1, p+1))
#     indice = np.zeros(n+1)
#     for i in range(0, n+1):
#         # print(i)
#         for j in range(0, p+1):
#             # print(z[j] >= h[i])
#             if z[j] >= h[i]:
#                 indice[i] = int(j)
#                 cores[i, j] = 1
#                 break
#             if j == p:
#                 indice[i] = int(p)
#                 cores[i, p] = 1
#                 break
#
#     plt.pcolormesh(rr, zz, np.transpose(cores), cmap=plt.cm.jet)
#     plt.show()
#
#     plt.plot(indice)
#     plt.show()
#
#     return indice
#
#
# indice = colorir(N_R, N_Z, r, z, frames_h[FRAME], rr, zz)
# print(indice[0])
#
#
# ''' FUNCAO DE CORRENTE '''
# @njit
# def calcular_funcao_corrente(r, z, ur, uz, n, p, dr, dz, tol, h, indice):
#     psi = np.zeros((n+1, p+1))
#
#     lamda = -(1/dr**2+2/dz**2)
#     erro = 100
#     while erro > tol:
#         res_max = 0
#         for i in range(1, n-1):
#             for j in range(1, int(indice[i])):
#                 # print('oi')
#                 res = (ur[i, j+1] - ur[i, j-1])/2/dz \
#                       - (uz[i+1, j] - uz[i-1, j])/2/dr - \
#                       (1/r[i]*(psi[i+1, j] - psi[i-1, j])/2/dr +
#                        (psi[i+1, j] - 2*psi[i, j] + psi[i-1, j])/dr**2 +
#                        (psi[i, j+1] - 2*psi[i, j] + psi[i, j-1])/dz**2)
#                 res = res/lamda
#                 psi[i, j] = psi[i, j] + res
#                 if np.abs(res) > res_max:
#                     res_max = np.abs(res)
#         erro = res_max
#         print(erro)
#     # print(np.nanmax(psi))
#     # print(np.nanmin(psi))
#     return psi
#
# #
# # for i in range(0, len(indice)):
# #     indice[i] = int(indice[i])
#
# psi = calcular_funcao_corrente(r, z, ur, uz, N_R, N_Z, dr, dt, 1e-7, frames_h[FRAME], indice)
# plt.contour(rr, zz, np.transpose(psi))
# plt.colorbar()
#
# print(psi[20, 20])
#
# plt.show()


