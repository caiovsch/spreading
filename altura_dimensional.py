import numpy as np
import timeit
from matplotlib import pyplot as plt
from numba import njit
import funcoes

''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
''' PARAMETROS DE ENTRADA '''

''' PARAMETROS FISICOS NO SI '''
GRAVIDADE = 9.81
VISCOSIDADE_CINEMATICA = 13.2e-4
VOLUME = 0.933e-3
ESPESSURA_PRE_MOLHADA = 1e-8

''' PARAMETROS NUMERICOS '''
N_R = 200  # numero de divisoes em r
N_Z = 200  # numero de divisoes em z
DELTA_T = 0.025
R_MAX = 1.0
TEMPO_FINAL = 10000

''' PARAMETROS POS-PROCESSAMENTO '''
NUMERO_FRAMES_TEMPO = 200
TEMPO_PLOT_LOG = np.logspace(1, 6, 10)
CAMINHO = './resultados/'

''' GERAR MALHA E INICIALIZAR SOLUCAO '''
dr = R_MAX / N_R
r = np.arange(0, R_MAX + dr, dr)
t = np.arange(0, TEMPO_FINAL + DELTA_T, DELTA_T)

raio_inicial = (3/2/np.pi*VOLUME)**(1/3)
h0 = funcoes.esferico(raio_inicial, ESPESSURA_PRE_MOLHADA, r, N_R)
h0_max = np.max(h0)

z_max = 1.1*h0_max
dz = z_max / N_Z
z = np.arange(0, z_max + dz, dz)

''' PARAMETRO AUXILIAR '''
lamda = (GRAVIDADE/VISCOSIDADE_CINEMATICA) * DELTA_T/12/dr**2
print(f'lamda = {lamda}')

''' VARIAVEIS PARA SALVAR ALGUNS INSTANTES DE TEMPO '''
numero_passos_tempo = len(t) - 1
intervalo_captura = numero_passos_tempo // NUMERO_FRAMES_TEMPO
indices_tempo_selecionados = np.arange(0, len(t), intervalo_captura)

''' INICIALIZANDO VARIAVEIS '''
frames_h = np.zeros((NUMERO_FRAMES_TEMPO + 1, N_R + 1))
frames_raio = np.zeros(NUMERO_FRAMES_TEMPO + 1)
frames_volume = np.zeros_like(frames_raio)

frame_atual = 0
indices_tempo_selecionados[0] = 0
frames_h[0] = h0
frames_volume[0] = funcoes.calcular_volume(r, h0, dr)
frames_raio[0] = funcoes.raio_espalhamento(r, h0, ESPESSURA_PRE_MOLHADA)

''' SALVAR ALGUNS INSTANTES ESPACADOS LOGARITMAMENTE '''
tempo_log = np.zeros(10)
raio_log = np.zeros(10)
contador_log = 0

''' AVANCO NO TEMPO '''
start = timeit.default_timer()  # cronometro para avaliar custo computacional

h = np.copy(h0)
for k in range(1, len(t)):
    h = funcoes.calcular_ck(h, r, N_R, dr, lamda)

    ''' SALVAR ALGUNS INSTANTES DE TEMPO '''
    if k % intervalo_captura == 0:
        frame_atual += 1

        frames_h[frame_atual] = h
        frames_raio[frame_atual] = funcoes.raio_espalhamento(r, h, ESPESSURA_PRE_MOLHADA)
        frames_volume[frame_atual] = funcoes.calcular_volume(r, h, dr)

        print(f'tempo = {indices_tempo_selecionados[frame_atual] * DELTA_T}')

    ''' INSTANTES LOGARITMAMENTE ESPACADOS '''
    if contador_log < 10 and k*DELTA_T >= TEMPO_PLOT_LOG[contador_log]:
        tempo_log[contador_log] = k*DELTA_T
        raio_log[contador_log] = funcoes.raio_espalhamento(r, h, ESPESSURA_PRE_MOLHADA)
        contador_log += 1

end = timeit.default_timer()
duracao = end - start
print(f'\nDuração de {duracao:.5f} s \n')

''' VERIFICACAO DO VOLUME DA GOTICULA '''
volume_inicial = funcoes.calcular_volume(r, h0, dr)
volume_final = funcoes.calcular_volume(r, h, dr)
diferenca_percentual = (volume_final-volume_inicial)/volume_inicial*100
print(f'Volume inicial = {volume_inicial}')
print(f'Volume final = {volume_final} ({diferenca_percentual} %)')

''' PARAMETROS DOS GRAFICOS '''
plt.style.use(['science', 'notebook', 'grid'])
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
''' RAIO DE ESPALHAMENTO '''

''' DADOS HUPPERT (1982) '''
tempo_huppert = np.array([
    10, 51.012, 101.226, 200.868, 508.568, 1009.18, 2002.57, 4993.59, 10061.1, 19964.8, 50548,
    100305, 199041, 503942, 1e6,
])

raio_huppert = np.array([
    1.16428, 1.42108, 1.54549, 1.67199, 1.88637, 2.04077, 2.24283, 2.50399, 2.70896, 2.97717,
    3.32384, 3.59591, 3.93127, 4.36607, 4.79837
])

''' COMPARACAO COM A SIMULACAO '''
fig = plt.subplots(figsize=(7, 3.75))
tempo = indices_tempo_selecionados*DELTA_T
plt.plot(tempo_log, (3*VISCOSIDADE_CINEMATICA/GRAVIDADE/VOLUME**3)**(1/8) * raio_log,
         'k--', label='Trabalho Presente')
plt.plot(tempo_huppert, raio_huppert,
         marker='d',
         markerfacecolor='None',
         markeredgecolor='red',
         markeredgewidth=2,
         markersize=10,
         linestyle='None',
         label='Huppert (1982)')

plt.legend(fontsize=11)
plt.grid(True, which="both", axis='y')

plt.xlabel(r'$t$ \,\, (s)', fontsize=11)
plt.ylabel(r'$(3\nu/gV^3)^{1/8} \mathcal{R}$ \,\, ($\mathrm{s}^{1/8}$)', fontsize=11)
plt.yscale('log')
plt.xscale('log')
plt.xlim([9, 1.1e6])
plt.ylim([0.99, 10.01])
plt.xticks(fontsize=11), plt.yticks(fontsize=11)

plt.savefig(f'{CAMINHO}raio_huppert.png', format='png', dpi=700, bbox_inches='tight')
plt.show()

''' PLOTAR PARA VERIFICACAO '''

plt.plot(r, h, 'b.-')
plt.axis('scaled')
plt.xlabel('$R$')
plt.ylabel('$Z$')
plt.xlim([0, R_MAX])
plt.ylim([-0.01, 1.05*z_max])

plt.show()

''''''''''''''''''''''''''''''''''''
''' SALVAR DADOS PARA O POS-PROCESAMENTO '''
np.savez(f'{CAMINHO}malha_dim.npz', r=r, z=z, t=t, n=N_R, p=N_Z)
np.savez(f'{CAMINHO}frames_dim.npz',
         indices_tempo_selecionados=indices_tempo_selecionados,
         frames_h=frames_h,
         frames_raio=frames_raio,
         frames_volume=frames_volume,
         tempo_huppert=tempo_huppert,
         raio_huppert=raio_huppert
         )
