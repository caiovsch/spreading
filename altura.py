import numpy as np
import timeit
from matplotlib import pyplot as plt
import funcoes

''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
''' PARAMETROS DE ENTRADA '''

''' PARAMETROS FISICOS '''
ESPESSURA_PRE_MOLHADA = 1e-6
RAIO_INICIAL = 1

''' PARAMETROS NUMERICOS '''
N_R = 400  # numero de divisoes em r
N_Z = 400  # numero de divisoes em z
DELTA_T = 0.05
R_MAX = 3.2
Z_MAX = 1.0
T_FINAL = 5000

''' PARAMETROS POS-PROCESSAMENTO '''
NUMERO_FRAMES_TEMPO = 200
CAMINHO = "./resultados/"

''' GERAR MALHA E INICIALIZAR SOLUCAO '''
dr = R_MAX / N_R
print(f'dr={dr}')
dz = Z_MAX / N_Z
r = np.arange(0, R_MAX + dr, dr)
z = np.arange(0, Z_MAX + dz, dz)
t = np.arange(0, T_FINAL + DELTA_T, DELTA_T)

h0 = funcoes.esferico(RAIO_INICIAL, ESPESSURA_PRE_MOLHADA, r, N_R)
# h0 = funcoes.sigmoide(r, ESPESSURA_PRE_MOLHADA)
# altura = 2/3*RAIO_INICIAL
# h0 = funcoes.esferico(RAIO_INICIAL, altura, ESPESSURA_PRE_MOLHADA, r, N_R)

''' PARAMETRO AUXILIAR '''
lamda = DELTA_T/12/dr**2
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

''' AVANCO NO TEMPO '''
start = timeit.default_timer()  # cronometro para avaliar custo computacional

h = np.copy(h0)
for k in range(1, len(t)):
    h = funcoes.calcular_ck(h, r, N_R, dr, lamda)
    # h = funcoes.calcular_imp(h, r, N_R, dr, lamda)
    # h = funcoes.calcular_exp(h, r, N_R, dr, lamda)

    ''' SALVAR ALGUNS INSTANTES DE TEMPO '''
    if k % intervalo_captura == 0:
        frame_atual += 1

        frames_h[frame_atual] = h
        frames_raio[frame_atual] = funcoes.raio_espalhamento(r, h, ESPESSURA_PRE_MOLHADA)
        frames_volume[frame_atual] = funcoes.calcular_volume(r, h, dr)

        print(f'tempo = {indices_tempo_selecionados[frame_atual] * DELTA_T}')

end = timeit.default_timer()
duracao = end - start
print(f'\nDuração de {duracao:.5f} s \n')

''' VERIFICACAO DO VOLUME DA GOTICULA '''
volume_inicial = funcoes.calcular_volume(r, h0, dr)
volume_final = funcoes.calcular_volume(r, h, dr)
diferenca_percentual = (volume_final-volume_inicial)/volume_inicial*100
print(f'Volume inicial = {volume_inicial}')
print(f'Volume final = {volume_final} ({diferenca_percentual} %)')


''''''''''''''''''''''''''''''''''''
''' SALVAR DADOS PARA O POS-PROCESAMENTO '''
np.savez(f'{CAMINHO}malha.npz', r=r, z=z, t=t, N_R=N_R, N_Z=N_Z)
np.savez(f'{CAMINHO}frames.npz',
         indices_tempo_selecionados=indices_tempo_selecionados,
         frames_h=frames_h,
         frames_raio=frames_raio,
         frames_volume=frames_volume
         )

''' PLOTAR PARA VERIFICACAO '''
plt.style.use(['science', 'notebook'])

plt.plot(r, h, 'b.-')
plt.axis('scaled')
plt.xlabel('$R$')
plt.ylabel('$Z$')
plt.xlim([0, R_MAX])
plt.ylim([-0.01, 1])

plt.show()
