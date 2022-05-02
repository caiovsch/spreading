import numpy as np
from matplotlib import pyplot as plt

''''''''''''''''''''''''''''''''''''''''''
''' ABRIR OS DADOS DO PROCESSAMENTO '''

caminho = './resultados/'

malha = np.load(f'{caminho}malha.npz')
r = malha['r']
z = malha['z']
t = malha['t']
N_R = malha['N_R']
N_Z = malha['N_Z']

frames = np.load(f'{caminho}frames.npz')
frames_h = frames['frames_h']
frames_raio = frames['frames_raio']
frames_volume = frames['frames_volume']
indices_tempo_selecionados = frames['indices_tempo_selecionados']

dt = t[1] - t[0]


def config_basicas_plot():
    plt.axis('scaled')
    plt.xlabel('$R$', fontsize=11)
    plt.ylabel('$Z$', fontsize=11)
    plt.xticks(fontsize=11), plt.yticks(fontsize=11)


''' GRAFICOS '''
plt.style.use(['science', 'notebook', 'grid'])
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

''' VOLUME '''
fig4 = plt.subplots(figsize=(6, 3.75))
tempo = indices_tempo_selecionados*dt
plt.plot(tempo, frames_volume/frames_volume[0], 'r--')

plt.xlabel('$T$', fontsize=11)
plt.ylabel(r'$V/V_0$', fontsize=11)
plt.xticks(fontsize=11), plt.yticks(fontsize=11)
# plt.xscale('log')
# plt.yscale('log')
plt.savefig(f'{caminho}volume.png', format='png', dpi=700, bbox_inches='tight')

plt.show()

''' EVOLUCAO DA SUPERFICIE LIVRE '''

fig2 = plt.subplots(figsize=(7.25, 3.75))
''' ESCOLHENDO OS INSTANTES QUE SERAO PLOTADOS '''
# t_plot = np.array([0, 0.1, 0.2, 0.5, 1])
# t_plot = np.array([0, 1, 2, 5, 10])
t_plot = np.array([0, 1, 4, 15, 100])
# t_plot = np.array([0, 100, 200, 500, 1000])
k_plot = t_plot/dt
print(k_plot)
estilos = ["solid", "dotted", "dashed", "dashdot",  (0, (3, 1, 1, 1, 1, 1))]

cont = -1
for k in k_plot:
    cont += 1
    indice = np.where(indices_tempo_selecionados == k)[0][0]
    plt.plot(r, frames_h[indice], color='blue', linestyle=estilos[cont], label=f'$T={t_plot[cont]}$', linewidth=1.25)

config_basicas_plot()
plt.xlim([0, r[-1]])
plt.ylim([-0.01, 1.01])
plt.legend(fontsize=11)

plt.savefig(f'{caminho}cilindrico_evolucao.png', format='png', dpi=700, bbox_inches='tight')
plt.show()

''' EVOLUCAO MUITOS INSTANTES '''
fig3 = plt.subplots(figsize=(7.5, 3.75))

print(dt*indices_tempo_selecionados)
for k in range(0, len(indices_tempo_selecionados)):
    plt.plot(r, frames_h[k], color='blue', linewidth=1)

config_basicas_plot()
plt.xlim([0, r[-1]])
plt.ylim([-0.01, 1.01])

plt.savefig(f'{caminho}evolucao_muitos.png', format='png', dpi=700, bbox_inches='tight')
plt.show()

''' RAIO DE ESPALHAMENTO '''
g = 9.81
nu = 13.2e-4
volume = 1e-3
fator = (3*nu/g/volume**3)**(1/8)
print(f'teste={fator*0.078}')
fig3 = plt.subplots(figsize=(7.5, 3.75))
tempo = indices_tempo_selecionados*dt
plt.plot(tempo[1:], fator*frames_raio[1:], 'b*-')
plt.plot(tempo, fator*tempo**(1/8), 'g*-')
plt.yscale('log')
plt.xscale('log')

plt.show()
