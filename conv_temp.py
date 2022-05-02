import numpy as np
from matplotlib import pyplot as plt

CAMINHO = './resultados/'

ESPESSURA_PRE_MOLHADA = 1e-6
RAIO_INICIAL = 1
N_R = 100
R_MAX = 3.0
T_FINAL = 1000

DELTA_T = np.array([
    20,
    10,
    5,
    2,
    1,
    0.5,
])

ALTURA_CK = np.array([
    0.1624199148948138,
    0.15392702865179173,
    0.15093768509556535,
    0.1498409393219877,
    0.1496669963374607,
    0.1496378206517274
])

ALTURA_IMP = np.array([
    0.16404011964867377,
    0.15473045219651127,
    0.1513889435345765,
    0.15005677831480863,
    0.14978835237753071,
    0.1497022244906474
])

ALTURA_EXATA_CK = 0.14963023543195797
ALTURA_EXATA_IMP = 0.1496309975839562

erro_abs_ck = np.abs(ALTURA_CK - ALTURA_EXATA_CK)
erro_abs_imp = np.abs(ALTURA_IMP - ALTURA_EXATA_IMP)

erro_rel_ck = erro_abs_ck/ALTURA_EXATA_CK
erro_rel_imp = erro_abs_imp/ALTURA_EXATA_IMP

''' GRAFICO '''
plt.style.use(['science', 'notebook', 'grid'])
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

fig = plt.subplots(figsize=(7, 3.75))
plt.plot(DELTA_T, erro_rel_ck, 'b*-', label='Crank-Nicolson')
plt.plot(DELTA_T, erro_rel_imp, 'rd-', label='Implícito')

plt.yscale('log')
plt.xscale('log')
plt.xlabel(r'$\Delta T$', fontsize=11)
plt.ylabel('erro relativo', fontsize=11)
plt.xticks(fontsize=11), plt.yticks(fontsize=11)
plt.legend(fontsize=11)
plt.grid(True, which="both", axis='x')

plt.savefig(f'{CAMINHO}convergencia.png', format='png', dpi=700, bbox_inches='tight')
plt.show()

ordem_ck = (np.log(erro_abs_ck[-1]/erro_abs_ck[0])) / (np.log(DELTA_T[-1]/DELTA_T[0]))
ordem_imp = (np.log(erro_abs_imp[-1]/erro_abs_imp[0])) / (np.log(DELTA_T[-1]/DELTA_T[0]))
print(ordem_ck)
print(ordem_imp)
