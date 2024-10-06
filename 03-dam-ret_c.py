# -*- coding: utf-8 -*-
"""
Exemplo MECANIC - ELEMENTO TRIANGULAR CST
@author: Josinaldo Leandro de Souza
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# <<<< ETAPA 01. DADOS DE ENTRADA e PARAMETROS INICIAS // MALHA DO GiD >>>>

E = 2446400000 #24.464.000.000      # módulo de elasticidade MPa
v = 0.2        # coeficiente de Poisson
t = 2          # espessura m
gama_agua = 9810 # peso especifico da agua N/m3
gama_conc = 22100 # peso especifico da agua N/m3
dmp = 0.25


# Relação constitutiva - estado plano de tensões
D = (E / (1 - v**2)) * np.array([[1, v, 0], [v, 1, 0], [0, 0, (1 - v) / 2]])

# Indicacao da malha CST

# Carregar os dados de cada aba do arquivo Excel
arquivo_excel = 'D:/Python/01 - Script/Mecanic 2D/03-dam-ret.xlsx'
nome_aba = ['coord','conec']

# Coordenadas
coord = pd.read_excel(arquivo_excel,sheet_name=nome_aba[0],usecols=[0,1,2], skiprows=0,nrows=6352,names=['NO','X','Y'])

# Conectividade
conec =pd.read_excel(arquivo_excel,sheet_name=nome_aba[1],usecols=[0,1,2,3], skiprows=0,nrows=12325,names=['Elem','V1','V2','V3'])

# <<<< ETAPA 02. LOOP PARA CONSTRUÇÃO DA MATRIZ DOS ELEMENTOS E MATRIZ GLOBAL >>>>

nn = coord.shape[0]
nel = conec.shape[0]
ngl = nn * 2

KG = np.zeros((ngl, ngl))  # Matriz global de rigidez
B_colec = np.zeros((3, 6, nel))  # Coleção de matrizes B dos elementos

for i in range(nel):
    # Nós correspondentes ao elemento
    no1 = conec.iloc[i, 1]
    no2 = conec.iloc[i, 2]
    no3 = conec.iloc[i, 3]
    
    # Coordenadas dos nós
    x1, y1 = coord.iloc[(no1-1),[1,2]]
    x2, y2 = coord.iloc[(no2-1),[1,2]]
    x3, y3 = coord.iloc[(no3-1),[1,2]]
    
    # Área do triângulo (A)
    xx = np.array([x1, x2, x3])
    yy = np.array([y1, y2, y3])
    A = 0.5 * np.abs(np.dot(xx, np.roll(yy, 1)) - np.dot(yy, np.roll(xx, 1)))
    
    # Matriz [B]
    B = (1 / (2 * A)) * np.array([
        [y2-y3, 0, y3-y1, 0, y1-y2, 0],
        [0, x3-x2, 0, x1-x3, 0, x2-x1],
        [x3-x2, y2-y3, x1-x3, y3-y1, x2-x1, y1-y2]
    ])
    
    B_colec[:, :, i] = B  # Armazena a coleção de matrizes B dos elementos
    
    # Matriz de rigidez do elemento 
    Ke = B.T @ D @ B * A * t
    
    # Graus de liberdade gl's (correspondência na global)
    gl = np.array([2*no1-1, 2*no1, 2*no2-1, 2*no2, 2*no3-1, 2*no3]) - 1
    
    # Loop com introdução de [Ke] em [KG]
    for j in range(6):
        for k in range(6):
            KG[gl[j], gl[k]] += Ke[j, k]

# <<<< ETAPA 03 - Aplicando as Condições de ContornoS >>>>

restric = np.array([1,2,6,11,18,25,35,48,63,78,95,113,132,156,183,207,235,265,296,
                    329,369,405,444,486,528,572,618,666,715,767,818,872,928,985,
                    1046,1108,1173,1240,1309,1378,1447,1523,1587,1649,1708,1764,
                    1821,1881,1938,1993,2046,2099,2152,2205,2262,2314,2368,2418,
                    2471,2524,2586,2648,2709,2772,2835,2902,2970,3044,3118,3193,
                    3270,3348,3425,3511,3598,3686,3773,3862,3952,4048,4148,4241,
                    4334,4427,4521,4611,4703,4797,4887,4980,5070,5157,5240,5319,
                    5400,5476,5551,5622,5690,5750,5809,5868,5923,5975,6029,6081,
                    6130,6177,6216,])
restric.sort()  # Ordenando o vetor restric

# nos com pressoes conhecidas a montante)
pm = np.array([4036,4091,4137,4188,4239,4291,4345,4397,4451,4502,4559,4612,4670,
               4725,4783,4837,4896,4954,5013,5072,5131,5187,5242,5298,5354,5407,
               5459,5511,5563,5612,5661,5708,5752,5797,5842,5883,5921,5960,6001,
               6041,6080,]) # Acao do empuxo de agua a montante
pm.sort()  # Ordenando o vetor restric

# Acao da pressao da lamina de agua a montante
pm2 = np.array([4036,4119,4204,4283,4370,4453,4531,4614,4693,4774,4858,4938,5021,
                5100,5173,5251,5326,5394,5466,5530,5598,5660,5719,5774,5830,5884,
                5935,5984,6030,6074,6120,6164,6208,6249,6277,6300,6319,6331,6341,
                6348,6351])
pm2.sort() # Ordenando o veto

## Encontrar a maior coordenada em Y dos pontos com pressao conhecida
ponto = np.zeros((pm.shape[0], 2))

for i in range(pm.shape[0]):
    # Número do nó correspondente
    no1 = pm[i]
    
    # Coordenadas do nó (usando iloc para indexação por posição)
    x1 = coord.iloc[no1-1, 1]  # Coordenada X
    y1 = coord.iloc[no1-1, 2]  # Coordenada Y
    
    # Armazena as coordenadas x1 e y1 no vetor 'ponto'
    ponto[i, 0] = x1
    ponto[i, 1] = y1

# Encontre o maior valor de coordenada x e y
maior_x = max(ponto, key=lambda ponto: ponto[0])[0]
maior_y = max(ponto, key=lambda ponto: ponto[1])[1]


# Valores da força devido o Empuxo de agua
Femp = np.zeros(ngl) # Criacao do vetor 

# Valores das pressoes Montante
for i in range(pm.shape[0]): 
    
    pos1 = 2 * (pm[i] - 1) # posição da carga
    # Coordenadas dos nos
    xx1, yy1 = coord.iloc[pm[i] - 1, [1, 2]]
    Femp[pos1] = gama_agua * (maior_y - yy1)# carga em X

for i in range(pm2.shape[0]): 
    
    pos1a =  2 * (pm2[i] - 1) + 1# posição da carga
    Femp[pos1a] = gama_agua *maior_y # carga em Y

# colocar o peso da barragem

Fpp = np.zeros(ngl) ## Valores da força devido ao peso proprio

for i in range(nel):
    # Nós correspondentes ao elemento
    no1 = conec.iloc[i, 1]
    no2 = conec.iloc[i, 2]
    no3 = conec.iloc[i, 3]
    
    # Coordenadas dos nós
    x1, y1 = coord.iloc[(no1-1),[1,2]]
    x2, y2 = coord.iloc[(no2-1),[1,2]]
    x3, y3 = coord.iloc[(no3-1),[1,2]]
    
    # Área do triângulo (A)
    xx = np.array([x1, x2, x3])
    yy = np.array([y1, y2, y3])
    A = 0.5 * np.abs(np.dot(xx, np.roll(yy, 1)) - np.dot(yy, np.roll(xx, 1)))

    # Valor referente ao peso proprio sob acao da gravidade
    Pppe_value = -A * gama_conc * t/3
    
    # Vetor [Pfe] que será adicionado em Pf
    Pppe = np.array([0, Pppe_value, 0, Pppe_value, 0, Pppe_value])

    # Índices globais correspondentes aos nós do elemento
    gl = np.array([2*no1-1, 2*no1, 2*no2-1, 2*no2, 2*no3-1, 2*no3]) - 1
      
    # Loop com introdução de [Pfe] em [Pf]
    for j in range(6):
        Fpp[gl[j]] += Pppe[j]

# Vetor forças é a soma de todas forças externas
Ff = Fpp + Femp

# <<<< ETAPA 04. SOLUÇÃO DO SISTEMA DE EQUAÇÕES >>>>

KGR = KG.copy()  # Acumula a matriz global antes do número grande

# vetor de forca do problema
F = np.zeros(ngl)

# vetor forca modificado considerando condicoes nao-homogeneas do problema
Fmod = F - np.dot(KG, Ff)

# eliminar linhas e colunas com restições
#KGR = np.delete(KGR, restric-1, axis=0)
#KGR = np.delete(KGR, restric-1, axis=1)
#FGR = np.delete(Fmod, restric-1)


for i in range(len(restric)):
    pos1 = restric[i] * 2 - 2  # Restrição x
    pos2 = restric[i] * 2 - 1  # Restrição y
    
    KGR[pos1, pos1] *= 1e10  # Número grande
    KGR[pos2, pos2] *= 1e10  # Número grande


sol = np.linalg.solve(KGR, Ff)  # Solução do sistema de equações após restrições


# Vetor completo de solução (inicialmente nulo)
#sol2 = np.zeros((ngl, 1))

# Posições não nulas
#pos = np.arange(1, ngl + 1)  # Criar array de 1 a ngl
#pos = np.delete(pos, restric - 1)  # Remove as posições em 'elim' (indexação ajustada para 0)

# Insere os valores de deslocamento não-nulos no vetor completo
#for i in range(len(pos)):
    #Vetor de solução completo
 #   sol2[pos[i] - 1, 0] = sol[i]  # Ajuste para indexação 0

# Inicializa a matriz para armazenar os deslocamentos em x, y e z
sol_gid = np.zeros((nn, 3))

# Para cada nó, registramos os deslocamentos x e y em uma única linha
for i in range(nn):
    sol_gid[i, 0] = sol[2 * i]       # deslocamento x
    sol_gid[i, 1] = sol[2 * i + 1]   # deslocamento y

# Adiciona uma coluna com o número dos nós
output = np.column_stack((np.arange(1, nn + 1), sol_gid))

# <<<< ETAPA 05 / Cálculo de tensões em cada elemento triangular de três nós >>>>

sigma_colec = np.zeros((3, 1, nel))

for i in range(nel):
    # Nós correspondentes ao elemento
    no1 = conec.iloc[i, 1]
    no2 = conec.iloc[i, 2]
    no3 = conec.iloc[i, 3]
    
    # Recupera deslocamentos associados aos nós do elemento
    pos = np.array([2*no1-2, 2*no1-1, 2*no2-2, 2*no2-1, 2*no3-2, 2*no3-1])
    d = sol[pos]
    
    # Tensões nos elementos (detalhada no capítulo 6 - Logan)
    sigma_colec[:, :, i] = (D @ B_colec[:, :, i] @ d).reshape(3, 1)

# Inicializa a matriz para armazenar as tensões dos elementos
tensao_gid = np.zeros((nel, 3))

# Loop para preencher tensao_gid com as tensões x, y, xy
for i in range(nel):
    tensao_gid[i, 0] = sigma_colec[0, 0, i]  # sigma_x
    tensao_gid[i, 1] = sigma_colec[1, 0, i]  # sigma_y
    tensao_gid[i, 2] = sigma_colec[2, 0, i]  # tau_xy

# Gera uma coluna com o número dos elementos
element_numbers = np.arange(1, nel + 1).reshape(nel, 1)

# Concatena os números dos elementos com as tensões
output2 = np.hstack((element_numbers, tensao_gid))


# Exportando o vetor Pf para um arquivo de texto formatado aceitavel pelo GID
output_file = "03-dam-ret_c.res"

# Escreve o cabeçalho do arquivo para os deslocamentos
with open(output_file, "w") as f:
    f.write("GiD Post Results File 1.0\n\n")
    f.write('Result "Displacements" "Load Analysis"  1  Vector OnNodes\n')
    f.write('ComponentNames "X-Displ", "Y-Displ", "Z-Displ"\n')
    f.write("Values\n")

    # Escreve os valores em formato exponencial
    np.savetxt(f, output, delimiter='\t', fmt=['%d', '%.6e', '%.6e', '%d'])
    
    f.write("End Values\n")

    # Escrever o cabeçalho para as tensoes
    f.write('GaussPoints "Board elements" ElemType Triangle "board"\n')
    f.write('  Number Of Gauss Points: 1\n')
    f.write('  Natural Coordinates: internal\n')
    f.write('end gausspoints\n\n')
    f.write('Result "Tensao" "Load Analysis" 1 Vector OnGaussPoints "Board elements"\n')
    f.write('ComponentNames "Sigma_x", "Sigma_y", "Tau_xy"\n')
    f.write('Values\n')
    
    # Escrever os valores de tensão para cada elemento
    for i in range(nel):
        f.write(f'{i+1}\t{tensao_gid[i, 0]:.6e}\t{tensao_gid[i, 1]:.6e}\t{tensao_gid[i, 2]:.6e}\n')
    
    # Escrever o rodapé
    f.write('End Values\n')


