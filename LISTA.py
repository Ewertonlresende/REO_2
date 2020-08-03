'''
print('-='*50)
print('DISCIPLINA: VISÃO COMPUTACIONAL NO MELHORAMENTO DE PLANTAS')
print('SEGUNDA LISTA DE EXERÍCIO - REO2')
print('ALUNOS:')
print('CAROL - ')
print('EWERTON - ')
print('MARIANA - ')
print('THIAGO - ')
print('PROFESSOR: VINÍCIUS QUINTÃO CARNEIRO')
print('-='*50)
'''
#Pacotes Utilizados
import numpy as np
import cv2
from matplotlib import pyplot as plt

print('-='*50)
print('Exercício 01: Selecione uma imagem a ser utilizada no trabalho prático e realize os seguintes processos '
      'utilizando o pacote OPENCV do Python')
print('-='*50)
print('01.a) Apresente a imagem e as informações de número de linhas e colunas; número de canais e número total de pixels')
arquivo = 'antracnose.jpeg'            # carregar a imagem
imagem = cv2.imread(arquivo, 1)     # 0 - imagem binária / 1 - imagem colorida
nl,nc,canais = np.shape(imagem)
print('Tipo: ', imagem.dtype)       # é um inteiro de 8 bytes
print('Número de linhas = ' + str(nl))
print('Número de colunas =  ' + str(nc))
print('Número de canais = ' + str(canais))
print('Número de pixel = ' + str(nl*nc))

# Converter a imagem para RGB
img_rgb = cv2.cvtColor(imagem,cv2.COLOR_BGR2RGB)
plt.imshow(img_rgb) # é uma função que mostra a função
plt.show()


print('-='*50)
print('01.b) Faça um recorte da imagem para obter somente a área de interesse. Utilize esta imagem para a solução das '
      'próximas alternativas')
imagem_mod_01 = np.copy(img_rgb)                 # ao se fazer uma cópia da imagem original podemos trabalhar com esta sem modificar a original
imagem_recortada = imagem_mod_01[250:810,395:860] #[250:810-> representa o intervalo das linhas/ 395:860=> representa o intervalo das colunas
plt.imshow(imagem_recortada)
plt.xticks([])                                  #eliminar o eixo X
plt.yticks([])                                  #eliminar o eixo y
plt.title('Imagem Recortada')
plt.show()

#Salvar imagem
#Devido ao fato de o OPENCV abrir as imagens em BGR, nós as convertemos em RGB para melhor visualização.
#mas quando vamos salvar a imagem devemos converter novamente para BGR e consequentemente salvar. para que essa fique no formato RGB.
salvar_img_rec = cv2.cvtColor(imagem_recortada,cv2.COLOR_RGB2BGR)
cv2.imwrite('imagem_recortada.png',salvar_img_rec)


print('-='*50)
print('01.c) Converta a imagem colorida para uma de escala de cinza (intensidade) e a apresente utilizando os mapas de'
      ' cores “Escala de Cinza” e “JET”')
imagem_cinza = cv2.cvtColor(imagem_recortada,cv2.COLOR_BGR2GRAY)

plt.figure('Figura gray and JET')
plt.subplot(1,2,1)
plt.imshow(imagem_cinza, cmap="gray")      # mapa de cor "gray" delimita entre o preto e o branco, intesidade
plt.title("Escala em Cinza")
plt.xticks([])
plt.yticks([])
plt.colorbar(orientation='horizontal')  #gera uma barra na escala do cinza

plt.subplot(1,2,2)
plt.imshow(imagem_cinza, cmap="jet")
plt.title("JET")
plt.xticks([])
plt.yticks([])
plt.colorbar(orientation='horizontal')
plt.show()

#para salvar a figura "Figura gray and JET" eu cliquei em salvar depois que esta apareceu.

print('-='*50)
print('01.D) Apresente a imagem em escala de cinza e o seu respectivo histograma; Relacione o histograma e a imagem.')

histograma = cv2.calcHist([imagem_cinza],[0],None,[256],[0,256])

#Função (cv2.calcHist) calcula a frequência de cada pixel, para cada valor de 0 a 255.
# [imagem_cinza] -> temos que colocar o objeto dentro dos colchetes
# [0] -> canal que queremos acessar, pois é imagem com apenas 1 canal.
# None -> se há máscara ou não, máscara é uma região especifica dentro da imagem
# [256] -> quantos pontos vamos utilizar
# [0,256] -> intervalo que queremos trabalhar

plt.figure('Imagem cinza e histograma')
plt.subplot(1,2,1)
plt.imshow(imagem_cinza, cmap="gray")      # mapa de cor "gray" delimita entre o preto e o branco
plt.title('Imagem Cinza')

plt.subplot(1,2,2)
plt.plot(histograma, color='black')
plt.title("Histograma")
plt.xlabel("Valores de Pixels")
plt.ylabel("Número de Pixels")
plt.show()

print('-='*50)
print('01.E) Utilizando a imagem em escala de cinza (intensidade) realize a segmentação da imagem de modo a remover o '
      'fundo da imagem utilizando um limiar manual e o limiar obtido pela técnica de Otsu. Nesta questão apresente o '
      'histograma com marcação dos limiares utilizados, a imagem limiarizada (binarizada) e a imagem colorida final '
      'obtida da segmentação. Explique resultados.')

limiar_cinza = 155
(L, img_limiar) = cv2.threshold(imagem_cinza,limiar_cinza,255,cv2.THRESH_BINARY)
# LIMIAR É O LIMITE QUE VAMOS SELECIONAR.
# # L -> Valor do limiar (100)
# # img_limiar -> è a imagem binaria com os valores acima e abaixo do threshold
# # ###
# # imagem_cinza -> é a imagem que vamos trabalhar
# # limiar_cinza -> limiar_cinza = 140
# # 255 -> valor max
# # cv2.THRESH_BINARY -> imagem binária, acima ou abaixo.
(L, img_limiar_inv) = cv2.threshold(imagem_cinza,limiar_cinza,255,cv2.THRESH_BINARY_INV)
# Aqui fazemos a inversão dos valores abaixo e acima do limiar, ou seja, o fundo era branco em uma passou a ser preto.
# Histograma escala de cinza
hist_cinza = cv2.calcHist([imagem_cinza],[0], None, [256],[0,256])

img_segmentada = cv2.bitwise_and(imagem_recortada,imagem_recortada,mask=img_limiar_inv)
plt.imshow(img_segmentada)
plt.show()
########################################################################################################################
# Limiarização - Thresholding - OTSU

(L1, imagem_otsu) = cv2.threshold(imagem_cinza,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
# thresh = 0
# max val = 0
#help(cv2.threshold)

hist_cinza = cv2.calcHist([imagem_cinza],[0], None, [256],[0,256])

plt.figure('Thresholding_gray')
plt.subplot(2,2,1)
plt.imshow(img_limiar,cmap='gray')
plt.title('Binário - L: ' + str(limiar_cinza))

plt.subplot(2,2,2)
plt.imshow(img_limiar_inv,cmap='gray')
plt.title('Binário Invertido: L: ' + str(limiar_cinza))

plt.subplot(2,2,3)
plt.imshow(imagem_otsu,cmap='gray')
plt.title('Limiar OTSU: L: ' + str(L1))

plt.subplot(2,2,4)
plt.plot(hist_cinza,color = 'black')
plt.axvline(x=L1,color = 'r')
plt.title("Histograma - Cinza")
plt.xlim([0,256])
plt.xlabel("Valores Pixels")
plt.ylabel("Número de Pixels")
plt.show()


print('-='*50)
print('01.F) Apresente uma figura contento a imagem selecionada nos sistemas RGB, Lab, HSV e YCrCb.')


img_Lab = cv2.cvtColor(imagem_recortada,cv2.COLOR_BGR2Lab)
img_HSV = cv2.cvtColor(imagem_recortada,cv2.COLOR_BGR2HSV)
img_YCrCb = cv2.cvtColor(imagem_recortada,cv2.COLOR_BGR2YCR_CB)

plt.figure('Sistemas de cores')
plt.subplot(2,2,1)
plt.imshow(imagem_recortada)
plt.title('RGB')

plt.subplot(2,2,2)
plt.imshow(img_Lab)
plt.title('Lab')

plt.subplot(2,2,3)
plt.imshow(img_HSV)
plt.title('HSV')

plt.subplot(2,2,4)
plt.imshow(img_YCrCb)
plt.title("YCrCb")
#plt.show()

print('-='*50)
print('01.G) Apresente uma figura para cada um dos sistemas de cores (RGB, HSV, Lab e YCrCb) contendo a imagem de cada '
      'um dos canais e seus respectivos histogramas.')
# Partição dos canais - Usando a função Cv2.split (mas comovamos indicar cada matiz abaixo não seria necessário)r,g,b = cv2.split(imagem_recortada)
#este comando abixo sera usado na qustao h.
L,a,b = cv2.split(img_Lab)
H,S,V = cv2.split(img_HSV)
Y,Cr,Cb = cv2.split(img_YCrCb)

#### RGB ###
hist_r = cv2.calcHist([imagem_recortada],[0], None, [256],[0,256]) #[0] queremos acessa o canal 0.(RED)
hist_g = cv2.calcHist([imagem_recortada],[1], None, [256],[0,256]) #[1] queremos acessa o canal 1.(GREEN)
hist_b = cv2.calcHist([imagem_recortada],[2], None, [256],[0,256]) #[2] queremos acessa o canal 2.(BLUE)


plt.figure('RGB')
plt.subplot(2,3,1)
plt.imshow(imagem_recortada[:,:,0],cmap = 'gray')
plt.title('Segmentada - R')
plt.xticks([])
plt.yticks([])

plt.subplot(2,3,2)
plt.imshow(imagem_recortada[:,:,1],cmap = 'gray')
plt.title('Segmentada - G')
plt.xticks([])
plt.yticks([])

plt.subplot(2,3,3)
plt.imshow(imagem_recortada[:,:,2],cmap = 'gray')
plt.title('Segmentada - B')
plt.xticks([])
plt.yticks([])

plt.subplot(2,3,4)
plt.plot(hist_r,color = 'r')
plt.xlim([0,256])
plt.xlabel('Valores de Pixels')
plt.ylabel('Número de Pixels')

plt.subplot(2,3,5)
plt.plot(hist_r,color = 'g')
plt.xlim([0,256])
plt.xlabel('Valores de Pixels')
plt.ylabel('Número de Pixels')

plt.subplot(2,3,6)
plt.plot(hist_r,color = 'b')
plt.xlim([0,256])
plt.xlabel('Valores de Pixels')
plt.ylabel('Número de Pixels')
plt.show()


############### Lab ###################
hist_L = cv2.calcHist([img_Lab],[0],None,[256],[0,256])
hist_a = cv2.calcHist([img_Lab],[1],None,[256],[0,256])
hist_b = cv2.calcHist([img_Lab],[2],None,[256],[0,256])

plt.figure('Lab')
plt.subplot(2,3,1)
plt.imshow(img_Lab[:,:,0],cmap = 'gray')
plt.title('Segmentada - L')
plt.xticks([])
plt.yticks([])

plt.subplot(2,3,2)
plt.imshow(img_Lab[:,:,1],cmap = 'gray')
plt.title('Segmentada - a')
plt.xticks([])
plt.yticks([])

plt.subplot(2,3,3)
plt.imshow(img_Lab[:,:,2],cmap = 'gray')
plt.title('Segmentada - b')
plt.xticks([])
plt.yticks([])

plt.subplot(2,3,4)
plt.plot(hist_L, color = 'black')
plt.title('Histograma - L')
plt.xlim([0,256])
plt.xlabel('Valores de Pixels')
plt.ylabel('Número de Pixels')

plt.subplot(2,3,5)
plt.plot(hist_a, color = 'black')
plt.title('Histograma - a')
plt.xlim([0,256])
plt.xlabel('Valores de Pixels')
plt.ylabel('Número de Pixels')

plt.subplot(2,3,6)
plt.plot(hist_b, color = 'black')
plt.title('Histograma - b')
plt.xlim([0,256])
plt.xlabel('Valores de Pixels')
plt.ylabel('Número de Pixels')

########### HSV ##########
hist_H = cv2.calcHist([img_HSV],[0],None,[256],[0,256])
hist_S = cv2.calcHist([img_HSV],[1],None,[256],[0,256])
hist_V = cv2.calcHist([img_HSV],[2],None,[256],[0,256])

plt.figure('HSV')
plt.subplot(2,3,1)
plt.imshow(img_HSV[:,:,0],cmap = 'gray')
plt.title('Segmentada - H')
plt.xticks([])
plt.yticks([])

plt.subplot(2,3,2)
plt.imshow(img_HSV[:,:,1],cmap = 'gray')
plt.title('Segmentada - S')
plt.xticks([])
plt.yticks([])

plt.subplot(2,3,3)
plt.imshow(img_HSV[:,:,2],cmap = 'gray')
plt.title('Segmentada - V')
plt.xticks([])
plt.yticks([])

plt.subplot(2,3,4)
plt.plot(hist_H, color = 'black')
plt.title('Histograma - H')
plt.xlim([0,256])
plt.xlabel('Valores de Pixels')
plt.ylabel('Número de Pixels')

plt.subplot(2,3,5)
plt.plot(hist_S, color = 'black')
plt.title('Histograma - S')
plt.xlim([0,256])
plt.xlabel('Valores de Pixels')
plt.ylabel('Número de Pixels')

plt.subplot(2,3,6)
plt.plot(hist_V, color = 'black')
plt.title('Histograma - V')
plt.xlim([0,256])
plt.xlabel('Valores de Pixels')
plt.ylabel('Número de Pixels')



######## YCrCb ##########
hist_Y = cv2.calcHist([img_YCrCb],[0],None,[256],[0,256])
hist_CR = cv2.calcHist([img_YCrCb],[1],None,[256],[0,256])
hist_CB = cv2.calcHist([img_YCrCb],[2],None,[256],[0,256])

plt.figure('YCrCb')
plt.subplot(2,3,1)
plt.imshow(img_YCrCb[:,:,0],cmap = 'gray')
plt.title('Segmentada - Y')
plt.xticks([])
plt.yticks([])

plt.subplot(2,3,2)
plt.imshow(img_YCrCb[:,:,1],cmap = 'gray')
plt.title('Segmentada - Cr')
plt.xticks([])
plt.yticks([])

plt.subplot(2,3,3)
plt.imshow(img_YCrCb[:,:,2],cmap = 'gray')
plt.title('Segmentada - Cb')
plt.xticks([])
plt.yticks([])

plt.subplot(2,3,4)
plt.plot(hist_Y, color = 'black')
plt.title('Histograma - Y')
plt.xlim([0,256])
plt.xlabel('Valores de Pixels')
plt.ylabel('Número de Pixels')

plt.subplot(2,3,5)
plt.plot(hist_CR, color = 'black')
plt.title('Histograma - Cr')
plt.xlim([0,256])
plt.xlabel('Valores de Pixels')
plt.ylabel('Número de Pixels')

plt.subplot(2,3,6)
plt.plot(hist_CB, color = 'black')
plt.title('Histograma - Cb')
plt.xlim([0,256])
plt.xlabel('Valores de Pixels')
plt.ylabel('Número de Pixels')
plt.show()

print('-='*50)
print('01.H) h) Encontre o sistema de cor e o respectivo canal que propicie melhor segmentação da imagem de modo a '
      'remover o fundo da imagem utilizando limiar manual e limiar obtido pela técnica de Otsu. Nesta questão apresente'
      ' o histograma com marcação dos limiares utilizados, a imagem limiarizada (binarizada) e a imagem colorida final '
      'obtida da segmentação. Explique resultados e sua escolha pelo sistema de cor e canal utilizado na segmentação. '
      'Nesta questão apresente a imagem limiarizada (binarizada) e a imagem colorida final obtida da segmentação.')

## Threshhold - Manual ##
limiar_manual = 140
(L_m, img_limiar_manual) = cv2.threshold(S,limiar_manual,255,cv2.THRESH_BINARY)


## Threshhold - OTSU ##
(L_O, img_thresh) = cv2.threshold(S,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
#L -> é o limiar de forma automatica!!
# img_thresh -> é a mascara. essa imagem serve para

## Histograma escala de cinza ##
hist_S = cv2.calcHist([S],[0], None, [256],[0,256])
## Imagem com Mascara ##
imagem_seg = cv2.bitwise_and(imagem_recortada,imagem_recortada,mask=img_thresh)

###
font = {'family': 'serif',
        'color':  'red',
        'weight': 'normal',
        'size': 8,
        }

font1 = {'family': 'serif',
        'color':  'green',
        'weight': 'normal',
        'size': 8,
        }

## Apresentar figura ##
plt.figure('Questão_h')
plt.subplot(2,2,3)
plt.plot(hist_S,color = 'black')
plt.axvline(x=L_O,color = 'r')
plt.text(40, 40000, r'Limiar(OTSU) ', fontdict=font)
plt.axvline(x=L_m,color = 'g')
plt.text(150, 40000, r'Limiar(Manual) ', fontdict=font1)
plt.title("Histograma - S")
plt.xlim([0,256])
plt.xlabel("Valores Pixels")
plt.ylabel("Número de Pixels")


plt.subplot(2,2,4)
plt.imshow(imagem_seg)
plt.title('Segmentada - RGB')
plt.xticks([])
plt.yticks([])

plt.subplot(2,2,2)
plt.imshow(img_thresh,cmap = 'gray')
plt.title('Limiar - OTSU')
plt.xticks([])
plt.yticks([])

plt.subplot(2,2,1)
plt.title('Limiar - Manual')
plt.imshow(img_limiar_manual,'gray')
plt.xticks([])
plt.yticks([])
plt.show()

print('-='*50)
print('01.I)  Obtenha o histograma de cada um dos canais da imagem em RGB, utilizando como mascara a imagem limiarizada'
      ' (binarizada) da letra h. ')
hist_seg_r = cv2.calcHist([imagem_seg],[0],img_thresh,[256],[0,256])
hist_seg_g = cv2.calcHist([imagem_seg],[1],img_thresh,[256],[0,256])
hist_seg_b = cv2.calcHist([imagem_seg],[2],img_thresh,[256],[0,256])

# Apresentar figuras

plt.subplot(2,3,1)
plt.imshow(imagem_seg[:,:,0],cmap = 'gray')
plt.title('Segmentada - R')
plt.xticks([])
plt.yticks([])

plt.subplot(2,3,2)
plt.imshow(imagem_seg[:,:,1],cmap = 'gray')
plt.title('Segmentada - G')
plt.xticks([])
plt.yticks([])

plt.subplot(2,3,3)
plt.imshow(imagem_seg[:,:,2],cmap = 'gray')
plt.title('Segmentada - B')
plt.xticks([])
plt.yticks([])

plt.subplot(2,3,4)
plt.plot(hist_seg_r,color = 'r')
plt.title("Histograma - R")
plt.xlim([0,256])
plt.xlabel("Valores Pixels")
plt.ylabel("Número de Pixels")

plt.subplot(2,3,5)
plt.plot(hist_seg_g,color = 'g')
plt.title("Histograma - G")
plt.xlim([0,256])
plt.xlabel("Valores Pixels")
plt.ylabel("Número de Pixels")

plt.subplot(2,3,6)
plt.plot(hist_seg_b,color = 'b')
plt.title("Histograma - B")
plt.xlim([0,256])
plt.xlabel("Valores Pixels")
plt.ylabel("Número de Pixels")
plt.show()

###############trabalhando em imagem seg####
##neste caso estava tentando fazer a seguimentação para poder evidenciar as lesoes.
img_Lab_2 = cv2.cvtColor(imagem_seg,cv2.COLOR_BGR2Lab)
img_HSV_2 = cv2.cvtColor(imagem_seg,cv2.COLOR_BGR2HSV)
img_YCrCb_2 = cv2.cvtColor(imagem_seg,cv2.COLOR_BGR2YCR_CB)

L1,a1,b1 = cv2.split(img_Lab_2)
#H,S,V = cv2.split(img_HSV_2)
Y1,Cr1,Cb1 = cv2.split(img_YCrCb_2)
plt.figure('RGB')
plt.subplot(2,3,1)
plt.imshow(imagem_seg[:,:,0],cmap = 'gray')
plt.title('Segmentada - R')
plt.xticks([])
plt.yticks([])

plt.subplot(2,3,2)
plt.imshow(imagem_seg[:,:,1],cmap = 'gray')
plt.title('Segmentada - G')
plt.xticks([])
plt.yticks([])

plt.subplot(2,3,3)
plt.imshow(imagem_seg[:,:,2],cmap = 'gray')
plt.title('Segmentada - B')
plt.xticks([])
plt.yticks([])

#####lab

plt.figure('Lab')
plt.subplot(2,3,1)
plt.imshow(img_Lab_2[:,:,0],cmap = 'gray')
plt.title('Segmentada - L')
plt.xticks([])
plt.yticks([])

plt.subplot(2,3,2)
plt.imshow(img_Lab_2[:,:,1],cmap = 'gray')
plt.title('Segmentada - a')
plt.xticks([])
plt.yticks([])

plt.subplot(2,3,3)
plt.imshow(img_Lab_2[:,:,2],cmap = 'gray')
plt.title('Segmentada - b')
plt.xticks([])
plt.yticks([])

####HSV
'''
plt.figure('HSV')
plt.subplot(2,3,1)
plt.imshow(img_HSV_2[:,:,0],cmap = 'gray')
plt.title('Segmentada - H')
plt.xticks([])
plt.yticks([])

plt.subplot(2,3,2)
plt.imshow(img_HSV_2[:,:,1],cmap = 'gray')
plt.title('Segmentada - S')
plt.xticks([])
plt.yticks([])

plt.subplot(2,3,3)
plt.imshow(img_HSV_2[:,:,2],cmap = 'gray')
plt.title('Segmentada - V')
plt.xticks([])
plt.yticks([])

#####

plt.figure('YCrCb')
plt.subplot(2,3,1)
plt.imshow(img_YCrCb_2[:,:,0],cmap = 'gray')
plt.title('Segmentada - Y')
plt.xticks([])
plt.yticks([])

plt.subplot(2,3,2)
plt.imshow(img_YCrCb_2[:,:,1],cmap = 'gray')
plt.title('Segmentada - Cr')
plt.xticks([])
plt.yticks([])

plt.subplot(2,3,3)
plt.imshow(img_YCrCb[:,:,2],cmap = 'gray')
plt.title('Segmentada - Cb')
plt.xticks([])
plt.yticks([])
plt.show()
'''
###### seguimentação
(L_O, img_thresh_1) = cv2.threshold(L1,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
#(L_O, img_thresh_1) = cv2.threshold(L1,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
plt.subplot(1,2,1)
plt.imshow(imagem_seg)

plt.subplot(1,2,2)
plt.imshow(img_thresh_1, cmap='jet')
plt.show()

print('-='*50)
print('01.J) Realize operações aritméticas na imagem em RGB de modo a realçar os aspectos de seu interesse. '
      'Exemplo (2*R-0.5*G). Explique a sua escolha pelas operações aritméticas. Segue abaixo algumas sugestões.')

#img_opr_01 = np.uint8(1.5*img_rgb[:,:,2]) - np.uint8(2*img_rgb[:,:,0])
####
#vamos pegar todas as(:,) linhas e (,:,)colunas dos x canais.

img_opr_01 = (imagem_recortada[:,:,1]- 1.5*imagem_recortada[:,:,2])
img_opr_02 = (2*imagem_recortada[:,:,1] - imagem_recortada[:,:,0] - imagem_recortada[:,:,2])/(2*imagem_recortada[:,:,1] + imagem_recortada[:,:,0] + imagem_recortada[:,:,2]) #Green Leaf Index
img_opr_03 = 1.8*imagem_recortada[:,:,2] - 1.5*imagem_recortada[:,:,0] - imagem_recortada[:,:,1]   # A melhor para realçar as lesões de antracnose na nervura

print(img_opr_01)
#As novas imagens, após a operação, possuem apenas um canal!!!

img_opr_01 = img_opr_01.astype(np.uint8) #intervalo de 0 a 255.
print(img_opr_01.astype(np.uint8))


img_opr_02 = img_opr_02.astype(np.uint8) #intervalo de 0 a 255.
print(img_opr_02.astype(np.uint8))



########################################################################################################################
# Apresentar imagens

# Figura das modificações

plt.figure('MODIFICAÇÕES')
plt.subplot(2,2,1)
plt.imshow(imagem_recortada)
plt.title("RGB")

plt.subplot(2,2,2)
plt.imshow(img_opr_01,cmap='gray')
plt.title("chute")

plt.subplot(2,2,3)
plt.imshow(img_opr_02,cmap='gray')
plt.title("Green Leaf Index")

plt.subplot(2,2,4)
plt.imshow(img_opr_03,cmap='gray')
plt.title("1.8B-1.5R-G")

plt.show()