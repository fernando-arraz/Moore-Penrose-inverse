import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import glob
import math
import os


#Funcao que escolhe o dataset
def chooseDataset (dataset):
    if dataset == "Dataset":
        path = "DatasetMML"
        lenImage = (255,255)
    elif dataset == "lfw":
        path = "lfw"
        lenImage = (250,250)
    else:
        print("Dataset não existe.")
    return path, lenImage


def files_in_folder(path):
    """[summary]

    Args:
        path ([type]): [description]

    Returns:
        [type]: [description]
    """

    files = []
    for dirname, _, filenames in os.walk(path):
        # for dirname, _, filenames in os.walk('/kaggle/input'):
        for filename in filenames:
            # print(os.path.join(dirname, filename))
            files.append(os.path.join(dirname, filename))
    return files
        

# Funcao de processamento do dataset
def readImages (path):
    # Leitura das imagens
    # imgs = glob.glob(path+'/*.'+extension)
    imgs = path
    base = [Image.open(i).convert('L') for i in imgs]
    
    # Tamanho do dataset
    size = len(base)
    
    # Passar as imagens para um array
    X = np.array([base[i].getdata() for i in range(size)])
    return X, base, size

def procrustes(X, path):
    B = np.ones([X.shape[0], X.shape[0]])
    lines = path
    cols  = path
    lin = 0
    for i in lines:
        col = 0
        for j in cols:          
            if i.split('\\')[-1].split('_')[:2] == j.split('\\')[-1].split('_')[:2]:
                B[lin, col] = 0
            col += 1
        lin +=1
    return np.dot(np.linalg.pinv(X), B), B



# Implementacao do PCA
# def pca(X, confidence=0.8):
#     # Media do dataset
#     mean = np.mean(X,0)
    
#     # Centrar os dados
#     phi = X - mean
    
#     # Calcular os vetores e valores proprios atraves do SVD
#     eigenvectors, sigma, variance = np.linalg.svd(phi.transpose(), full_matrices=False)
#     eigenvalues = sigma*sigma
    
#     # Ordenacao dos valores pp
#     idx = np.argsort(-eigenvalues)
#     eigenvalues = eigenvalues[idx]
#     eigenvectors = eigenvectors[:,idx]
    
#     # Determinar o n. de vectores pp a usar
#     k = 0
#     traco = np.sum(eigenvalues)
#     while(np.sum(eigenvalues[:k])/traco < confidence):
#         k = k+1
        
#     print(k)
    
#     # Escolher os vetores pp associados
#     eigenvectors = eigenvectors[:,0:k]
#     return eigenvalues, eigenvectors, phi, mean, variance


# Calculo dos coeficientes da projeccao
def coefProj(phi, eigenvectors, size):
    coef_proj = [np.dot(phi[i], eigenvectors) for i in range(size)]
    return coef_proj


# Verificar se identifica ou nao o input
def testar(input_img, mean, eigenvectors, eigenvalues, size, coef_proj, distance="mahalanobis", dataset="yale_training"):
    # Centrar o input
    gamma = np.array(input_img.getdata())
    test_phi = gamma - mean
    
    # Calcular os coeficientes da projeccao do input
    test_coef_proj = np.dot(test_phi, eigenvectors)
    
    if distance == "euclidian":
        dist = [np.linalg.norm(coef_proj[i] - test_coef_proj) for i in range(size)]
        if dataset=="yale_training":
            limit = 7600
        elif dataset=="DatasetMML":
            limit = 1900
        else:
            print("Dataset inválido.")
    elif distance == "mahalanobis":
        dist = mahalanobis(coef_proj, test_coef_proj, eigenvalues, eigenvectors.shape[1])
        if dataset=="yale_training":
            limit = 0.05
        elif dataset=="DatasetMML":
            limit = 0.035
        else:
            print("Dataset inválido.")
    else:
        print("Distância inválida.")

    d_min = np.min(dist)
    
    if d_min < limit:
        print('Imagem nr.: '+str(np.argmin(dist))+'\n'+'Distancia minima: '+str(d_min)+'\n')
        return 0
    else:
        print('Falhou no reconhecimento.')
        return (-1)
        
    #return test_coef_proj

# Distancia euclidiana
def euclidian(x, y):
    if x.size != y.size:
        return (-1)
    z = y - x
    distance = math.sqrt(sum(z**2))
    return round(distance, 2)

# Distancia de Mahalanobis
def mahalanobis(x, y, eigenvalues, k):
    if len(x[0]) != len(y):
        return (-1)
    N = len(x)
    distance=[]
    for i in range(N):
        distance.append(np.sum(np.divide((x[i]-y)**2, eigenvalues[:k])))
    return distance


if __name__ == "__main__":
    # main("Hello World Farraz")
    test = files_in_folder("lfw")
    print(test)
    print(len(test))

    path, lenImage = chooseDataset("lfw")
    path

    # Ler das imagens para uma matriz 'X' e um array 'lista'
    X, lista, size = readImages(files_in_folder("lfw")) 
    # print(X.shape)   

    # print(np.linalg.pinv(X).shape)

    # print(procrustes(X, files_in_folder("lfw")))

    Q, B = procrustes(X, files_in_folder("lfw"))

    Test = Image.open("C:\Projetos\Penrose_inverse_face_rec\Aaron_Peirsol_0004.jpg").convert('L')

    Test = np.array([Test.getdata()])
    print(B.shape, Q.shape, Test.shape)
    diff = []
    for i in range(B.shape[0]):
        # print(">>>>>>>>>>>>>>>>>",i.shape)
        diff.append( np.linalg.norm(B[i,:] - np.dot(Test, Q)))
        # print(len(diff))
    print(np.argmin(diff))
    print(files_in_folder("lfw")[np.argmin(diff)])


    # print(diff[0])