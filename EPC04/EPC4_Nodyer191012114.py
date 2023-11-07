'''
Nome: Nodyer Henrique Nakanishi dos Anjos
RA: 191012114
'''

import numpy as np
import matplotlib.pyplot as plt

# Geração dos dados para o treinamento e teste com 600 padrões
class Dataset:

    def create_data(self, number_examples=600):

        x1 = np.random.uniform(low=0, high=np.pi/2, size=(number_examples,1))
        x2 = np.random.uniform(low=0, high=np.pi/2, size=(number_examples,1))
        x3 = np.random.uniform(low=0, high=np.pi/2, size=(number_examples,1))
        x = np.concatenate((x1, x2, x3), axis=1)

        # Vetor de valores desejado
        d=1/3*(np.sin(x1)+np.sin(x2)+np.sin(x3))

        # Matriz de padrões de treinamento e teste 
        xd = np.concatenate((x,d), axis=1)

        return xd

    # Divisão dos dados para validação cruzada
    def partition_data(self, number_examples=600, number_test_examples=100):

        xd = self.create_data(number_examples)
        n_features = xd.shape[0]
        xdt = []

        for i in range(1, number_test_examples+1):
            k = int(np.round((n_features-i)*np.random.rand()))
            xdt.append(xd[k])
            xd = np.delete(xd, k, 0)
        
        x, d = np.delete(xd, 3, 1), np.delete(xd, [0,1,2], 1)
        xt, dt = np.delete(xdt, 3, 1), np.delete(xdt, [0,1,2], 1)
        
        return x, d, xt, dt 


class RNA:
    def __init__(self, n_neurons=8, learning_decay = False):
        self.n_neurons = n_neurons

        self.w01 = 0 
        self.w02 = 0 
        self.w1 = 0 
        self.w2 = 0

        self.graph_x = [] 
        self.graph_y = [] 

        self.learning_decay = learning_decay

        self.y1 = np.ones((self.n_neurons,1))
    
    def tanh(self, a):
        return (np.exp(a) - np.exp(-a))/(np.exp(a) + np.exp(-a))
  
    def predict(self, activation_func, x):
        y2_list = []
        for n in range(x.shape[0]):
            for j in range(self.n_neurons):
                self.y1[j] = activation_func((np.dot(x[n],self.w1[j].T) + self.w01[j].T))                                                               
            y2 = activation_func((np.dot(self.y1.T, self.w2) + self.w02))
            y2_list.append(y2[0][0])
        return y2_list
    
    def train(self, x, d, activation_func, error_threshold=0.000001, learning_rate=0.1, decay_rate=0.001, max_epoch=2000):
        epoch = 0
        eqm_ant = 1
        eqm_atual = 0
        nro_padr = x.shape[0]
        nro_entr = x.shape[1]
        self.w2 = np.random.rand(self.n_neurons,1)
        self.w1 = np.random.rand(self.n_neurons,nro_entr)
        self.w02 = np.random.rand()
        self.w01 = np.random.rand(self.n_neurons,1)

        self.graph_x = []
        self.graph_y = []

        while abs(eqm_atual - eqm_ant) > error_threshold and epoch <= max_epoch:
            epoch+=1
            eqm_ant = eqm_atual
            dw2 = 0
            dw1 = np.ones((self.n_neurons,nro_entr))
            dw01 = np.ones((self.n_neurons,1))
            grad_N_escond = np.ones((self.n_neurons,1))
            soma_eq = 0

            for n in range(nro_padr):
                for j in range(self.n_neurons):
                    combination_linear = np.dot(x[n],self.w1[j].T) + self.w01[j].T
                    self.y1[j] = activation_func(combination_linear[0])
                y2 = activation_func((np.dot(self.y1.T, self.w2)[0][0] + self.w02))
                error = d[n][0] - y2 
                grad_saida = error * y2 * (1 - y2)
                dw2 = learning_rate * grad_saida * self.y1
                dw02 = learning_rate * grad_saida * (-1)
                self.w2 += dw2
                self.w02 += dw02

                for j in range(self.n_neurons):
                    grad_N_escond[j] = (self.y1[j]) * (1-self.y1[j]) * grad_saida * self.w2[j]
                    dw1[j]= learning_rate * grad_N_escond[j] * x[n] 
                    dw01[j]= learning_rate * grad_N_escond[j] * (-1)                     
                self.w1 += dw1 
                self.w01 += dw01

                soma_eq += 0.5*error**2

            eqm_atual = soma_eq/nro_padr
            
            if self.learning_decay:
                learning_rate *= 1/(1 + decay_rate + epoch) 

            self.graph_x.append(epoch)
            self.graph_y.append(eqm_atual)

    def graph_values(self):
        '''Retorna, em uma lista, os valores de X e Y obtidos no treinamento'''
        return self.graph_x, self.graph_y
    
if __name__ == "__main__":

    Dataset = Dataset()

    x1, d1, xt1, dt1 = Dataset.partition_data(600, 100)
    x2, d2, xt2, dt2 = Dataset.partition_data(700, 100)
    x3, d3, xt3, dt3 = Dataset.partition_data(800, 100)

    # Parâmetros fornecidos pelo usuário para rede 1
    learning_rate = 0.1
    decay_rate = 0.001
    error_threshold = 0.000001
    max_epoch = 2000
    n_numbers = 8

    RNA1 = RNA(n_numbers, learning_decay=False)

    RNA1.train(x1, d1, RNA1.tanh, error_threshold, learning_rate, decay_rate, max_epoch)

    # Rede 2
    n_numbers2 = 5

    RNA2 = RNA(n_numbers2, learning_decay=False)
    RNA2.train(x1, d1, RNA1.tanh, error_threshold, learning_rate, decay_rate, max_epoch)

    # Rede 3
    n_numbers3 = 10

    RNA3 = RNA(n_numbers3, learning_decay=False)
    RNA3.train(x1, d1, RNA1.tanh, error_threshold, learning_rate, decay_rate, max_epoch)

    # Rede 4
    RNA4 = RNA(n_numbers, learning_decay=False)
    RNA4.train(x2, d2, RNA1.tanh, error_threshold, learning_rate, decay_rate, max_epoch)

    # Rede 5
    RNA5 = RNA(n_numbers, learning_decay=False)
    RNA5.train(x3, d3, RNA1.tanh, error_threshold, learning_rate, decay_rate, max_epoch)

    # Função f para 10 pontos

    results = RNA1.predict(RNA1.tanh, xt1)

    arr = np.array(results, ndmin=2)
    arr = arr.T

    print("Rede","\t","\t","\t","Valor real", "\t","\t","Diferença")
    for i in range(11):
        print(f'{arr[i][0]} \t {dt1[i][0]} \t {np.abs(arr[i][0] - dt1[i][0])}')

    # Exibição dos resultado em gráfico (Questão a)

    x1, y1 = RNA1.graph_values()
    x2, y2 = RNA2.graph_values()
    x3, y3 = RNA3.graph_values()
    x4, y4 = RNA4.graph_values()
    x5, y5 = RNA5.graph_values()

    plt.figure(figsize=(10,8))

    titles = ['Rede 1', 'Rede 2', 'Rede 3', 'Rede 4', 'Rede 5']
    x = [x1, x2, x3, x4, x5]
    y = [y1, y2, y3, y4, y5]

    for i in range(5):
        plt.subplot(3,2,i+1),plt.plot(x[i], y[i])
        plt.xlabel("Época")
        plt.ylabel("Erro quadrático médio")
        plt.title(titles[i])
    plt.show()