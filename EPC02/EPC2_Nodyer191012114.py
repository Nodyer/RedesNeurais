'''
Nome: Nodyer Henrique Nakanishi dos Anjos
RA: 191012114
'''

import numpy as np
import matplotlib.pyplot as plt

# Definição da classe Adaline

class Adaline:
    def __init__(self, input_data):
        self.input_data = input_data
        self.weights = np.random.rand(self.input_data.shape[1]+1,) # onde o primeiro valor é o bias (theta)
        #print('Pesos gerados: %s' %self.weights)
   
    def activation_function(self, linear_combination):
        # Função degrau bipolar
        return np.where(linear_combination >= 0,1,-1) 

    def linear_combination(self, input):
        return np.dot(input, self.weights)

    # Previsão do modelo
    def predict(self, test_input_data):
        test_input_data = np.insert(test_input_data,0,-1,1)        
        return self.activation_function(self.linear_combination(test_input_data))

    # Treinamento do modelo
    def train(self, labels, learning_rate, training_error, max_epochs=10000):

        current_mse = 0
        previous_mse = 1
        list_mse = []
        epoch = 0

        # Inserindo a entrada constante de -1
        training_input = np.insert(self.input_data,0,-1,1)

        while abs(current_mse  - previous_mse) > training_error and epoch <= max_epochs:
            previous_mse = current_mse
            epoch+=1
            sum_error = 0
            for input, label in zip(training_input, labels):
                update = label - self.linear_combination(input)
                self.weights += learning_rate * update * input
                sum_error += update**2
            current_mse = sum_error/training_input.shape[0]
            list_mse.append(current_mse)
        return epoch, list_mse

if __name__ == "__main__":

    # Entradas do  treinamento
    inputs = np.array([
        [0, 1, 1], 
        [1, 1, 1], 
        [0, 1, 0], 
        [1, 1, 0],
        [1, 0, 1],
        [1, 0, 0],
        [0, 0, 1],
        [0, 0, 0]
        ])

    # Saída desejada
    labels = np.array(
        [-1, -1, 1, 1, -1, 1, -1, 1]
        )

    # Taxa de aprendizagem
    learning_rate = 0.1

    # Erro de treinamento
    training_error = 0.00001

    adaline = Adaline(inputs)
    num_epochs, errors = adaline.train(labels, learning_rate, training_error)

    #print("Número de épocas: ", num_epochs)
    #print("Pesos finais: ", adaline.weights)
    #print("Erros: ", errors)

    # Gerando ruido

    noise = np.random.randn()/5
    sigma = np.random.rand()
    mu = np.random.rand()

    contaminated_inputs_1 = noise + inputs
    contaminated_inputs_2 = sigma*noise + mu + inputs

    test_1 = adaline.predict(contaminated_inputs_1)
    test_2 = adaline.predict(contaminated_inputs_2)

    # Mostrando a saída

    print("\t","\t", "Sinal","\t","\t","\t","Sinal com ruído","\t","\t","Saída","\t","Categoria")
    for i in range(8):
        print("Sinal", i+1, "==> ", "\t", inputs[i], "\t",contaminated_inputs_1[i], "\t", test_1[i], "\t", "Classe A" if test_1[i] == -1 else "Classe B")
    for i in range(8):
        print("Sinal", i+9, "==> ", "\t", inputs[i], "\t",contaminated_inputs_2[i], "\t", test_1[i], "\t", "Classe A" if test_2[i] == -1 else "Classe B")
    
    # Gerando gráfico

    x_axis = [i for i in range(num_epochs)]
    y_axis = errors

    plt.plot(x_axis,y_axis); plt.axis(); plt.xlabel("N° de épocas"); plt.ylabel("EMQ"); plt.title("N° de épocas x EMQ")
    plt.show()