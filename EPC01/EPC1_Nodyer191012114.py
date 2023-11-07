'''
Nome: Nodyer Henrique Nakanishi dos Anjos
RA: 191012114
'''

import numpy as np

#Definição da classe Perceptron

class Perceptron:
    def __init__(self, inputs):
        self.weights = inputs
   
    def activation_function(self, combination_linear):
        #Função degrau bipolar
        return 1 if combination_linear >= 0 else -1 

    #Previsão do modelo
    def predict(self, input):
        combination_linear = np.dot(input, self.weights)
        return self.activation_function(combination_linear)

    #Treinamento do modelo
    def train(self, training_inputs, labels, learning_rate, max_epochs=100):
        errors = []
        for epoch in range(max_epochs):
            error = 0
            for input, label in zip(training_inputs, labels):
                prediction = self.predict(input)
                update = learning_rate * (label - prediction)
                self.weights += update * input
                error += int(update != 0.0)
            errors.append(error)
            if error == 0:
                break
        return epoch+1, errors

if __name__ == "__main__":

    # Entradas do  treinamento
    training_inputs = np.array([
        [-1, 0, 1, 1], 
        [-1, 1, 1, 1], 
        [-1, 1, 0, 1], 
        [-1, 1, 1, 0],
        [-1, 0, 0, 0]
        ])

    # Saída desejada
    labels = np.array(
        [-1, -1, -1, 1, 1]
        )

    # Tava de aprendizagem
    learning_rate= 0.01

    # Pesos
    weight = np.array(
        [0.07, 0.63, 0.89, 0.27]
    )

    perceptron = Perceptron(weight)
    num_epochs, errors = perceptron.train(training_inputs, labels, learning_rate)

    print("Número de épocas: ", num_epochs)
    print("Pesos finais: ", perceptron.weights[1:])
    print("Bias final: ", perceptron.weights[0])

    #Verificando os padrões

    input_test = np.array([
        [-1, 0, 1, 0],
        [-1, 1, 0, 0],
        [-1, 0, 0, 1]
    ])

    for i in range(len(input_test)):
        test = perceptron.predict(input_test[i])
        if test == -1:
            print("Classe A: ", test)
        else:
            print("Classe B: ", test)