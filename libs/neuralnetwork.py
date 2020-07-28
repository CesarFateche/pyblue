# -*- coding: utf-8 -*-
"""
Created on Sat Jun  6 20:49:04 2020

@author: César
"""

import numpy as np
import torch


class MLP(torch.nn.Module):
    """
    MLP: Multiple Layer Perceptron
    """
    
    def __init__(self, input_shape, output_shape, hidden_shape):
        """
        Parameters
        ----------
        input_shape :
            Tamaño o forma de los datos de entrada.
        output_shape :
            Tamaño o forma de los datos de salida.
        hidden_shape :
            Tamanño o forma de la capaoculta.
        """
        super(MLP,self).__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.hidden_shape = hidden_shape
        
        self.fc1 = torch.nn.Linear(self.input_shape, self.hidden_shape[0])
        if len(self.hidden_shape) > 1:
            self.fc2 = torch.nn.Linear(self.hidden_shape[0], self.hidden_shape[1])
        if len(self.hidden_shape) > 2:
            self.fc3 = torch.nn.Linear(self.hidden_shape[1], self.hidden_shape[2])
        if len(self.hidden_shape) > 3:
            self.fc4 = torch.nn.Linear(self.hidden_shape[2], self.hidden_shape[3])
        if len(self.hidden_shape) > 4:
            self.fc5 = torch.nn.Linear(self.hidden_shape[3], self.hidden_shape[4])
        if len(self.hidden_shape) > 5:
            self.fc6 = torch.nn.Linear(self.hidden_shape[4], self.hidden_shape[5])
        if len(self.hidden_shape) > 6:
            self.fc7 = torch.nn.Linear(self.hidden_shape[5], self.hidden_shape[6])
        if len(self.hidden_shape) > 7:
            self.fc8 = torch.nn.Linear(self.hidden_shape[6], self.hidden_shape[7])
        self.out = torch.nn.Linear(self.hidden_shape[-1], self.output_shape)
        
    def forward(self, x):
        x = torch.from_numpy(x).float()
        x = torch.sigmoid(self.fc1(x))
        if len(self.hidden_shape) > 1:
            x = torch.sigmoid(self.fc2(x))
        if len(self.hidden_shape) > 2:
            x = torch.sigmoid(self.fc3(x))
        if len(self.hidden_shape) > 3:
            x = torch.sigmoid(self.fc4(x))
        if len(self.hidden_shape) > 4:
            x = torch.nn.functional.sigmoid(self.fc5(x))
        if len(self.hidden_shape) > 5:
            x = torch.nn.functional.sigmoid(self.fc6(x))
        if len(self.hidden_shape) > 6:
            x = torch.nn.functional.sigmoid(self.fc7(x))
        if len(self.hidden_shape) > 7:
            x = torch.nn.functional.sigmoid(self.fc8(x))
        x = self.out(x)
        return x

    
class CNN(torch.nn.Module):
    """
    Una red neuronal convolucional que tomará decisiones según los píxeles de la imagen
    """
    def __init__(self, input_shape, output_shape, hidden_shape):
        """
        Parameters
        ----------
        input_shape :
            Tamaño o forma de los datos de entrada.
        output_shape :
            Tamaño o forma de los datos de salida.
        hidden_shape :
            Tamanño o forma de la capaoculta.
        """
        super(CNN, self).__init__()
        len1 = hidden_shape[0]
        len2 = len1
        len3 = len1
        if len(hidden_shape) > 1:
            len2 = hidden_shape[1]
        if len(hidden_shape) > 2:
            len3 = hidden_shape[2]
        self.layer1 = torch.nn.Sequential(
            torch.nn.Dropout(p=0.5),
            torch.nn.Conv2d(input_shape[0], len1, kernel_size = 4, stride = 2, padding = 1),
            torch.nn.ReLU()
            )
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(len1, len2, kernel_size = 4, stride = 2, padding = 0),
            torch.nn.ReLU()
            )
        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(len2, len3, kernel_size = 3, stride = 1, padding = 0),
            torch.nn.ReLU()
            )
        self.out = torch.nn.Linear(19536, output_shape)

    def forward(self, x):
        x = torch.from_numpy(x).float()
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.view(x.shape[0], -1)
        x = self.out(x)
        return x        
        
if __name__ == "__main__":
    model = MLP(4, 2, [4,3])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    
    x = np.array([1,2,0,0])
    z = model.forward(x)
    #print(np.argmax(model(x).detach().numpy()))
    # Visualización
    def visualization(model):
        w = []
        b = []
        _x = torch.from_numpy(x).float()
        for i in range(int(len(list(model.parameters())) / 2)):
            w.append(list(model.parameters())[2*i].detach().numpy())
            b.append(list(model.parameters())[2*i+1].detach().numpy())
        print("+---+---+ w11: {:.2f}, w12: {:.2f}, w13: {:.2f}, w14: {:.2f}".
              format(w[0][0][0], w[0][0][1], w[0][0][2], w[0][0][3]))
        print("|", x[0], "|", x[1], "| w21: {:.2f}, w22: {:.2f}, w23: {:.2f}, w24: {:.2f}".
              format(w[0][1][0], w[0][1][1], w[0][1][2], w[0][1][3]))
        print("+---+---+ w31: {:.2f}, w32: {:.2f}, w33: {:.2f}, w34: {:.2f}".
              format(w[0][2][0], w[0][2][1], w[0][2][2], w[0][2][3]))
        print("|", x[2], "|", x[3], "| w41: {:.2f}, w42: {:.2f}, w43: {:.2f}, w44: {:.2f}".
              format(w[0][3][0], w[0][3][1], w[0][3][2], w[0][3][3]))
        print("+---+---+ b1: {:.2f}, b2: {:.2f}, b3: {:.2f}, b4: {:.2f}".
              format(b[0][0], b[0][1], b[0][2], b[0][3]))
        print()
        _x = torch.nn.functional.relu(model.fc1(_x))#x @ w[0] + b[0]
        print("+----+----+ w'11: {:.2f}, w'12: {:.2f}, w'13: {:.2f}, w'14: {:.2f}".
              format(w[1][0][0], w[1][0][1], w[1][0][2], w[1][0][3]))
        print("|{:.2f}|{:.2f}| w'21: {:.2f}, w'22: {:.2f}, w'23: {:.2f}, w'24: {:.2f}".
              format(_x[0], _x[1], w[1][1][0], w[1][1][1], w[1][1][2], w[1][1][3]))
        print("+----+----+ w'31: {:.2f}, w'32: {:.2f}, w'33: {:.2f}, w'34: {:.2f}".
              format(w[1][2][0], w[1][2][1], w[1][2][2], w[1][2][3]))
        print("|{:.2f}|{:.2f}| b'1: {:.2f}, b'2: {:.2f}, b'3: {:.2f}".
              format(_x[2], _x[3], b[1][0], b[1][1], b[1][2]))
        print("+----+----+")
        print()
        _x = torch.nn.functional.relu(model.fc2(_x))
        print("+----+----+----+ w''11: {:.2f}, w''12: {:.2f}, w''13: {:.2f}".
              format(w[2][0][0], w[2][0][1], w[2][0][2]))
        print("|{:.2f}|{:.2f}|{:.2f}| w''21: {:.2f}, w''22: {:.2f}, w''23: {:.2f}".
              format(_x[0], _x[1], _x[2], w[2][1][0], w[2][1][1], w[2][1][2]))
        print("+----+----+----+ b''1: {:.2f}, b''2: {:.2f}, b''3: {:.2f}".
              format(_x[2], _x[2], b[2][0], b[2][1]))
        print()
        _x = model.out(_x)
        print("+----+----+")
        print("|{:.2f}|{:.2f}|".format(_x[0], _x[1]))
        print("+----+----+")
    """
    print("Model's state_dict:")
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())
        
    print("Optimizer's state_dict:")
    for var_name in optimizer.state_dict():
        print(var_name, "\t", optimizer.state_dict()[var_name])
        
    """
    file_name = "Test2.ptm"
    # Load model
    agent_state = torch.load(file_name, map_location = lambda storage, loc: storage)
    model.load_state_dict(agent_state["Q"])
    #model.eval()
    
    # Before train
    #print(list(model.parameters())[0])
    visualization(model)
    print(model.forward(x))
    #"""
    # Train
    model_optimizer = torch.optim.Adam(model.parameters(), lr = 0.01)
    td_error = torch.nn.functional.mse_loss(model(x)[1], torch.tensor(-1.0, requires_grad = True))
    model_optimizer.zero_grad()
    td_error.backward()
    model_optimizer.step()
    #"""
    # After train
    #print(list(model.parameters()))
    visualization(model)
    print(model.forward(x))
    
    # Save model
    agent_state = {"Q": model.state_dict()}
    torch.save(agent_state, file_name)
    #print(model.state_dict())
    
    
    #print(x)
    #print(z)
