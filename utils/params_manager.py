# -*- coding: utf-8 -*-
"""
Created on Thu May 14 20:30:17 2020

@author: César
"""


import json

class ParamsManager(object):
    """ Administrador de parámetros almacenados en un archivo json """
    def __init__(self, params_file):
        """
        :param params_file: archivo donde están almacenados los parámetros
        :return:
        """
        self.params = json.load(open(params_file, "r"))
        
    def get_params(self):
        """
        :return: Todos los parámetros almacenados en el archivo
        """
        return self.params
    
    def get_goal_params(self):
        """
        :return: Parámetros objetivo
        """
        return self.params["goal"]
    
    def get_hidden_params(self):
        """
        :return: Forma de las capas ocultas para las NN de cada objetivo
        """
        for key, value in self.params["hidden_layer_shape"].items():
            value = value.split(",")
            value = list(map(int, value))
            self.params["hidden_layer_shape"][key] = value
        return self.params["hidden_layer_shape"]
    
    def get_strategy_params(self):
        """
        :return: Parámetros de la NN estrategia
        """
        return self.params["strategy"]
       
    def get_agent_params(self):
        """
        :return: Parámetros generales para cada agente o NN orientada a objetivo
        """
        return self.params["agent"]
    
    def get_emulator_params(self):
        """
        :return: Parámetros del emulador
        """
        return self.params["emulator"]
    
    def update_agent_params(self, **kwargs):
        for key, value in kwargs.items():
            if key in self.get_agent_params().keys():
                self.params["agent"][key] = value
        
    def export_goal_params(self, file_name):
         with open(file_name, "w") as f:
            json.dump(self.params["goal"], f, indent = 4, separators = (",", ":"), sort_keys = True)
            f.write("\n")       
    
    def export_agent_params(self, file_name):
        with open(file_name, "w") as f:
            json.dump(self.params["agent"], f, indent = 4, separators = (",", ":"), sort_keys = True)
            f.write("\n")

    def export_emulator_params(self, file_name):
        with open(file_name, "w") as f:
            json.dump(self.params["emulator"], f, indent = 4, separators = (",", ":"), sort_keys = True)
            f.write("\n")
            
            
if __name__ == "__main__":
    print("Probando nuestro manager de parametros...")
    param_file = "../parameters.json"
    manager = ParamsManager(param_file)
   
    goal_params = manager.get_goal_params()
    print("Los parámetros objetivo son: ")
    for key, value in goal_params.items():
        print(key, ": ", value)
    print()
   
    hidden_params = manager.get_hidden_params()
    print("Los parámetros ocultos son: ")
    for key, value in hidden_params.items():
        print(key, ": ", value)
    print()    
    
    agent_params = manager.get_agent_params()
    print("Los parámetros del agente son: ")
    for key, value in agent_params.items():
        print(key, ": ", value)
    print()
         
    emulator_params = manager.get_emulator_params()
    print("Los parámetros del emulador son: ")
    for key, value in emulator_params.items():
        print(key, ": ", value)
    print()
    
    manager.update_agent_params(learning_rate = 0.01, gamma = 0.92)
    
    agent_params_updated = manager.get_agent_params()
    print("Los parámetros del agente actualizados son: ")
    for key, value in agent_params_updated.items():
        print(key, ": ", value)
    print()
    
    print("Fin de la prueba")
            
            
            
            
            