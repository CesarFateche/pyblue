# -*- coding: utf-8 -*-
"""
Created on Tue May 12 19:01:31 2020

@author: César
"""

from collections import namedtuple
import random

Experience = namedtuple("Experience", ["obs", "action", "reward", "next_obs", "done"])

class ExperienceMemory(object):
    """ Buffer que simula la memoria, experiencia del agente """
    def __init__(self, capacity = int(1e6)):
        """
        :param capacity: Capacidad total de la memoria cíclica (número máximo de experiencias almacenables)
        :return:
        """
        self.capacity = capacity
        self.memory = []
        self.index = 0 # Indice de la experiencia actual      
    
    def store(self, *args):
        """
        :param *args: Objeto experiencia a ser almacenado en memoria
        :return:
        """
        if self.__len__() < self.capacity:
            self.memory.append(None)
        self.memory[self.index] = Experience(*args)
        self.index = int((self.index + 1) % self.capacity)
    
    def sample(self, batch_size):
        """
        :param batch_size: Tamaño de la memoria a recuperar
        :return: Una muestra del tamaño batch_size de experiencias aleatorias de la memoria
        """
        assert batch_size <= self.__len__(), "El tamño de la muestra es superior a la memoria disponible"
        return random.sample(self.memory, int(batch_size))
    
    def __len__(self):
        """
        :return: Número de experiencias almacenadas en la memoria
        """
        return len(self.memory)
    

        
        
        
        
        
        
        
        
        
        
        
        