# -*- coding: utf-8 -*-
"""
Created on Sun Jun  7 12:35:29 2020

@author: César
"""
##############################################################################
# Script encargado de la interacción entre el agente y el emulador
##############################################################################

import os
import numpy as np
from pyboy import PyBoy, WindowEvent

if __name__ == "__main__":
    import utils
else:
    from envs.utils import rgb2bit, tile_array, MemoryMap


def get_games_list():
    if __name__ == "__main__":
        files = os.listdir("../roms")
    else:
        files = os.listdir("roms")
    roms = []
    for file in files:
        if file[-3:] == ".gb" or file[-4:] == ".gbc":
            roms.append(file)
    return roms

def initialize_emulator(emu_conf = {"gamerom": get_games_list()[0]}):

    gamerom = "roms/"+emu_conf["gamerom"]
    
    if "bootrom_file" in emu_conf:
        bootrom_file = emu_conf["bootrom_file"]
    else:
        bootrom_file = None

    if "profiling" in emu_conf:
        profiling = emu_conf["profiling"]
    else:
        profiling = False   
        
    if "disable_renderer" in emu_conf:
        disable_renderer = emu_conf["disable_renderer"]
    else:
        disable_renderer = False    
    
    if "sound" in emu_conf:
        sound = emu_conf["sound"]
    else:
        sound = True
    
    if "color_palette" in emu_conf:
        color_palette = rgb2bit(emu_conf["color_palette"])
    else:
        color_palette = rgb2bit("Blue")
    
    if "gamespeed" in emu_conf:
        gamespeed = emu_conf["gamespeed"]
    else:
        gamespeed = 0
    
    kwargs_pyboy = {"bootrom_file": bootrom_file,
                    "profiling": profiling,
                    "disable_renderer": disable_renderer,
                    "sound": sound,
                    "color_palette": color_palette        
                    }

    gameboy = PyBoy(gamerom, **kwargs_pyboy)
    gameboy.set_emulation_speed(gamespeed)
    print(gameboy.cartridge_title(), "speed:", gamespeed)
    if disable_renderer:
        gameboy.stop(save = False)
    
    return gameboy

def reset_emulator(emulator, emu_conf):
    emulator.stop(save = False)
    return initialize_emulator(emu_conf)

def save_state(emulator, emu_conf, number):
    file = open("states/" + emu_conf["gamerom"] + str(number) + ".state", "wb")
    emulator.save_state(file)

def load_state(emulator, emu_conf, number):
    file = open("states/" + emu_conf["gamerom"] + str(number) + ".state", "rb")
    emulator.load_state(file)

##############################################################################
# Clase encargada de la observación
##############################################################################

class Observer:
    
    def __init__(self, emulator = None):
        self.emulator = emulator
        #self.mm = MemoryMap(emulator)
        
    def observation(self):
        image = self.emulator.botsupport_manager().screen().screen_ndarray().transpose((2,0,1)) / 255.0
        image = np.expand_dims(image, 0)
        return image
        """
        vram = self.mm.get_vram()
        
        sprites = []
        for i in range(40):
            sprite = self.emulator.botsupport_manager().sprite(i)
            identifier = sprite.tiles[0].tile_identifier
            if sprite.on_screen:
                x = (sprite.x + 8) / 168
                y = (sprite.y + 8) / 152
                flipx = float(sprite.attr_x_flip)
                flipy = float(sprite.attr_y_flip)
                image = tile_array(vram[16*identifier:16*(identifier+1)], pooling = 2)
            else:
                x, y, flipx, flipy, image = 0, 0, 0, 0, np.zeros(16)
            sprites.append(np.concatenate(([x,y,flipx,flipy],image)))
        sprites = np.array(sprites).flatten()
        
        pool = 4
        dim = int(64 / pool ** 2)
        windows = np.zeros((360*dim))
        #backgrounds = np.zeros((360*16))
        window = self.emulator.botsupport_manager().tilemap_window()
        #background = self.emulator.botsupport_manager().tilemap_background()
        (scx, scy), (wx, wy) = self.emulator.botsupport_manager().screen().tilemap_position()
        scx, scy = int(scx / 8), int(scy / 8)
        for x in range(20):
            for y in range(18):
                index = (x + y * 20) * dim
                windows[index:index + dim] = tile_array(vram[16 * window[x + scx, y + scy]:
                                                            16 * (window[x + scx, y + scy] + 1)],
                                                       pooling = pool)
                #backgrounds[index:index + 16] = tile_array(vram[16 * background[x + scx, y + scy]:
                #                                            16 * (background[x + scx, y + scy] + 1)],
                #                                       pooling = 2)
            
        return windows#np.concatenate((sprites, windows, backgrounds))
        """
##############################################################################
# Clase encargada de la ejecución de acciones en el emulador
##############################################################################
        
class Actuator:
    
    def __init__(self, emulator, agent_params):
        self.emulator = emulator
        self.params = agent_params
    
    def step(self, order):
        self.order = order
        
        buttons = {"up"    : [WindowEvent.PRESS_ARROW_UP, WindowEvent.RELEASE_ARROW_UP],
                   "down"  : [WindowEvent.PRESS_ARROW_DOWN, WindowEvent.RELEASE_ARROW_DOWN],
                   "left"  : [WindowEvent.PRESS_ARROW_LEFT, WindowEvent.RELEASE_ARROW_LEFT],
                   "right" : [WindowEvent.PRESS_ARROW_RIGHT, WindowEvent.RELEASE_ARROW_RIGHT],
                   "A"     : [WindowEvent.PRESS_BUTTON_A, WindowEvent.RELEASE_BUTTON_A],
                   "B"     : [WindowEvent.PRESS_BUTTON_B, WindowEvent.RELEASE_BUTTON_B],
                   "select": [WindowEvent.PRESS_BUTTON_SELECT, WindowEvent.RELEASE_BUTTON_SELECT],
                   "start" : [WindowEvent.PRESS_BUTTON_START, WindowEvent.RELEASE_BUTTON_START]}
        
        #print(order)
        if order == "none":
            self.emulator.tick()
        else:
            self.emulator.send_input(buttons[self.order][0])
            self.emulator.tick()
            self.emulator.send_input(buttons[self.order][1])
        for _ in range(self.params["frame_skip"]):
            self.emulator.tick()

##############################################################################
# Clase encargada de asignar recompensas
##############################################################################
    
class Rewarder:
    
    def __init__(self, emulator, goal_params):
        self.emulator = emulator
        self.mm = MemoryMap(emulator)
        self.params = goal_params
        self.is_done = np.zeros(len(self.params), dtype = bool)
    
    def reward(self):
        reward = np.zeros(len(self.params))
        
        # Check player's name
        goal_player_name = self.params["player_name"]
        actual_player_name = self.mm.player_name()
        if actual_player_name != "" and actual_player_name != "NINTEN":
            if self.is_done[0]:
                reward[0] = -100
            else:
                # Largo del nombre
                rew1 = 1 - np.abs(len(goal_player_name) - len(actual_player_name)) / len(goal_player_name)
                # Coincidencia de letras
                rew2 = 0
                for letter in actual_player_name:
                    if letter in goal_player_name:
                        rew2 += 1 / len(actual_player_name)
                    elif letter.lower() in goal_player_name.lower():
                        rew2 += 0.5 / len(actual_player_name.lower())
                # Coincidencia de letra bien ubicada
                rew3 = 1
                factor = np.min([len(goal_player_name), len(actual_player_name)])
                for letter in range(factor):
                    if goal_player_name[letter].lower() != actual_player_name[letter].lower():
                        rew3 = rew3 - 1/factor
                    elif goal_player_name[letter] != actual_player_name[letter]:
                        rew3 = rew3 - 0.5/factor
                # Recompensa por letras seguidas
                for i in range(1, factor):
                    for j in range(factor - i):
                        if goal_player_name[j:j + i + 1] == actual_player_name[j:j + i + 1]:
                            rew3 += 1
                # Castigo por elegir nombre del menú de nombres
                if actual_player_name[:4] in ["AZUL", "GARY", "JUAN"]:
                    rew4 = -1
                else:
                    rew4 = 1
                # Castigo por tiempo que se tardó en poner el nombre
                reward[0] = rew4 * (0.5 * rew1 + rew2 + rew3) / 3# - self.emulator.frame_count / 100000
                reward[0] = reward[0]
                print("Goal player name: {}, Actual player name: {}".format(goal_player_name,
                                                                        actual_player_name))
                self.is_done[0] = True
            
        # Check rival's name
        goal_rival_name = self.params["rival_name"]
        actual_rival_name = self.mm.rival_name()
        if actual_rival_name != "" and actual_rival_name != "SON" and not self.is_done[1]:
            if self.is_done[1]:
                reward[1] = -100
            else:
                if len(goal_rival_name) == len(actual_rival_name):
                    reward[1] += 1 # Recompensa por longitud de nombre
                for letter in actual_rival_name:
                    if letter in goal_rival_name:
                        reward[1] += 1 # Recompensa por existir la letra en el nombre objetivo
                let_rew = 0
                for letter in range(np.min([len(goal_rival_name),len(actual_rival_name)])):
                    let_rew += 1
                    if goal_rival_name[letter] == actual_rival_name[letter]:
                        reward[1] += let_rew # Recompensa por cada letra en su lugar correcto
                reward[1] -= self.emulator.frame_count / 1000
                print("Goal rival name: {}, Actual rival name: {}".format(goal_rival_name,
                                                                        actual_rival_name))
                self.is_done[1] = True
        
        # Pick up visible items
            
        # Pick up hidden items
        
        # Catch pokemon
        
        # Train pokemon
        
        # Beat wild pokemon
        
        # Beat Trainer
        
        """
        # Dummy
        goal_dummy = self.params["dummy"]
        actual_dummy = dummy
        if not self.is_done[2]:
            if actual_dummy >= goal_dummy:
                reward[2] += 1
                print("Goal dummy: {}, Actual dummy: {}".format(goal_dummy, actual_dummy))
                self.is_done[2] = True
        """
        # Max num of frames
        if self.emulator.frame_count > 10000:
            print("Finalización por conteo de {} frames".format(self.emulator.frame_count))
            reward[0] = 0
            self.is_done = np.ones(len(self.params), dtype = bool)
        
        return reward
         
if __name__ == "__main__":
    print(get_games_list())
