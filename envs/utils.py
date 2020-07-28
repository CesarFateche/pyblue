# -*- coding: utf-8 -*-
"""
Created on Sun Jun  7 13:08:56 2020

@author: César
"""

import numpy as np

# Paletas de colores de 2-bit

def rgb_hex(red, green, blue):
    return red*256*256+green*256+blue

def rgb2bit(palette = "Blue"):
    maps = {"Blue"    : (rgb_hex(255,255,255), rgb_hex(127,127,255), rgb_hex(0,0,127), rgb_hex(0,0,0)),
            "Blue_r"  : (rgb_hex(0,0,0), rgb_hex(0,0,127), rgb_hex(127,127,255), rgb_hex(255,255,255)),
            "Red"     : (rgb_hex(255,255,255), rgb_hex(255,127,127), rgb_hex(127,0,0), rgb_hex(0,0,0)),
            "Red_r"   : (rgb_hex(0,0,0), rgb_hex(127,0,0), rgb_hex(255,127,127), rgb_hex(255,255,255)),
            "Green"   : (rgb_hex(255,255,255), rgb_hex(127,255,127), rgb_hex(0,127,0), rgb_hex(0,0,0)),
            "Green_r" : (rgb_hex(0,0,0), rgb_hex(0,127,0), rgb_hex(127,255,127), rgb_hex(255,255,255)),
            "Yellow"  : (rgb_hex(255,255,255), rgb_hex(255,255,127), rgb_hex(127,127,0), rgb_hex(0,0,0)),
            "Yellow_r": (rgb_hex(0,0,0), rgb_hex(127,127,0), rgb_hex(255,255,127), rgb_hex(255,255,255)),
            "Gray"    : (rgb_hex(255,255,255), rgb_hex(170,170,170), rgb_hex(85,85,85), rgb_hex(0,0,0)),
            "Gray_r"  : (rgb_hex(0,0,0), rgb_hex(85,85,85), rgb_hex(170,170,170), rgb_hex(255,255,255)),
    }
    if palette in maps.keys():
        return maps[palette]
    else:
        print("ERROR: No existe la paleta de colores '", palette, "'. Se utilizará por defecto 'Blue'.",
              "Paletas disponibles: ", maps)
        return maps["Blue"]
    
# Convertidor de tile bytes a tile array

def tile_array(tile, pooling = None):
    pixels = []
    for byte in range(0, 16, 2):
        byte1 = np.binary_repr(tile[byte], width = 8)
        byte2 = np.binary_repr(tile[byte + 1], width = 8)
        for bit in range(8):
            pixels.append((int(byte1[bit]) + 2*int(byte2[bit]))/3)
    pixels = np.array(pixels)
    if pooling != None:
        pool = []
        pmax = 8 // pooling + (1 if (8 % pooling) > 0 else 0)
        for py in range(pmax):
            for px in range(pmax):
                new_pixel = 0
                for y in range(pooling):
                    for x in range(pooling):
                        new_pixel = new_pixel + pixels[(px + py * pmax * 2) * pooling + x + 8 * y]
                pool.append(new_pixel / pooling ** 2)
        pixels = np.array(pool)
    return pixels

# Mapa de Memoria
    
address = {"SRAM": [0x0000,0x8000],
           "VRAM": [0x8000,0xA000],
           "XRAM": [0xA000,0xC000],
           "WRAM": [0xC000,0xE000],
           "ECHO": [0xE000,0xFE00],
           "OAM" : [0xFE00,0xFEA0],
           "I/O" : [0xFF00,0xFF80],
           "HRAM": [0xFF80,0xFFFF]}

# Interpretador de bytes

letters = {0x00:"", 0x30:"", 0x49:"", 0x4A: "PKMN", 0x4B: "_cont", 0x4C:"autocont", 0x4E:"", 0x4F:"",
           0x50:"", 0x5A:"user", 0x5B:"PC", 0x5C:"TM", 0x5D:"TRAINER", 0x5E:"ROCKET", 0x5F:"dex",
           0x60:"'", 0x61:'"', 0x6D:":", 0x7F:" ", 0x80:"A", 0x81:"B", 0x82:"C", 0x83:"D", 0x84:"E",
           0x85:"F", 0x86:"G", 0x87:"H", 0x88:"I", 0x89:"J", 0x8A:"K", 0x8B:"L", 0x8C:"M", 0x8D:"N",
           0x8E:"O", 0x8F:"P", 0x90:"Q", 0x91:"R", 0x92:"S", 0x93:"T", 0x94:"U", 0x95:"V", 0x96:"W",
           0x97:"X", 0x98:"Y", 0x99:"Z", 0x9A:"(", 0x9B:")", 0x9C:":", 0x9D:";", 0x9E:"[", 0x9F:"]",
           0xA0:"a", 0xA1:"b", 0xA2:"c", 0xA3:"d", 0xA4:"e", 0xA5:"f", 0xA6:"g", 0xA7:"h", 0xA8:"i",
           0xA9:"j", 0xAA:"k", 0xAB:"l", 0xAC:"m", 0xAD:"n", 0xAE:"o", 0xAF:"p", 0xB0:"q", 0xB1:"r",
           0xB2:"s", 0xB3:"t", 0xB4:"u", 0xB5:"v", 0xB6:"w", 0xB7:"x", 0xB8:"y", 0xB9:"z", 0xBA:"é",
           0xBB:"'d", 0xBC:"'l", 0xBD:"'s", 0xBE:"'t", 0xBF:"'v", 0xE0:"'", 0xE1:"PK", 0xE2:"MN",
           0xE3:"-", 0xE4:"'r", 0xE5:"'m", 0xE6:"?", 0xE7:"!", 0xE8:".", 0xEC:"►", 0xED:"►", 0xEE:"▼",
           0xEF:"♂", 0xF0:"¥", 0xF1:"×", 0xF2:".", 0xF3:"/", 0xF4:",", 0xF5:"♀", 0xF6:"0", 0xF7:"1",
           0xF8:"2", 0xF9:"3", 0xFA:"4", 0xFB:"5", 0xFC:"6", 0xFD:"7", 0xFE:"8", 0xFF:"9"}

class MemoryMap:
    
    def __init__(self, gameboy):
        self.gameboy = gameboy

    def mem_byte(self, address):
        if type(address) == int:
            return self.gameboy.get_memory_value(address)
        else:
            mem_array = []
            for i in range(address[0], address[1] + 1):
                mem_array.append(self.gameboy.get_memory_value(i))
            return mem_array

    def mem_bit(self, address):
        if type(address) == int:
            return np.binary_repr(self.mem_byte(address // 16), width = 8)[address % 16] if address % 16 <8 else None
        else:
            mem_array = []
            for i in range(address[0], address[1] + 1):
                if i % 16 < 8:
                    mem_array.append(np.binary_repr(self.mem_byte(i // 16), width = 8)[i % 16])
            return mem_array
    
    def get_mem_map(self):
        memory_map = np.zeros(0x10000, dtype = "int")
        for byte in range(0x10000):
            memory_map[byte] = self.gameboy.get_memory_value(byte)
        return memory_map

    def get_sram(self):
        i = "SRAM"
        sram = np.zeros(address[i][1] - address[i][0], dtype = "int")
        for byte in range(address[i][0], address[i][1]):
            sram[byte-address[i][0]] = self.gameboy.get_memory_value(byte)
        return sram
    
    def get_vram(self):
        i = "VRAM"
        vram = np.zeros(address[i][1] - address[i][0], dtype = "int")
        for byte in range(address[i][0], address[i][1]):
            vram[byte-address[i][0]] = self.gameboy.get_memory_value(byte)
        return vram

    def get_sram(self):
        i = "XRAM"
        xram = np.zeros(address[i][1] - address[i][0], dtype = "int")
        for byte in range(address[i][0], address[i][1]):
            xram[byte-address[i][0]] = self.gameboy.get_memory_value(byte)
        return xram
    
    def get_sram(self):
        i = "WRAM"
        wram = np.zeros(address[i][1] - address[i][0], dtype = "int")
        for byte in range(address[i][0], address[i][1]):
            wram[byte-address[i][0]] = self.gameboy.get_memory_value(byte)
        return wram

    def get_sram(self):
        i = "ECHO"
        echo = np.zeros(address[i][1] - address[i][0], dtype = "int")
        for byte in range(address[i][0], address[i][1]):
            echo[byte-address[i][0]] = self.gameboy.get_memory_value(byte)
        return echo

    def get_sram(self):
        i = "OAM"
        oam = np.zeros(address[i][1] - address[i][0], dtype = "int")
        for byte in range(address[i][0], address[i][1]):
            oam[byte-address[i][0]] = self.gameboy.get_memory_value(byte)
        return oam

    def get_sram(self):
        i = "I/O"
        i_o = np.zeros(address[i][1] - address[i][0], dtype = "int")
        for byte in range(address[i][0], address[i][1]):
            i_o[byte-address[i][0]] = self.gameboy.get_memory_value(byte)
        return i_o

    def get_hram(self):
        i = "HRAM"
        hram = np.zeros(address[i][1] - address[i][0], dtype = "int")
        for byte in range(address[i][0], address[i][1]):
            hram[byte-address[i][0]] = self.gameboy.get_memory_value(byte)
        return hram    
   
    def player_name(self):
        memory = range(0xD158, 0xD162 + 1)
        bytes_list = np.zeros(len(memory), dtype = "int")
        string = ""
        for byte in memory:
            bytes_list[byte - memory[0]] = self.gameboy.get_memory_value(byte)
            string += letters[bytes_list[byte-memory[0]]]
        return string
    
    def rival_name(self):
        memory = range(0xD34A, 0xD351 + 1)
        bytes_list = np.zeros(len(memory), dtype = "int")
        string = ""
        for byte in memory:
            bytes_list[byte - memory[0]] = self.gameboy.get_memory_value(byte)
            string += letters[bytes_list[byte-memory[0]]]
        return string
        
if __name__ == "__main__":
    print(letters[152])
    tile = np.random.randint(256, size=16)
    #tile = [0xE7,0xE7, 0xDB,0xD3, 0xDB,0xC3, 0xE7,0xE7, 0xE7,0xE7, 0xDB,0xD3, 0xDB,0xC3, 0xE7,0xE7]
    import matplotlib.pyplot as plt
    plt.imshow(tile_array(tile).reshape(8,8), cmap = "gray", vmin = 0, vmax = 1)
    plt.show()
    plt.imshow(tile_array(tile, 2).reshape(4,4), cmap = "gray", vmin = 0, vmax = 1)
    plt.show()
    plt.imshow(tile_array(tile, 4).reshape(2,2), cmap = "gray", vmin = 0, vmax = 1)
    plt.show()
    rgb2bit("Azul")

    
        