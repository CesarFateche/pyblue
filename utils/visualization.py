# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 17:56:47 2020

@author: César
"""

import numpy as np
import matplotlib.pyplot as plt

# Conversión de datos de memoria en una tira de tiles

def graph(buffer, tiles_shape = (5,20), orden = True):
    
    tiles = []
    for byte in range(0,len(buffer),2):
        byte1 = np.binary_repr(int(buffer[byte]), width = 8)
        byte2 = np.binary_repr(int(buffer[byte+1]), width = 8)
        pixel = np.zeros(8)
        for bit in range(8):
            pixel[bit] = int(byte1[bit]) + int(byte2[bit]*2)
        tiles.append(pixel)
    ancho = tiles_shape[0]
    alto = tiles_shape[1]    
    graphic = np.zeros((8*alto,8*ancho))
    for x in range(ancho):
        for y in range(alto):
            if orden:
                graphic[y*8:(y+1)*8,x*8:(x+1)*8] = np.array(tiles)[8*x+ancho*8*y:8*(x+1)+ancho*8*y,:]
            else:
                graphic[y*8:(y+1)*8,x*8:(x+1)*8] = np.array(tiles)[alto*8*x+8*y:alto*8*x+8*(y+1),:]
    return graphic

# Visualización de los datos de la VRAM

def print_vram(vram, mode = "Boot"):
    
    vsprite0 = vram[:0x800]        # 128 tiles / 2048 bytes
    vsprite1 = vram[0x800:0x1000]  # 128 tiles / 2048 bytes
    vsprite2 = vram[0x1000:0x1800] # 128 tiles / 2048 bytes
    vbgmap0 = vram[0x1800:0x1C00]  #  64 tiles / 1024 bytes
    vbgmap1 = vram[0x1C00:]        #  64 tiles / 1024 bytes
    
    if mode == "Boot":
        fig, axes = plt.subplots(2,4, figsize = (10,5))
        axes[0,0].imshow(graph(vsprite0[:0x2A0], tiles_shape = (7,6), orden = False), cmap = "Blues")
        axes[1,0].imshow(graph(vsprite0[0x2A0:0x480], tiles_shape = (5,6), orden = False), cmap = "Blues")
        axes[0,1].imshow(graph(vsprite0[0x480:0x720], tiles_shape = (7,6), orden = False), cmap = "Blues")
        axes[1,1].imshow(graph(vsprite0[0x720:], tiles_shape = (7,2), orden = False), cmap = "Blues")
        axes[0,2].imshow(graph(vsprite1, tiles_shape = (8,16), orden = True), cmap = "Blues")
        axes[1,2].imshow(graph(vsprite2[:0x600], tiles_shape = (7,5), orden = False), cmap = "Blues")
        axes[0,3].imshow(graph(vsprite2[0x600:], tiles_shape = (8,4), orden = True), cmap = "Blues")
        
    if mode == "Title":
        fig, axes = plt.subplots(1,6, figsize = (10,5))
        axes[0].imshow(graph(vsprite0[:0x230], tiles_shape = (5,7), orden = True), cmap = "Blues")
        axes[1].imshow(graph(vsprite0[0x230:0x460], tiles_shape = (5,7), orden = True), cmap = "Blues")
        axes[2].imshow(graph(vsprite0[0x460:], tiles_shape = (5,7), orden = True), cmap = "Blues")
        axes[3].imshow(graph(vsprite1, tiles_shape = (16,8), orden = True), cmap = "Blues")
        axes[4].imshow(graph(vsprite2[:0x400], tiles_shape = (7,7), orden = False), cmap = "Blues")
        axes[5].imshow(graph(vsprite2[0x400:], tiles_shape = (16,4), orden = True), cmap = "Blues")
        
    if mode == "Overworld":
        fig, axes = plt.subplots(6,10, figsize = (10,6))
        axes[0,0].imshow(graph(vsprite0[:0x040], tiles_shape = (2,2), orden = True), cmap = "Blues")
        axes[0,1].imshow(graph(vsprite0[0x040:0x080], tiles_shape = (2,2), orden = True), cmap = "Blues")
        axes[0,2].imshow(graph(vsprite0[0x080:0x0C0], tiles_shape = (2,2), orden = True), cmap = "Blues")
        axes[0,3].imshow(graph(vsprite0[0x0C0:0x100], tiles_shape = (2,2), orden = True), cmap = "Blues")
        axes[0,4].imshow(graph(vsprite0[0x100:0x140], tiles_shape = (2,2), orden = True), cmap = "Blues")
        axes[0,5].imshow(graph(vsprite0[0x140:0x180], tiles_shape = (2,2), orden = True), cmap = "Blues")
        axes[0,6].imshow(graph(vsprite0[0x180:0x1C0], tiles_shape = (2,2), orden = True), cmap = "Blues")
        axes[0,7].imshow(graph(vsprite0[0x1C0:0x200], tiles_shape = (2,2), orden = True), cmap = "Blues")
        axes[0,8].imshow(graph(vsprite0[0x200:0x240], tiles_shape = (2,2), orden = True), cmap = "Blues")
        axes[0,9].imshow(graph(vsprite0[0x240:0x280], tiles_shape = (2,2), orden = True), cmap = "Blues")
        axes[1,0].imshow(graph(vsprite0[0x280:0x2C0], tiles_shape = (2,2), orden = True), cmap = "Blues")
        axes[1,1].imshow(graph(vsprite0[0x2C0:0x300], tiles_shape = (2,2), orden = True), cmap = "Blues")
        axes[1,2].imshow(graph(vsprite0[0x300:0x340], tiles_shape = (2,2), orden = True), cmap = "Blues")
        axes[1,3].imshow(graph(vsprite0[0x340:0x380], tiles_shape = (2,2), orden = True), cmap = "Blues")
        axes[1,4].imshow(graph(vsprite0[0x380:0x3C0], tiles_shape = (2,2), orden = True), cmap = "Blues")
        axes[1,5].imshow(graph(vsprite0[0x3C0:0x400], tiles_shape = (2,2), orden = True), cmap = "Blues")
        axes[1,6].imshow(graph(vsprite0[0x400:0x440], tiles_shape = (2,2), orden = True), cmap = "Blues")
        axes[1,7].imshow(graph(vsprite0[0x440:0x480], tiles_shape = (2,2), orden = True), cmap = "Blues")
        axes[1,8].imshow(graph(vsprite0[0x480:0x4C0], tiles_shape = (2,2), orden = True), cmap = "Blues")
        axes[1,9].imshow(graph(vsprite0[0x4C0:], tiles_shape = (2,2), orden = True), cmap = "Blues")
        axes[2,0].imshow(graph(vsprite1[:0x040], tiles_shape = (2,2), orden = True), cmap = "Blues")
        axes[2,1].imshow(graph(vsprite1[0x040:0x080], tiles_shape = (2,2), orden = True), cmap = "Blues")
        axes[2,2].imshow(graph(vsprite1[0x080:0x0C0], tiles_shape = (2,2), orden = True), cmap = "Blues")
        axes[2,3].imshow(graph(vsprite1[0x0C0:0x100], tiles_shape = (2,2), orden = True), cmap = "Blues")
        axes[2,4].imshow(graph(vsprite1[0x100:0x140], tiles_shape = (2,2), orden = True), cmap = "Blues")
        axes[2,5].imshow(graph(vsprite1[0x140:0x180], tiles_shape = (2,2), orden = True), cmap = "Blues")
        axes[2,6].imshow(graph(vsprite1[0x180:0x1C0], tiles_shape = (2,2), orden = True), cmap = "Blues")
        axes[2,7].imshow(graph(vsprite1[0x1C0:0x200], tiles_shape = (2,2), orden = True), cmap = "Blues")
        axes[2,8].imshow(graph(vsprite1[0x200:0x240], tiles_shape = (2,2), orden = True), cmap = "Blues")
        axes[2,9].imshow(graph(vsprite1[0x240:0x280], tiles_shape = (2,2), orden = True), cmap = "Blues")
        axes[3,0].imshow(graph(vsprite1[0x280:0x2C0], tiles_shape = (2,2), orden = True), cmap = "Blues")
        axes[3,1].imshow(graph(vsprite1[0x2C0:0x300], tiles_shape = (2,2), orden = True), cmap = "Blues")
        axes[3,2].imshow(graph(vsprite1[0x300:0x340], tiles_shape = (2,2), orden = True), cmap = "Blues")
        axes[3,3].imshow(graph(vsprite1[0x340:0x380], tiles_shape = (2,2), orden = True), cmap = "Blues")
        axes[3,4].imshow(graph(vsprite1[0x380:0x3C0], tiles_shape = (2,2), orden = True), cmap = "Blues")
        axes[3,5].imshow(graph(vsprite1[0x3C0:0x400], tiles_shape = (2,2), orden = True), cmap = "Blues")
        axes[3,6].imshow(graph(vsprite1[0x400:0x440], tiles_shape = (2,2), orden = True), cmap = "Blues")
        axes[3,7].imshow(graph(vsprite1[0x440:0x480], tiles_shape = (2,2), orden = True), cmap = "Blues")
        axes[3,8].imshow(graph(vsprite1[0x480:0x4C0], tiles_shape = (2,2), orden = True), cmap = "Blues")
        axes[3,9].imshow(graph(vsprite1[0x4C0:], tiles_shape = (2,2), orden = True), cmap = "Blues")
        axes[4,0].imshow(graph(vsprite2[:0x040], tiles_shape = (2,2), orden = True), cmap = "Blues")
        axes[4,1].imshow(graph(vsprite2[0x040:0x080], tiles_shape = (2,2), orden = True), cmap = "Blues")
        axes[4,2].imshow(graph(vsprite2[0x080:0x0C0], tiles_shape = (2,2), orden = True), cmap = "Blues")
        axes[4,3].imshow(graph(vsprite2[0x0C0:0x100], tiles_shape = (2,2), orden = True), cmap = "Blues")
        axes[4,4].imshow(graph(vsprite2[0x100:0x140], tiles_shape = (2,2), orden = True), cmap = "Blues")
        axes[4,5].imshow(graph(vsprite2[0x140:0x180], tiles_shape = (2,2), orden = True), cmap = "Blues")
        axes[4,6].imshow(graph(vsprite2[0x180:0x1C0], tiles_shape = (2,2), orden = True), cmap = "Blues")
        axes[4,7].imshow(graph(vsprite2[0x1C0:0x200], tiles_shape = (2,2), orden = True), cmap = "Blues")
        axes[4,8].imshow(graph(vsprite2[0x200:0x240], tiles_shape = (2,2), orden = True), cmap = "Blues")
        axes[4,9].imshow(graph(vsprite2[0x240:0x280], tiles_shape = (2,2), orden = True), cmap = "Blues")
        axes[5,0].imshow(graph(vsprite2[0x280:0x2C0], tiles_shape = (2,2), orden = True), cmap = "Blues")
        axes[5,1].imshow(graph(vsprite2[0x2C0:0x300], tiles_shape = (2,2), orden = True), cmap = "Blues")
        axes[5,2].imshow(graph(vsprite2[0x300:0x340], tiles_shape = (2,2), orden = True), cmap = "Blues")
        axes[5,3].imshow(graph(vsprite2[0x340:0x380], tiles_shape = (2,2), orden = True), cmap = "Blues")
        axes[5,4].imshow(graph(vsprite2[0x380:0x3C0], tiles_shape = (2,2), orden = True), cmap = "Blues")
        axes[5,5].imshow(graph(vsprite2[0x3C0:0x400], tiles_shape = (2,2), orden = True), cmap = "Blues")
        axes[5,6].imshow(graph(vsprite2[0x400:0x440], tiles_shape = (2,2), orden = True), cmap = "Blues")
        axes[5,7].imshow(graph(vsprite2[0x440:0x480], tiles_shape = (2,2), orden = True), cmap = "Blues")
        axes[5,8].imshow(graph(vsprite2[0x480:0x4C0], tiles_shape = (2,2), orden = True), cmap = "Blues")
        axes[5,9].imshow(graph(vsprite2[0x4C0:], tiles_shape = (2,2), orden = True), cmap = "Blues")