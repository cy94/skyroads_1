import math
import numpy as np
import cv2
import struct

def get_color(c,x,m):
    colors = [ [1,0,1], [0,0,1],[0,1,1],[0,1,0],[1,1,0],[1,0,0] ]
    ratio = (float(x)/m)*5
    i = math.floor(ratio)
    j = math.ceil(ratio)
    print(i , j)
    i=int(i)
    j=int(j)
    ratio -= i;
    r = (1-ratio) *colors[i][c] + ratio*colors[j][c]
    return r


def get_rgb(label,num_classes):
    offset= label*123457 % num_classes
    red = get_color(2,offset,num_classes)
    green = get_color(1,offset,num_classes)
    blue = get_color(0,offset,num_classes)
    color=[int (red*255),int(green*255),int(blue*255)]
    return color

