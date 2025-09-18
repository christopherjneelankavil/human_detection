# this program is to simulate the movements of the droid in 2D environment and to generate 
# a heatmap of its movement

from ctypes import ARRAY
import numpy as np
import matplotlib.pyplot as plt

## creating a square shaped matrix (2D) to save the movement
# 0 -> visited 1 -> not visited
ARRAY_SIZE = 100

# an array to store the movement of the droid 
movementArray = np.zeros(dtype = np.short, shape = (ARRAY_SIZE, ARRAY_SIZE))

# x and y positions are set to middle
x_pos, y_pos = len(movementArray[0])//2, len(movementArray)//2

# setting the starting position as visited
movementArray[y_pos][x_pos] = 1

# input from the user about the motion of the droid
while True:

    choice = input("input : ")
    if choice == 'w':
        y_pos -= 1
        movementArray[y_pos][x_pos] = 1
    elif choice == 's':
        y_pos += 1
        movementArray[y_pos][x_pos] = 1
    elif choice == 'a':
        x_pos -= 1
        movementArray[y_pos][x_pos] = 1
    elif choice == 'd' :
        x_pos += 1
        movementArray[y_pos][x_pos] = 1
    else :
        break

plt.imshow(movementArray, cmap='hot', interpolation='nearest')
plt.show()
