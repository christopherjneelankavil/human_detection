from curses import echo
import numpy as np
import time
from threading import Thread
from gpiozero import OutputDevice, PWMOutputDevice, DistanceSensor
import matplotlib.pyplot as plt


## ultrasonic sensor pins
TRIGGER_PIN = 17
ECHO_PIN = 27

# Left motor
IN3 = OutputDevice(12)   # Left motor forward
IN4 = OutputDevice(16)   # Left motor backward
IN1 = OutputDevice(20)   # Left motor forward
IN2 = OutputDevice(21)   # Left motor backward
ENB = PWMOutputDevice(18)  # Left motor speed (PWM)
ENB2 = PWMOutputDevice(13) # Left motor speed (PWM)

# Right motor
IN1_2 = OutputDevice(6)   # Right motor forward
IN2_2 = OutputDevice(5)   # Right motor backward
IN3_2 = OutputDevice(22)  # Right motor forward
IN4_2 = OutputDevice(27)  # Right motor backward
ENA_2 = PWMOutputDevice(23) # Right motor speed (PWM)
ENB_2 = PWMOutputDevice(24) # Right motor speed (PWM)

# Ultrasonic sensor
ultrasonic_sensor = DistanceSensor(echo = ECHO_PIN, trigger = TRIGGER_PIN)  

# speed settings
SPEED = 0.7
ENB.value = SPEED
ENB2.value = SPEED
ENA_2.value = SPEED
ENB_2.value = SPEED

# ultrasonic detection threshold settings
OBSTACLE_DISTANCE_THRESHOLD = 30  # unit -> cm

# current heading for updating positions
CURRENT_HEADING = 0 # 0 -> North, 90 -> East, 180 -> South, 270 -> West

# mapping for heading to coordinate changes
headingMap = {
    0 : (0, 1), # moving front value of x remains same, y increases by 1
    90 : (1, 0), # moving right value of y remains same, x increases by 1
    180 : (0, -1), # moving back value of x remains same, y decreases by 1
    270 : (-1, 0) # moving left value of y remains same, x decreases by 1
}

# an array to store the headings and movement values
movementArray = np.zeros(dtype = np.short, shape = (1000, 1000))

# x and y coordinates
# at the beginning the droid is at the center of the array
x_pos, y_pos = 500, 500
movementArray[y_pos][x_pos] = 1  # marking the starting position as visited

# TODO check if the motor pins on / off modes are correct
# Motor control functions
def move_forward():
    IN3.on()    # Left motor forward
    IN4.off()
    IN1.off()   # TODO check if this should be on or off  <--
    IN2.on()
    IN1_2.on()  # Right motor forward
    IN2_2.off()
    IN3_2.off()
    IN4_2.on()

def move_backward():
    IN3.off()   # Left motor backward
    IN4.on()
    IN1.on()
    IN2.off()
    IN1_2.off() # Right motor backward
    IN2_2.on()
    IN3_2.on()
    IN4_2.off()

def rotate_left(duration):
    IN3.off()   # Left motor backward
    IN4.on()
    IN1.on()
    IN2.off()
    IN1_2.on()  # Right motor forward
    IN2_2.off()
    IN3_2.off()
    IN4_2.on()
    time.sleep(duration)
    stop_motor()

def rotate_right(duration):
    IN3.on()    # Left motor forward
    IN4.off()
    IN1.off()
    IN2.on()
    IN1_2.off() # Right motor backward
    IN2_2.on()
    IN3_2.on()
    IN4_2.off()
    time.sleep(duration)
    stop_motor()

def stop_motor():
    IN3.off()
    IN4.off()
    IN1.off()
    IN2.off()
    IN1_2.off()
    IN2_2.off()
    IN3_2.off()
    IN4_2.off()

# function to check the obstacle
def check_obstacle():
    distance = ultrasonic_sensor.distance * 100  # to get the result in centi-meteres
    print(f'[INFO] : Current Distance : {distance:.2f} cm(s)')

    ## checking for obstacle
    if(distance < OBSTACLE_DISTANCE_THRESHOLD) :
        print('[WARN] : Obstacle Detected... Stopping the robot')
        return True
    
    return False

## navigate function to navigate the droid as a fallback method -> implementing wall following navigation

def navigate():
    global CURRENT_HEADING, x_pos, y_pos  # Declare global variables
    
    while True: # run loop forever
        
        #check whether any obstacle is found, if not move forward
        if not check_obstacle():  # returns false whenever obstacle is far away than the threshold value

            # update the current heading values
            x_pos += headingMap[CURRENT_HEADING][0]
            y_pos += headingMap[CURRENT_HEADING][1]
            movementArray[y_pos][x_pos] = 1  # y_pos is row, x_pos is column
            move_forward()
        
        # if any obstacle is detected
        # that is when the check_obstacle function returns true
        # it returns true when obstacle is near (less than the threshold value)
        
        else:
            #stop motor movement
            stop_motor()
            
            ## turning left
            print("[INFO] : Turning left 90 degrees...")
            rotate_left(5)
            CURRENT_HEADING = (CURRENT_HEADING + (360 - 90)) % 360 # updating the heading to left
            print("[INFO] : Done turning left 90 degrees.")

            # check again whether any obstacle is at the front
            if not check_obstacle(): # will return false if no obstacle is found
                continue    # continue to next iteration if no obstacle is found

            # if obstacle is still there, then turn backwards
            # turn 180 degrees left, which will make the droid face backwards
            print("[WARN] : Obstacle still detected, turning left 180 degrees...")
            rotate_left(10)
            CURRENT_HEADING = (CURRENT_HEADING + (360 - 180)) % 360 # updating the heading to backwards
            print("[INFO] : Done turning left 180 degrees.")

            # check again whether any obstacle is at the front
            if not check_obstacle(): # will return false if no obstacle is found
                continue    # continue to next iteration if no obstacle is found

            # if obstacle is still there, then turn right
            print("[WARN] : Obstacle still detected, turning right 90 degrees...")
            print("[INFO] : Droid cannot move forward\nTurning backwards...")
            rotate_right(5)
            CURRENT_HEADING = (CURRENT_HEADING + 90) % 360
            print("[INFO] : Done turning right 90 degrees.")


# main function
if __name__ == "__main__":
    try:
        #calling the navigate function to start navigating the droid -> wall following navigation
        navigate()
    # on Ctrl C
    except KeyboardInterrupt:
        stop_motor()
        print("[ERROR] : Keyboard interrupt detected")
        print("[INFO] : Stopping the motors")
        print("[INFO] : Exiting navigation...")
        print("[INFO] : Printing heatmap")
        plt.imshow(movementArray, cmap='hot', interpolation='nearest')
        plt.show()
        print("[INFO] : Program exited successfully")
        exit(0)
