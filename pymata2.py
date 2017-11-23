import cv2
#use this to test stuff lel
from PyMata.pymata import PyMata
# from PYMataICServoV2 import servoMotor
import time


# cap = cv2.VideoCapture(0)
board = PyMata("/COM7")
# board.set_pin_mode(6, board.INPUT, board.DIGITAL)
###Constants

PIN_SERVO_0 = 10
PIN_SERVO_1 = 11
PIN_SERVO_2 = 12

PIN_STEPPER_DIRECTION = 50
PIN_STEPPER_STEP = 48

PIN_IR_0 = 6
# 0 is the one at the camera
PIN_IR_1 = 7
PIN_IR_2 = 8
PIN_IR_3 = 9

#set the PIN modes
board.set_pin_mode(PIN_IR_0, board.INPUT, board.DIGITAL)
board.set_pin_mode(PIN_IR_1, board.INPUT, board.DIGITAL)
board.set_pin_mode(PIN_IR_2, board.INPUT, board.DIGITAL)
board.set_pin_mode(PIN_IR_3, board.INPUT, board.DIGITAL)

board.servo_config(PIN_SERVO_0)
board.servo_config(PIN_SERVO_1)
board.servo_config(PIN_SERVO_2)

#assume the motor is in m
# STEPPER_MOTOR_RADIUS = 1

# for x in range(0, 100):
while True:
    n = board.digital_read(PIN_IR_1)
    if n == 1:
        print("motor moving")
        board.analog_write(PIN_SERVO_0, 170)
        time.sleep(1)
        board.analog_write(PIN_SERVO_0, 80)
        time.sleep(1)
    else:
        print("motor not moving")



