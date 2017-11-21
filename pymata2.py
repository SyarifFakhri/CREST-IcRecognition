import cv2
#use this to test stuff lel
from PyMata.pymata import PyMata
from PYMataICServoV2 import servoMotor


# cap = cv2.VideoCapture(0)
board = PyMata("/COM7")
board.set_pin_mode(6, board.INPUT, board.DIGITAL)

stepper = servoMotor.stepperMotor(5, PIN_STEPPER_STEP, PIN_STEPPER_DIRECTION)





