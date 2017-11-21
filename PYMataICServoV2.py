from PyMata.pymata import PyMata
import time
import cv2
import ICRecognition
import threading

# SERVO_MOTOR = 5  # servo attached to this pin

# create a PyMata instance
board = PyMata("/COM7")
from PyMata.pymata import PyMata

"""
Notes:
- IR will initialize with 0 as default, then change to read the actual value
- 1 is detected, 0 is not detected - for our program
"""
cap = cv2.VideoCapture(0)

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
STEPPER_MOTOR_RADIUS = 1

###End constants

#servoMotor class
class servoMotor(object):
    def __init__(self, pinServo, pinIR):
        self.pin = pinServo
        self.pinIR = pinIR

        self.count = 0
        self.ignoreArray = []

        self.previousIR = 0

    def turnServo(self):
        n = board.digital_read(self.pinIR)

        if self.ignoreArray == [] and n == 1:
            print("motor moving")
            board.analog_write(self.pin, 180)
            time.sleep(1)
            board.analog_write(self.pin, 0)
            time.sleep(1)

    def addIgnoreArray(self):
        self.ignoreArray.append(1)

    def removeIgnoreArray(self):
        if self.ignoreArray != []:
            n = board.digital_read(self.pinIR)
            print(n)
            time.sleep(0.5)
            if n != self.previousIR:

                self.count = self.count + 1
                self.previousIR = n
                print("count increased to ", self.count)

                if self.count == 2:
                    self.ignoreArray.pop()
                    self.count = 0

class stepperMotor(object):
    def __init__(self, speed, pinStep, pinDirection):
        board.set_pin_mode(pinStep, board.OUTPUT, board.DIGITAL)
        board.set_pin_mode(pinDirection, board.OUTPUT, board.DIGITAL)
        #speed is in m/s
        self.delay = 1/speed
        self.stepsPerLoop = 10
        self.run = True

    def moveStepper(self):
        #will run one rotation by default - may need to change that
        #200 steps is one rotation
        #note the board will only be able to stop if it completes the for loop - that may be a problem of accurracy

        #while self.run != False:
        for i in range(0, self.stepsPerLoop):
            board.digital_write(48, 1)
            time.sleep(self.delay)
            board.digital_write(48, 0)
            time.sleep(self.delay)

class camera(object):



def stepperMotorThreadFunction(stepper):
    t = threading.current_thread()
    while True:
        while getattr(t, "run", True):
            stepper.moveStepper()

def cameraThreadFunction():
    t = threading.current_thread()
    while True:
        ret, img = cap.read()

        #always show the image
        cv2.imshow("camera image",img)

        if getattr(t, "detect", True):
            typeOfIC = ICRecognition.getICType(img)


def servoThreadFunction(servoMotor):
    #this thread depends on the servo motor that you input into the thread, in theory there will be 3
    #if the servo detects an IC then it will exectue the relevant thing, either to ignore or to turn

    pass

if __name__ == '__main__':
    servo0 = servoMotor(PIN_SERVO_0, PIN_IR_0)
    #Initialize the servo motor objects
    servo1 = servoMotor(PIN_SERVO_1, PIN_IR_1)
    servo2 = servoMotor(PIN_SERVO_2, PIN_IR_2)

    #Initialize the stepper Motor
    stepper = stepperMotor(300, PIN_STEPPER_STEP, PIN_STEPPER_DIRECTION)

    #total of 6 threads, 5 subthreads and the main thread
    stepperMotorThread = threading.Thread(target=stepperMotorThreadFunction, args=(stepper, ))
    cameraThread = threading.Thread(target=cameraThreadFunction, args=())
    servo0Thread = threading.Thread(target=servoThreadFunction, args=(servo0, ))
    servo1Thread = threading.Thread(target=servoThreadFunction, args=(servo1, ))
    servo2Thread = threading.Thread(target=servoThreadFunction, args=(servo2, ))

    #initialize the threads
    #we do this to initialize the IR lol - otherwise it always initializes to 0 and breaks things
    IRSensorReading0 = board.digital_read(PIN_IR_0)

    time.sleep(5)

    stepperMotorThread.start()
    cameraThread.start()

    #main thread doubles as the sensor thread
    #start with the stepper motor, we just run that
    stepperMotorThread.run = True
    cameraThread.detect = False

    while True:
        IRSensorReading0 = board.digital_read(PIN_IR_0)

        #if it detects an IC, then stop the steppermotor, run the camera and get the IC type
        if IRSensorReading0 == 0:
            stepperMotorThread.run = False





