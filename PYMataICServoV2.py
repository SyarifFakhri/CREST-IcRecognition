from PyMata.pymata import PyMata
import time
import cv2
import ICRecognition
import threading

lock = threading.RLock()

# SERVO_MOTOR = 5  # servo attached to this pin
cap = cv2.VideoCapture(1)
# create a PyMata instance
board = PyMata("/COM7")
from PyMata.pymata import PyMata

"""
Notes:
- IR will initialize with 0 as default, then change to read the actual value
- 1 is detected, 0 is not detected - for our program
"""

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

###End constants

#servoMotor class
class servoMotor(object):
    def __init__(self, pinServo, pinIR):
        self.pin = pinServo
        self.pinIR = pinIR

        self.count = 0
        self.ignoreArray = []

        self.previousIR = 0

        board.analog_write(self.pin, 80)


    def turnServo(self):
        n = board.digital_read(self.pinIR)
        # time.sleep(1)
        with lock:
            if self.ignoreArray == [] and n == 1:
                time.sleep(1)
                print("motor moving")
                board.analog_write(self.pin, 170)
                time.sleep(2)
                board.analog_write(self.pin, 80)
                time.sleep(2)

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
        board.digital_write(50, 0)
        # speed is in m/s

        self.delay = 1/speed
        self.stepsPerLoop = 200
        # self.run = True

        # board.stepper_config(200, [48, 50])

    def moveStepper(self):
        #will run one rotation by default - may need to change that
        #200 steps is one rotation
        #note the board will only be able to stop if it completes the for loop - that may be a problem of accurracy
        with lock:
            board.digital_write(48, 1)
            time.sleep(self.delay)
            board.digital_write(48, 0)
            time.sleep(self.delay)

        # time.sleep(0.05)
        # board.stepper_step(50, -10)
        # time.delay(0.05)

class camera(object):
    def __init__(self, camera):
        self.typeDetected = -1
        self.cameraObj = camera
        self.templates = ICRecognition.getTemplate()

    def detectIC(self):
        img = self.getImage()
        self.typeDetected = ICRecognition.getICType(img, self.templates)

    def setICType(self, t):
        self.typeDetected = t

    def show(self):
        img = self.getImage()
        cv2.imshow("ORIGINAL", img)

        key = cv2.waitKey(1)

        if key == 27:  # (escape to quit)
            cv2.destroyAllWindows()


    def getImage(self):
        __, img = self.cameraObj.read()
        return img


def stepperMotorThreadFunction(stepper):
    t = threading.current_thread()
    while True:
        while getattr(t, "run", True):
            stepper.moveStepper()

def cameraThreadFunction(camera):
    t = threading.current_thread()
    while True:
        camera.show()

        if getattr(t, "detect", True):
            camera.detectIC()
            print("IC detected is: ", camera.typeDetected)

        else:
            camera.setICType(-1)
            # print("Detect attr is false!")

        t.detect = False


def servoThreadFunction(servo):
    #this thread depends on the servo motor that you input into the thread, in theory there will be 3
    #if the servo detects an IC then it will exectue the relevant thing, either to ignore or to turn
    # t = threading.current_thread()
    while True:
        # print("turning Servo")
        servo.turnServo()

if __name__ == '__main__':
    cameraObj = camera(cap)

    #Initialize the servo motor objects
    servo0 = servoMotor(PIN_SERVO_0, PIN_IR_1)
    servo1 = servoMotor(PIN_SERVO_1, PIN_IR_2)
    servo2 = servoMotor(PIN_SERVO_2, PIN_IR_3)

    #Initialize the stepper Motor
    stepper = stepperMotor(200, PIN_STEPPER_STEP, PIN_STEPPER_DIRECTION)

    #total of 6 threads, 5 subthreads and the main thread
    stepperMotorThread = threading.Thread(target=stepperMotorThreadFunction, args=(stepper, ))
    cameraThread = threading.Thread(target=cameraThreadFunction, args=(cameraObj, ))
    servo0Thread = threading.Thread(target=servoThreadFunction, args=(servo0, ))
    # servo1Thread = threading.Thread(target=servoThreadFunction, args=(servo1, ))
    # servo2Thread = threading.Thread(target=servoThreadFunction, args=(servo2, ))

    #initialize the threads
    # we do this to initialize the IR lol - otherwise it always initializes to 0 and then the actual value
    # and that breaks things
    IRSensorReading0 = board.digital_read(PIN_IR_0)

    time.sleep(5)

    cameraThread.detect = False

    stepperMotorThread.start()
    cameraThread.start()
    servo0Thread.start()

    stepperMotorThread.run = True
    time.sleep(5)

    #main thread doubles as the sensor thread
    #start with the stepper motor, we just run that

    IRSensorReading0Prev = 1

    while True:
        IRSensorReading0 = board.digital_read(PIN_IR_0)

        # print("IR current: ", IRSensorReading0)
        # print("IR previous: ", IRSensorReading0Prev)

        #if it detects an IC, then stop the steppermotor, run the camera and get the IC type
        if IRSensorReading0 == 0 and IRSensorReading0 != IRSensorReading0Prev:
            print("IC DETECTED AT IR 0")
            stepperMotorThread.run = False
            cameraThread.detect = True
        # stepperMotorThread.join()
        # cameraThread.join()
        IRSensorReading0Prev = IRSensorReading0







