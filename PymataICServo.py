from PyMata.pymata import PyMata
import time
import cv2
# import ICRecognition
import threading

# SERVO_MOTOR = 5  # servo attached to this pin

# create a PyMata instance
board = PyMata("/COM6")
from PyMata.pymata import PyMata

"""
Notes:
- IR will initialize with 0 as default, then change to read the actual value, that's why count is -1
- 1 is detected, 0 is not detected
"""
cap = cv2.VideoCapture(0)


PIN_SERVO_0 = 10
PIN_SERVO_1 = 11
PIN_SERVO_2 = 12

PIN_IR_0 = 6
#0 is the one at the camera

PIN_IR_1 = 7
PIN_IR_2 = 8
PIN_IR_3 = 9

board.set_pin_mode(PIN_IR_0, board.INPUT, board.DIGITAL)
board.set_pin_mode(PIN_IR_1, board.INPUT, board.DIGITAL)
board.set_pin_mode(PIN_IR_2, board.INPUT, board.DIGITAL)
board.set_pin_mode(PIN_IR_3, board.INPUT, board.DIGITAL)

board.servo_config(PIN_SERVO_0)
board.servo_config(PIN_SERVO_1)
board.servo_config(PIN_SERVO_2)


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


servo0 = servoMotor(PIN_SERVO_0, PIN_IR_0)
servo1 = servoMotor(PIN_SERVO_1, PIN_IR_1)
servo2 = servoMotor(PIN_SERVO_2, PIN_IR_2)

# while True:
#     ret, img = cap.read()
#     #run the stepper constantly until it detects IR at the camera as high
#
#     #then stop the stepper
#
#     #then run the camera and get the type of IC
#
#     #based on the type of IC update the ignore array etc
#
#     #Run the stepper again until it hits the IR again
#
#     servo0.turnServo()
#     servo0.removeIgnoreArray()


# Stepper motor code
board.set_pin_mode(48, board.OUTPUT, board.DIGITAL)
board.set_pin_mode(50, board.OUTPUT, board.DIGITAL)

# global countStart
# global countStop
# countStart = 0
# countStop = 0

def steps(direc, steps, speed):
    # global countStart
    # global countStop
    # for i in range(steps):
    t = threading.current_thread()
    while getattr(t, "run", True):
        board.digital_write(50, direc)
        for i in range(0, steps):
            board.digital_write(48, 1)
            time.sleep(speed)
            board.digital_write(48, 0)
            time.sleep(speed)


def test():
    while True:
        #time.sleep(1)
        ret, img = cap.read()
        # img = cv2.resize(img, (ICRecognition.widthImg, ICRecognition.heightImg), interpolation=cv2.INTER_LINEAR)
        cv2.imshow("image", img)

        key = cv2.waitKey(1)
        if key == 27:  # (escape to quit)
            cv2.destroyAllWindows()
            break

print("starting Thread")

t = threading.Thread(target=test, args=())
# t.daemon = True
t.start()

t2 = threading.Thread(target=steps, args=(1,10, 0.001))
# t2.daemon = True
t2.start()

while True:
    n = board.digital_read(7)
    print(n)
    if n == 1:
        t2.run = False
        t2.join()
        break


print("ran")




