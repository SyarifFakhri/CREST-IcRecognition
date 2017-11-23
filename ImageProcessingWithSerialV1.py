import serial
from time import sleep
import ICRecognition
# flag = False
import cv2

# frameCount = 0

totalFramesToCount = 50
try:
    ser = serial.Serial('/COM6', 115200)
    # ser.timeout(0.05)
    print("connection established successfully!")
    # ser.write(str("ACK").encode())

except Exception as e:
    print('could not find')

sleep(1) #give the connection a second to settle
# ser.write((str("Hello!")).encode())

# while True:
    # print("reading")
    # data = ser.readline().strip(
    # data = line.decode
    # )
    # if data:
    #     print (data.rstrip('\n')) #strip out the new lines for now
    #     # (better to do .read() in the long run for this reason


# while (ser.inWaiting() > 0): # check if there are available ports to read

cap = cv2.VideoCapture(1)
templates = ICRecognition.getTemplate()

while True:
    ret, img = cap.read()
    cv2.imshow("image", img)

    if (ser.inWaiting() > 0):
        # ser.write(str("01").encode())  # to send string to through the UART line
        # print("sending...")
        print("reading")

        for count in range(0, totalFramesToCount):
            ret, img = cap.read()
            cv2.imshow("capture", img)

            if cv2.waitKey(1) == 27:
                break
        line = ser.readline().strip()

        try:
            values = line.decode()

            print(values)
        except:
            print("Could not read")
            values = None

        if (values == "1"):
            ##DO ALL THE IC PROCESSING STUFF HERE
            IcType = ICRecognition.getICType(img, templates)
            if IcType is None:
                IcType = 3
            print("DONE")
            ##THEN SEND THE TYPE AND IC AND SEND TO THE ARUDINO TO DO THE TYPE
            print("ICTYPE: ", IcType)
            ser.write(("1" + str(IcType)).encode())
            #first refers to the stepper and second to the type

    if cv2.waitKey(1) == 27:
        break
