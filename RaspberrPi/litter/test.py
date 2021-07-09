import serial
import time
ser = serial.Serial('/dev/ttyUSB0', 9600, timeout = 1)
while True:
    ser.write(b's')
    time.sleep(1)   

ser.close()
