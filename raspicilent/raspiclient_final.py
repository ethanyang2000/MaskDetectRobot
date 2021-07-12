import serial
import time
import vlc
import detector_tflite as detector
import subprocess
from enum import Enum
from timeit import default_timer as timer
import socket
import time
import threading
import os
from io import StringIO
import base64
from PIL import Image
from io import BytesIO
import cv2
import struct


import numpy as np


from tensorflow.lite.python.interpreter import Interpreter
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


import torch.nn as nn
import torch.nn.functional as F
import tensorflow as tf
if tf.__version__ > '2':
    import tensorflow.compat.v1 as tf  
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
#控制参数
class Args():
    def __init__(self):
        #图片分辨率
        self.photo_resolution = [2592, 1944]
        #拍照准备时间（ms）
        self.shoot_gap = "10"
        #arduino通信端口
        self.arduino_port = '/dev/ttyUSB0'
        #波特率
        self.baud_rate = 9600
        #提醒戴口罩语音路径
        self.video_warning_path = "/home/pi/Documents/maskdetector/src/maskwarning.mp3"
        #一般情况下的语音
        self.video_normal_path = "/home/pi/Documents/maskdetector/src/normal.mp3"
        #self.image_path = "/home/pi/Documents/maskdetector/src/image1"
        self.pre_audio_order = 0
        #树莓派串口
        self.serial = serial.Serial('/dev/ttyUSB0', 9600, timeout = 1)
        #待机状态    0：待机    1：运行
        self.motor_condition = 0
        #PC-raspiberry socket 对象
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        #传输图片开关    raspiberry->PC
        self.image_trans_switch = 0
        #PC执行上位机遥控的开关
        self.remote_switch = 0
        #遥控开关打开后初始的状态为停止
        self.remote_direction = "p"
        #picture tosend
        self.picture_bytes = b''
        self.socket_port = 5550
        self.socket_address = '172.20.10.2'
        self.cur_audio_player = vlc.MediaPlayer("/home/pi/Documents/maskdetector/src/normal.mp3")
args = Args()
threadLock = threading.Lock()

class Parameter(Enum):
    park = "p"
    forward_straight = "w"
    back_straight = "s"
    left_straight = "a"
    right_straight = "d"
    idle = "i"
    forward = "f"
    back = "b"
    left = "l"
    right = "r"

        
def nearest_maskless(ans):
    nearest = []
    diagonal = 0
    if ans == []:
        pass
    else :   
        for face in ans:
            if face[0] == 1:
                if diagonal < (face[4] - face[2]) * (face[4] - face[2]) + (face[5] - face[3]) * (face[5] - face[3]):
                    nearest = face
    return nearest
def separa_maskless(ans):
    masklesslist = []
    for face in ans:
        if face[0] == 1:
            masklesslist.append(face)
    return masklesslist

def adjust_target(target):
    adjust = Parameter.park.value
    if target == []:
        pass
    else :
        if abs((target[4]+target[2]) - args.photo_resolution[0]) < 330:
            if target[5] - target[3] < 700:
                adjust = Parameter.left.value
            elif target[5] - target[3] > 800:
                adjust = Parameter.right.value
            else:
                adjust = Parameter.park.value
        else :
            if (target[4]+target[2]) < args.photo_resolution[0]:
                adjust = Parameter.back.value
            else :
                adjust = Parameter.forward.value
    return adjust
            
def serial_to_arduino(order):
    if (args.serial.isOpen() == False):
        args.serial.open()

    args.serial.write(order.encode())
    #ser.close()
    
#cur_audio_player = vlc.MediaPlayer(args.video_normal_path)
#cur_audio_player.play()
def audio_play(order):
    #cur_audio_player = vlc.MediaPlayer(args.video_normal_path)
    if order == []:
        if args.pre_audio_order != 0:
            args.cur_audio_player.stop()
            args.cur_audio_player.release()
            args.cur_audio_player = vlc.MediaPlayer(args.video_normal_path)
            args.cur_audio_player.play()
            args.pre_audio_order = 0
        if (args.cur_audio_player.is_playing() == 0):
            args.cur_audio_player.play()

    else :
        if (args.pre_audio_order != 1):
            args.cur_audio_player.stop()
            args.cur_audio_player.release()
            args.cur_audio_player = vlc.MediaPlayer(args.video_warning_path)
            args.cur_audio_player.play()
            args.pre_audio_order = 1
        if (args.cur_audio_player.is_playing() == 0):
            args.cur_audio_player.play()

def sendThreadFunc():
    filepath = "/home/pi/final.jpg"
    if os.path.isfile(filepath):
        fileifo_size = struct.calcsize('128sl')
        fhead = struct.pack('128sl', bytes(os.path.basename(filepath).encode('utf-8')), os.stat(filepath).st_size)
        print(fhead)
        args.sock.send(fhead)
        print('client filepath: {0}'.format(filepath))
        fp = open(filepath, 'rb')
        while True:
            data = fp.read(1024)
            if not data:
                print('{0} file send over'.format(filepath))
                break
            args.sock.send(data)
    
def recvThreadFunc():
    while True:
        if args.image_trans_switch == 1:
            print("switch is opend!")
        else :
            print("switch is closed!")
        try:
            otherword = args.sock.recv(1024)
            #运行
            if otherword.decode() == '1':
                threadLock.acquire()
                args.motor_condition = 1
                threadLock.release()               
            #待机
            elif otherword.decode() == '2':
                print(otherword.decode())
                threadLock.acquire()
                args.motor_condition = 0
                threadLock.release()
                print(args.motor_condition)
            #开始/停止 发送图片
            elif otherword.decode() == '3':
                sendThreadFunc()
            #遥控前进
            elif otherword.decode() == '4':
                if args.remote_switch == 1:
                    args.remote_direction = "w"
            #遥控后退
            elif otherword.decode() == '5':
                if args.remote_switch == 1:
                    args.remote_direction = "s"
            #遥控左转
            elif otherword.decode() == '6': 
                if args.remote_switch == 1:
                    args.remote_direction = "a"   
            #遥控右转 
            elif otherword.decode() == '7':
                if args.remote_switch == 1:
                    args.remote_direction = "d"
            elif otherword.decode() == '8':
                if args.remote_switch == 0:
                    args.remote_switch = 1
                else :
                    args.remote_switch = 0
                if args.remote_switch == 0:
                    args.remote_direction = 'p'
            else:
                print("wrong order")
                pass
        except ConnectionAbortedError:
            print('Server closed this connection!')
 
        except ConnectionResetError:
            print('Server is closed!') 

def main_func():
    detec = detector.Detector()
    #try exception
    while True:
        if args.motor_condition == 0:
            continue
        tic = timer()
        p = subprocess.Popen(["raspistill -h %d -w %d -t %s -o -" \
            % (args.photo_resolution[0], args.photo_resolution[1], args.shoot_gap)], shell = True, stdout = subprocess.PIPE)
        raw = p.stdout.read()
        args.picture_bytes = raw
        toc = timer()
        print(toc -tic)
        ans, picture = detec.checkMask(raw)
        cv2.imwrite("/home/pi/final.jpg", picture)
        target = nearest_maskless(ans)
        audio_play(target)
        if args.remote_switch == 0:
            order = adjust_target(target)
        else :
            order = args.remote_direction
        serial_to_arduino(order)
if __name__ == "__main__":
    while (args.sock.connect_ex((args.socket_address, args.socket_port)) != 0):
        print("connection failed, anothor try after 3s······")
        time.sleep(3)
    print("connection succeeded!")
    th1 = threading.Thread(target = recvThreadFunc)
    th2 = threading.Thread(target = main_func)
    threads = [th1, th2]
 
    for t in threads:
        t.setDaemon(True)
        t.start()
    t.join()
