# MaskDetectRobot
## Introduction
This is the group project of the course *Project of Electronic Circuits*，where we have five members: @ethanyang2000, @EmberFeathers, @jupwalker, @StarryChenx, @SebastianKang5. We developed an autonomous robot that could detect and warn people who do not wear masks. The architecture of our product can be found in fig.1.
![Robot Architecture](https://github.com/ethanyang2000/MaskDetectRobot/blob/main/pics/arch.png)
And the robot is shown as fig.2.
![MaskDetectRobot](https://github.com/ethanyang2000/MaskDetectRobot/blob/main/pics/robot.png)
We completed the design and implementation of the system, including analog circuits and digital systems. Our design of the system contains an Arduino-controlled car, a Raspberry-Pi-controlled camera,and a PC software. More techniqual details can be found in our [report](https://github.com/ethanyang2000/MaskDetectRobot/blob/main/report.pdf)
## Files
```
.
|-- Arduino
|   |-- 01_01 // Codes for the arduino uno, which serves as the movement controller.
|   |   `-- 01_01.ino
|   `-- libraries // We rewrote some libraries as the driver of sensors and moters.
|       |-- MPUDriver
|       |   |-- MPUDriver.cpp
|       |   |-- MPUDriver.h
|       |   |-- dmpImage.h
|       |   |-- dmpKey.h
|       |   |-- dmpmap.h
|       |   `-- mpuDefine.h
|       |-- Move
|       |   |-- Move.cpp
|       |   `-- Move.h
|       `-- Sensors
|           |-- Sensors.cpp
|           `-- Sensors.h
|-- LICENSE
|-- README.md
|-- Server.ipynb // Codes for the PC software.
|-- pics
|   |-- arch.png
|   `-- robot.png
|-- raspicilent // Codes for the Raspberry-Pi, which processes the input of the camera, communicates with the software, and sends commands to the arduino controller.
|   |-- __pycache__
|   |   `-- detector_tflite.cpython-37.pyc
|   |-- detector_tflite.py // codes for the SSD-based objective detector.
|   |-- raspiclient_final.py // the code runner
|   |-- readme.txt
|   `-- src
|       |-- face_mask_detection.tflite // trained parameters for models.
|       |-- maskwarning.mp3
|       |-- normal.mp3
|       |-- test.mp3
|       `-- test1.mp3
`-- report.pdf

10 directories, 26 files

```

## Usage
Run `Arduino/01_01.ino` on the arduino  
Run `raspicilent/raspiclient_final.py` on Raspberry-Pi  
Run `Server.ipynb` on PC  
More details can be fould in our [report](https://github.com/ethanyang2000/MaskDetectRobot/blob/main/report.pdf)  
And we made a video for presentation:
<video id="video-sample" controls="" preload="none" poster="https://www.bilibili.com/video/BV1TB4y1N7X3">


