raspiclient文件夹下是树莓派端的文件 运行小车时需要把在树莓派端运行raspiclient_final.py
./raspiclient_final.py用于调用算法  调用摄像头获取信息  实现树莓派和arduino的串口通信  实现树莓派和PC端的socket通信
图片  声音等资源文件放在./src文件夹中  具体路径需要在./raspiclient_final.py中的args类的初始化中进行更改
./detector_tflite.py是具体的处理算法，采用的是./src/face_mask_detection.tflite的模型
树莓派需要安装TensorFlow、opencv和pytorch，可以采用.whl文件的安装，尽量避免编译安装

