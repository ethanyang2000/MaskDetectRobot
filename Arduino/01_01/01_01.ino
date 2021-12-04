/*Code:
'f''b':需要避障的前后（跟随）
‘r''l':左转/右转
'p':停止
'w''a''s''d''q''e''z''c':无需避障的自主操控
'i':巡逻
*/

#include "Move.h"
#include "Sensors.h"
#include <SPI.h>

//所有类的定义
//*********************************************
Move Core;      //小车运动模块
my_MPU mpu(53); //MPU模块

//Sonicbackward
Sonic BS(A0, A1);  //后方
Sonic BRS(A2, A3); //右后方
Sonic BLS(A4, A5); //左后方

//Sonicforward
Sonic FS(22, 23);    //前方
Sonic FLS(A8, A9);   //左前方
Sonic FRS(A10, A11); //右前方
//*********************************************
//类结束

//所有枚举型
//*********************************************
//整体小车的控制状态
enum CoreState
{
    FOLLOW, //跟随状态
    PATROL, //巡逻状态
    SLEEP,  //停止状态
    CONTROL //手动控制状态
};

enum SonicState
{
    STRIGHT,    //直行
    APPROACH,   //贴墙
    TURNRIGHT,  //自己转
    TURNLEFT,
    BETURNED    //被转
};
//*********************************************
//枚举型结束

//所有全局变量
//*********************************************

//以下为Serial
char lastRev = '0';
char Rev = '0';
char S = '0';

//以下部分为所有枚举型变量
CoreState CoreS;    //小车的控制状态
SonicState FSS;     //前进避障态
SonicState BSS;     //后退避障态

//以下为车车控制变量
int Obey = 1;   //用于判断此时车车能不能接受控制端信息，0为不能，1为能
int FC = 0; //用于前进后退的使用，0左走，1右走
int BC = 0;

//以下部分为角度控制
int setrefdelay = 1; //设定参照的delay，用于初始化参照时的延时，直到获得的值不是0的时候置为0
float ref = 0;       //直行/后退参照
int bttarget = 0;    //从蓝牙接收到的角度，设定要旋转的角度
int curtarget = 0;   //记录转弯开始时的角度
int deltatarget = 0; //记录转弯过程中每时刻的偏移角度

//以下部分为超声波传感器变量
int BLD = 100; //左后方
int BRD = 100; //右后方
int BD = 100;  //后方
int FLD = 100; //左前方
int FRD = 100; //右前方
int FD = 100;  //前方
int turnflag = 0; //是否结束转弯的标志，0为未转完，1为转完
int isturn = 0; //是否开始转弯的标志，0表示没转，1表示转了

//*********************************************
//全局变量结束

//所有函数声明
//*********************************************
void turnleft();
void turnright();
void forward();
void backward();
void Sonicfor();
void Sonicback();
void wait();
void setref();
void patrol();
//*********************************************
//函数结束
void setup()
{
    //初始化Serial
    Serial.begin(9600);
    //初始化各个枚举型
    CoreS = SLEEP;
    FSS = STRIGHT;
    BSS = STRIGHT;
    //初始化各个类
    Core.InitMove();
    mpu.InitMPU();
    FS.init();
    FLS.init();
    FRS.init();
    BS.init();
    BLS.init();
    BRS.init();
    //SPI
    SPI.beginTransaction(SPISettings(10000000, MSBFIRST, SPI_MODE0));
    SPI.begin();
    memset(mpu.lasteular, 3, 0);
    memset(mpu.eular, 3, 0);
    wait();
}

void loop()
{
    //获取初始的ref值
    while (setrefdelay)
    {
        setref();
        if (ref != 0)
        {
            setrefdelay = 0;
        }
    }
    //将curtarget设定为初始的ref值
    curtarget = ref;

    //Serial通信
    //开始接受第一个指令Rev，如果状态为跟随或巡逻态，需要接收后续指令Com，用于处理运动过程中的偏移
    if (Serial.available())
        {
            Rev = Serial.read();
        }
    if (!Obey) {
        Rev = lastRev;
    }
    else {
      lastRev = Rev;
    }
    switch(Rev){
      case'f':case'b':case'w':case'a':case's':case'd':case'i':case'p':case'q':case'e':case'z':case'c':{
        S = Rev;
      }
      default:{
        S = S;
      }
    }
    
    //状态方程        
    switch (CoreS)
    {
    case SLEEP:
    {
        if (S == 'f' || S == 'b'|| S == 'l' || S == 'r')
        {
            CoreS = FOLLOW;
        }
        else if (S == 'w' || S == 'a' || S == 's' || S == 'd' || S == 'q' || S == 'e' || S == 'z' || S == 'c')
        {
            CoreS = CONTROL;
        }
        else if (S == 'i')
        {
            CoreS = PATROL;
        }
        break;
    }
    case FOLLOW:
    {
        if (S == 'p')
        {
            CoreS = SLEEP;
        }
        else if (S == 'w' || S == 'a' || S == 's' || S == 'd' || S == 'q' || S == 'e' || S == 'z' || S == 'c')
        {
            CoreS = CONTROL;
        }
        else if (S == 'i')
        {
            CoreS = PATROL;
        }
        break;
    }
    case PATROL:
    {
        if (S == 'f' || S == 'b'|| S == 'l' || S == 'r')
        {
            CoreS = FOLLOW;
        }
        else if (S == 'w' || S == 'a' || S == 's' || S == 'd' || S == 'q' || S == 'e' || S == 'z' || S == 'c')
        {
            CoreS = CONTROL;
        }
        else if (S == 'p')
        {
            CoreS = SLEEP;
        }
        break;
    }
    case CONTROL:
    {
        if (S == 'f' || S == 'b'|| S == 'l' || S == 'r')
        {
            CoreS = FOLLOW;
        }
        else if (S == 'i')
        {
            CoreS = PATROL;
        }
        else if (S == 'p')
        {
            CoreS = SLEEP;
        }
        break;
    }
    }
    //驱动方程
    switch (CoreS)
    {
    case SLEEP:
    {
        Core.sleep();
        break;
    }
    case FOLLOW:
    {
        if (S == 'f')
        {
            Sonicfor();
        }
        else if (S == 'b')
        {
            Sonicback();
        }
        break;
    }
    case PATROL:
    {
        Core.sleep();
        break;
    }
    case CONTROL:
    {
        if (S == 'w')
        {
            if(FC == 0) {
              Core.forwardleft();
              FC = 1;
            }
            else {
              Core.forwardright();
              FC = 0;
            }
        }
        else if (S == 'a')
        {
            Core.turnleft();
        }
        else if (S == 's')
        {
            if(BC == 0) {
              Core.backwardleft();
              BC = 1;
            }
            else {
              Core.backwardright();
              BC = 0;
            }
        }
        else if (S == 'd')
        {
            Core.turnright();
        }
        else if (S == 'q')
        {
          Core.fl();
        }
        else if (S == 'e')
        {
          Core.fr();
        }
        else if (S == 'z')
        {
          Core.bl();
        }
        else if (S == 'c')
        {
          Core.br();
        }
        break;
    }
    }
}

/************************************************
【函数功能】避障前进
当遇到障碍物时，进行左转，到达安全位置后，右转
************************************************/
void Sonicfor(){
    //状态方程
    switch (FSS)
    {
    case STRIGHT:
    {
        if (FD <= 10) {
            FSS = TURNLEFT;
        }
        if (Rev == 'l' || Rev == 'r') {
            FSS = BETURNED;
        }
        break;
    }
    case APPROACH:
    {
        if (BRD >= 30) {
            FSS = TURNRIGHT;
        }
        break;
    }
    case TURNLEFT:
    {
        if (turnflag == 1) {
            FSS = APPROACH;
        }
        break;
    }
    case TURNRIGHT: 
    {
        if (turnflag == 1) {
            FSS = STRIGHT;
        }
        break;
    }
    case BETURNED:
    {
        if (turnflag == 1) {
            FSS = STRIGHT;
        }
        break;
    }
    }
    //驱动方程
    switch (FSS)
    {
    case STRIGHT:
    {
        turnflag = 0;
        isturn = 0;
        forward();
        delay(10);
        FD = FS.check();
        break;
    }
    case APPROACH:
    {
        turnflag = 0;
        isturn = 0;
        forward();
        BRD = BRS.check();
        delay(5);
        break;
    }
    case TURNLEFT:
    {
        if (isturn == 0) {
            bttarget = 65;
            isturn = 1;
            Obey = 0;
        }
        if (bttarget == 0) {
            ref += 90;
            curtarget += 90;
            isturn = 0;
            turnflag = 1;
        }
        turnleft();
        break;
    }
    case TURNRIGHT:
    {
        if (isturn == 0) {
            bttarget = 65;
            isturn = 1;
        }
        if (bttarget == 0) {
            ref -= 90;
            curtarget -= 90;
            isturn = 0;
            turnflag = 1;
            Obey = 1;
        }
        turnright();
        break;
    }
    case BETURNED:
    {
      Obey = 0;
      if(Rev == 'l') {
        turnflag = 0;
        bttarget = 65;
        ref += 90;
        while(bttarget != 0) {
          turnleft();
        }
        for (int i = 0; i < 150; i++){
          forward();
        }
        ref -= 90;
        for (int i = 0; i < 100; i++){
          forward();
        }
        turnflag = 1;
      }
      else if (Rev == 'r') {
        turnflag = 0;
        bttarget = 65;
        ref -= 90;
        while(bttarget != 0) {
          turnright();
        }
        for (int i = 0; i < 150; i++){
          forward();
        }
        ref += 90;
        for (int i = 0; i < 100; i++){
          forward();
        }
        turnflag = 1;
      }
      else {
        turnflag = 1;
      }
      Rev = '0';
      Obey = 1;
      break;
    }
    }
}

void Sonicback(){
    //状态方程
    switch (BSS)
    {
    case STRIGHT:
    {
        if (BD <= 10) {
            BSS = TURNLEFT;
        }
        if (Rev == 'l' || Rev == 'r') {
            BSS = BETURNED;
        }
        break;
    }
    case APPROACH:
    {
        if (FLD >= 30) {
            BSS = TURNRIGHT;
        }
        break;
    }
    case TURNLEFT:
    {
        if (turnflag == 1) {
            BSS = APPROACH;
        }
        break;
    }
    case TURNRIGHT: 
    {
        if (turnflag == 1) {
            BSS = STRIGHT;
        }
        break;
    }
    case BETURNED:
    {
        if (turnflag == 1) {
            BSS = STRIGHT;
        }
        break;
    }
    }
    //驱动方程
    switch (BSS)
    {
    case STRIGHT:
    {
        turnflag = 0;
        isturn = 0;
        backward();
        BD = BS.check();
        break;
    }
    case APPROACH:
    {
        turnflag = 0;
        isturn = 0;
        backward();
        FLD = FLS.check();
        delay(8);
        break;
    }
    case TURNLEFT:
    {
        if (isturn == 0) {
            bttarget = 55;
            isturn = 1;
            Obey = 0;
        }
        if (bttarget == 0) {
            isturn = 0;
            turnflag = 1;
            ref += 90;
            curtarget += 90;
        }
        turnleft();
        break;
    }
    case TURNRIGHT:
    {
        if (isturn == 0) {
            bttarget = 70;
            isturn = 1;
        }
        if (bttarget == 0) {
            ref -= 90;
            curtarget -= 90;
            isturn = 0;
            turnflag = 1;
            Obey = 1;
        }
        turnright();
        break;
    }
    case BETURNED:
    {
      Obey = 0;
        if(Rev == 'r') {
        turnflag = 0;
        bttarget = 60;
        ref += 90;
        while(bttarget != 0) {
          turnleft();
        }
        for (int i = 0; i < 150; i++){
          backward();
        }
        ref -= 90;
        for (int i = 0; i < 100; i++){
          backward();
        }
        turnflag = 1;
      }
      else if (Rev == 'l') {
        turnflag = 0;
        bttarget = 70;
        ref -= 90;
        while(bttarget != 0) {
          turnright();
        }
        for (int i = 0; i < 150; i++){
          backward();
        }
        ref += 90;
        for (int i = 0; i < 100; i++){
          backward();
        }
        turnflag = 1;
      }
      else {
        turnflag = 1;
      }
      Rev = '0';
      Obey = 1;
      break;
    }
    }
    
}
/************************************************
【函数功能】前进
根据设定的ref进行负反馈调节，
当偏右时（角度变小）向左走，当偏左时（角度变大）向右走
当偏移角度过大时，直接原地旋转至合适角度再前行
************************************************/
void forward()
{
    mpu.calculation();
    float bias = mpu.eular[1] - ref;
    if (bias > 0 && bias <= 30)
    {
        Core.fr();
    }
    else if (bias > 30 && bias <= 180)
    {
        Core.turnright();
    }
    else if (bias <= 0 && bias >= -30)
    {
        Core.fl();
    }
    else
    {
        Core.turnleft();
    }
}

/************************************************
【函数功能】后退
当偏左时(角度变大)，向左后退，当偏右时（角度变小），向右后退
************************************************/
void backward()
{
    mpu.calculation();
    float bias = mpu.eular[1] - ref;
    if (bias > 0 && bias <= 30)
    {
        Core.bl();
    }
    else if (bias > 30 && bias <= 180)
    {
        Core.turnright();
    }
    else if (bias <= 0 && bias >= -30)
    {
        Core.br();
    }
    else
    {
        Core.turnleft();
    }
}

/************************************************
【函数功能】左转一定角度
不停计算当前角度，让当前角度和原角度做差，当差值大于设定的bttarget时，停止转动
将bttarget设为0
************************************************/
void turnleft()
{
    if (bttarget != 0)
    {
        mpu.calculation();
        deltatarget = mpu.eular[1] - curtarget;
        Core.turnleft();
        if (deltatarget < -180)
        {
            deltatarget += 360;
        }
        if (deltatarget >= bttarget)
        {
            Core.sleep();
            bttarget = 0;
        }
    }
    else
    {
        Core.sleep();
    }
}

/************************************************
【函数功能】右转一定角度
和左转一样，只不过把bttarget置为负数
************************************************/
void turnright()
{
    if (bttarget > 0)
    {
        bttarget = -bttarget;
    }
    if (bttarget != 0)
    {
        mpu.calculation();
        deltatarget = mpu.eular[1] - curtarget;
        Core.turnright();
        if (deltatarget > 180)
        {
            deltatarget -= 360;
        }
        if (deltatarget <= bttarget)
        {
            Core.sleep();
            bttarget = 0;
        }
    }
    else
    {
        Core.sleep();
    }
}

/************************************************
【函数功能】设定ref
************************************************/
void setref()
{
    mpu.calculation();
    ref = mpu.eular[1];
}

/************************************************
【函数功能】等待15s让陀螺仪稳定初始化
************************************************/
void wait()
{
    delay(15000);
}

void patrol(){
    Core.sleep();
}
