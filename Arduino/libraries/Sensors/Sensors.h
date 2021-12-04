#ifndef MPU_DRIVER
#define MPU_DRIVER

#include <Arduino.h>
#include "MPUDriver.h"
#include <SPI.h>
#define DEFAULT_MPU_HZ 1000

class Sonic{
public:
    Sonic(int echo, int trig);
    int init();
    unsigned long check();

    int echo;
    int trig;
    float distance;
};

class my_MPU{
public:
    my_MPU(int cs = 53);
    my_MPU(const my_MPU& In_MPU);

    void InitMPU();

    int calculation();

    unsigned short inv_orientation_matrix_to_scalar(signed char *mtx);


    void eularcalcu(long * q,float * e,float * laste);

    unsigned short inv_row_2_scale(signed char *row);



    int cs;
    MPUDriver mpu;
    float eular[3];
    float lasteular[3];
    short gyro[3], accel[3], sensors;
    long quat[4];
    uint8_t read_r;
    unsigned long sensor_timestamp;
    unsigned char more;
};
#endif