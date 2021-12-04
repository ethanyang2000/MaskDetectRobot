#include <Arduino.h>
#include"MPUDriver.h"
#include <SPI.h>
#include <Sensors.h>
#define DEFAULT_MPU_HZ 1000
unsigned char HW_addr;

Sonic::Sonic(int echo_in, int trig_in){
    this->echo = echo_in;
    this->trig = trig_in;
    this->distance = 0.0;
}

int Sonic::init(){
    pinMode(this->trig, OUTPUT);
    pinMode(this->echo, INPUT);
}

unsigned long Sonic::check(){
    digitalWrite(this->trig,  HIGH);
    delayMicroseconds(5);
    digitalWrite(this->trig, LOW);
    this->distance = pulseIn(this->echo, HIGH) / 58;
    return this->distance;
}



my_MPU::my_MPU(int inp_cs){
    this->cs = inp_cs;
    MPUDriver mpu;
}

my_MPU::my_MPU(const my_MPU& In_MPU) {
    this->cs = In_MPU.cs;
    this->mpu = In_MPU.mpu;
}

void my_MPU::InitMPU(){
    SPI.beginTransaction(SPISettings(10000000, MSBFIRST, SPI_MODE0));
    SPI.begin();
    pinMode(this->cs,OUTPUT);
    digitalWrite(this->cs,LOW);
    memset(lasteular,3,0);
    memset(eular,3,0);
    HW_addr = this->cs;
    mpu.mpu_init(0); // without interrupt
    mpu.mpu_set_sensors(INV_XYZ_GYRO | INV_XYZ_ACCEL);
    mpu.mpu_configure_fifo(INV_XYZ_GYRO | INV_XYZ_ACCEL);
    mpu.mpu_set_sample_rate(DEFAULT_MPU_HZ);
    signed char gyro_orientation[9] = { 1, 0, 0, 0, 1, 0, 0, 0, 1};
    mpu.dmp_load_motion_driver_firmware();
    mpu.dmp_set_orientation(inv_orientation_matrix_to_scalar(gyro_orientation));
    unsigned short dmp_features = DMP_FEATURE_6X_LP_QUAT | DMP_FEATURE_SEND_RAW_ACCEL | DMP_FEATURE_SEND_CAL_GYRO | DMP_FEATURE_TAP | DMP_FEATURE_GYRO_CAL;
    mpu.dmp_enable_feature(dmp_features);
    mpu.dmp_set_fifo_rate(DEFAULT_MPU_HZ);
    mpu.mpu_set_dmp_state(1);  
    long gyr_bias[3] = { -72, 12, 12};
    mpu.mpu_set_gyro_bias_reg(gyr_bias);
    delay(10);
}

int my_MPU::calculation(){
    HW_addr = this->cs;
    memset(gyro, 0, 6);
    memset(accel, 0, 6);
    memset(quat, 0, 16);
    memset(&read_r, 0, 1);

    read_r = mpu.dmp_read_fifo(gyro, accel, quat, &sensor_timestamp, &sensors, &more);

    if (!read_r && sensors)
    {
        if ((sensors & INV_XYZ_GYRO) && (sensors & INV_XYZ_ACCEL) && (sensors & INV_WXYZ_QUAT))
        {
            for(int j=0;j<4;j++) quat[j]/=100;
            long double norm=0;
            for(int i=0;i<4;i++){norm+=(long double)(quat[i]*quat[i]);}
            norm=sqrt(norm);
            if(norm>0){
            quat[0] /= norm;
            quat[1] /= norm;
            quat[2] /= norm;
            quat[3] /= norm;
            eularcalcu(quat,eular,lasteular);
            }
            else return 1;
        }
    }
    return 0;
}    

unsigned short my_MPU::inv_orientation_matrix_to_scalar(signed char *mtx){
    unsigned short scalar;
    scalar = inv_row_2_scale(mtx);
    scalar |= inv_row_2_scale(mtx + 3) << 3;
    scalar |= inv_row_2_scale(mtx + 6) << 6;
    return scalar;
}


void my_MPU::eularcalcu(long * q,float * e,float * laste){
    laste[0]=e[0];
    laste[1]=e[1];
    e[0]=atan2(2*(q[2]*q[3]+q[0]*q[1]),q[0]*q[0]-q[1]*q[1]-q[2]*q[2]+q[3]*q[3])*57.3f;
    //e[2]=asin(-2*(q[1]*q[3]-q[0]*q[2]))*57.3f;
    e[1]=atan2(2*(q[1]*q[2]+q[0]*q[3]),q[0]*q[0]+q[1]*q[1]-q[2]*q[2]-q[3]*q[3])*57.3f;
}

unsigned short my_MPU::inv_row_2_scale(signed char *row){
    unsigned short b;
    if (row[0] > 0)
        b = 0;
    else if (row[0] < 0)
        b = 4;
    else if (row[1] > 0)
        b = 1;
    else if (row[1] < 0)
        b = 5;
    else if (row[2] > 0)
        b = 2;
    else if (row[2] < 0)
        b = 6;
    else
        b = 7;      // error
    return b;
}
