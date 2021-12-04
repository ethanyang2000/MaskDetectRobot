/*
 * MPUDriver.h
 *
 *  Created on: 2015Äê11ÔÂ26ÈÕ
 *      Author: Casey-NS-PC
 */

#ifndef MPUDRIVER_H_
#define MPUDRIVER_H_

//#define MPU6050
#define MPU6500
#define EMPL_TARGET_ARDUINO_UNO
//#define I2CDATACOMMUNICATION
#define SPIDATACOMMUNICATION



#include "Arduino.h"
#include "mpuDefine.h"
#include "dmpKey.h"
#include "dmpmap.h"
#include "dmpImage.h"

// define from inv_mpu_dmp_motion_driver.h
#define TAP_X               (0x01)
#define TAP_Y               (0x02)
#define TAP_Z               (0x04)
#define TAP_XYZ             (0x07)

#define TAP_X_UP            (0x01)
#define TAP_X_DOWN          (0x02)
#define TAP_Y_UP            (0x03)
#define TAP_Y_DOWN          (0x04)
#define TAP_Z_UP            (0x05)
#define TAP_Z_DOWN          (0x06)

#define ANDROID_ORIENT_PORTRAIT             (0x00)
#define ANDROID_ORIENT_LANDSCAPE            (0x01)
#define ANDROID_ORIENT_REVERSE_PORTRAIT     (0x02)
#define ANDROID_ORIENT_REVERSE_LANDSCAPE    (0x03)

#define DMP_INT_GESTURE     (0x01)
#define DMP_INT_CONTINUOUS  (0x02)

#define DMP_FEATURE_TAP             (0x001)
#define DMP_FEATURE_ANDROID_ORIENT  (0x002)
#define DMP_FEATURE_LP_QUAT         (0x004)
#define DMP_FEATURE_PEDOMETER       (0x008)
#define DMP_FEATURE_6X_LP_QUAT      (0x010)
#define DMP_FEATURE_GYRO_CAL        (0x020)
#define DMP_FEATURE_SEND_RAW_ACCEL  (0x040)
#define DMP_FEATURE_SEND_RAW_GYRO   (0x080)
#define DMP_FEATURE_SEND_CAL_GYRO   (0x100)

#define INV_WXYZ_QUAT       (0x100)
////////////////////////////////////////////////////////////
// end define from inv_mpu_dmp_motion_driver.h

// define from inv_mpu.h
#define INV_X_GYRO      (0x40)
#define INV_Y_GYRO      (0x20)
#define INV_Z_GYRO      (0x10)
#define INV_XYZ_GYRO    (INV_X_GYRO | INV_Y_GYRO | INV_Z_GYRO)
#define INV_XYZ_ACCEL   (0x08)
#define INV_XYZ_COMPASS (0x01)

#define MPU_INT_STATUS_DATA_READY       (0x0001)
#define MPU_INT_STATUS_DMP              (0x0002)
#define MPU_INT_STATUS_PLL_READY        (0x0004)
#define MPU_INT_STATUS_I2C_MST          (0x0008)
#define MPU_INT_STATUS_FIFO_OVERFLOW    (0x0010)
#define MPU_INT_STATUS_ZMOT             (0x0020)
#define MPU_INT_STATUS_MOT              (0x0040)
#define MPU_INT_STATUS_FREE_FALL        (0x0080)
#define MPU_INT_STATUS_DMP_0            (0x0100)
#define MPU_INT_STATUS_DMP_1            (0x0200)
#define MPU_INT_STATUS_DMP_2            (0x0400)
#define MPU_INT_STATUS_DMP_3            (0x0800)
#define MPU_INT_STATUS_DMP_4            (0x1000)
#define MPU_INT_STATUS_DMP_5            (0x2000)
////////////////////////////////////////////////////////////
// end define from inv_mpu.h
struct dmp_s {
    void (*tap_cb)(unsigned char count, unsigned char direction);
    void (*android_orient_cb)(unsigned char orientation);
    unsigned short orient;
    unsigned short feature_mask;
    unsigned short fifo_rate;
    unsigned char packet_length;
};

class MPUDriver {
public:
	MPUDriver();
	virtual ~MPUDriver();

public:

	struct dmp_s dmp;
	struct chip_cfg_s chip_cfg;

/* Driver functions for MPU process¡£ */
public:
	 /* Set up APIs */
	 uint8_t mpu_init(struct int_param_s *int_param);
	 int mpu_init_slave(void);
	 uint8_t mpu_set_bypass(unsigned char bypass_on);

	 /* Configuration APIs */
	 uint8_t mpu_lp_accel_mode(unsigned short rate);
	 uint8_t mpu_lp_motion_interrupt(unsigned short thresh, unsigned char time,
	     unsigned char lpa_freq);
	 void mpu_set_int_level(unsigned char active_low);
	 uint8_t mpu_set_int_latched(unsigned char enable);

	 uint8_t mpu_set_dmp_state(unsigned char enable);
	 void mpu_get_dmp_state(unsigned char *enabled);

	 void mpu_get_lpf(unsigned short *lpf);
	 uint8_t mpu_set_lpf(unsigned short lpf);

	 void mpu_get_gyro_fsr(unsigned short *fsr);
	 uint8_t mpu_set_gyro_fsr(unsigned short fsr);

	 uint8_t mpu_get_accel_fsr(unsigned char *fsr);
	 uint8_t mpu_set_accel_fsr(unsigned char fsr);

	 //void mpu_get_compass_fsr(unsigned short *fsr);

	 uint8_t mpu_get_gyro_sens(float *sens);
	 uint8_t mpu_get_accel_sens(unsigned short *sens);

	 uint8_t mpu_get_sample_rate(unsigned short *rate);
	 uint8_t mpu_set_sample_rate(unsigned short rate);
	 //int mpu_get_compass_sample_rate(unsigned short *rate);
	 //int mpu_set_compass_sample_rate(unsigned short rate);

	 void mpu_get_fifo_config(unsigned char *sensors);
	 uint8_t mpu_configure_fifo(unsigned char sensors);

	 void mpu_get_power_state(unsigned char *power_on);
	 uint8_t mpu_set_sensors(unsigned char sensors);

	 uint8_t mpu_read_6500_accel_bias(long *accel_bias);
	 uint8_t mpu_read_6500_gyro_bias(long *gyro_bias);
	 uint8_t mpu_set_gyro_bias_reg(long * gyro_bias);
	 uint8_t mpu_set_accel_bias_6500_reg(const long *accel_bias);
	 uint8_t mpu_read_6050_accel_bias(long *accel_bias);
	 uint8_t mpu_set_accel_bias_6050_reg(const long *accel_bias);

	 /* Data getter/setter APIs */
	 uint8_t mpu_get_gyro_reg(short *data, unsigned long *timestamp);
	 uint8_t mpu_get_accel_reg(short *data, unsigned long *timestamp);
	 //void mpu_get_compass_reg(short *data, unsigned long *timestamp);
	 uint8_t mpu_get_temperature(long *data, unsigned long *timestamp);

	 uint8_t mpu_get_int_status(short *status);
	 uint8_t mpu_read_fifo(short *gyro, short *accel, unsigned long *timestamp,
	     unsigned char *sensors, unsigned char *more);
	 uint8_t mpu_read_fifo_stream(unsigned short length, unsigned char *data,
	     unsigned char *more);
	 uint8_t mpu_reset_fifo(void);

	 uint8_t mpu_write_mem(unsigned short mem_addr, unsigned short length,
	      unsigned char *data);
	 uint8_t mpu_read_mem(unsigned short mem_addr, unsigned short length,
	     unsigned char *data);
	 uint8_t mpu_load_firmware(unsigned short length, const unsigned char *firmware,
	     unsigned short start_addr, unsigned short sample_rate);

	 //int mpu_reg_dump(void);
	 uint8_t mpu_read_reg(unsigned char reg, unsigned char *data);
	 uint8_t mpu_run_self_test(long *gyro, long *accel);
	 uint8_t mpu_run_6500_self_test(long *gyro, long *accel, unsigned char debug);
	 //int mpu_register_tap_cb(void (*func)(unsigned char, unsigned char));

	 unsigned short mpu_get_fifo_count();
	 
	 // set the interrupt of the mpu
	 uint8_t set_int_enable(unsigned char enable);
/* Drive functions for DMP. */
public:
	 /* Set up functions. */
	 uint8_t dmp_load_motion_driver_firmware(void);
	 uint8_t dmp_set_fifo_rate(unsigned short rate);
	 uint8_t dmp_get_fifo_rate(unsigned short *rate);
	 void dmp_enable_feature(unsigned short mask);
	 void dmp_get_enabled_features(unsigned short *mask);
	 uint8_t dmp_set_interrupt_mode(unsigned char mode);
	 uint8_t dmp_set_orientation(unsigned short orient);
	 uint8_t dmp_set_gyro_bias(long *bias);
	 uint8_t dmp_set_accel_bias(long *bias);

	 /* Tap functions. */
	 void dmp_register_tap_cb(void (*func)(unsigned char, unsigned char));
	 uint8_t dmp_set_tap_thresh(unsigned char axis, unsigned short thresh);
	 uint8_t dmp_set_tap_axes(unsigned char axis);
	 uint8_t dmp_set_tap_count(unsigned char min_taps);
	 uint8_t dmp_set_tap_time(unsigned short time);
	 uint8_t dmp_set_tap_time_multi(unsigned short time);
	 uint8_t dmp_set_shake_reject_thresh(long sf, unsigned short thresh);
	 uint8_t dmp_set_shake_reject_time(unsigned short time);
	 uint8_t dmp_set_shake_reject_timeout(unsigned short time);

	 /* Android orientation functions. */
	 void dmp_register_android_orient_cb(void (*func)(unsigned char));

	 /* LP quaternion functions. */
	 uint8_t dmp_enable_lp_quat(unsigned char enable);
	 uint8_t dmp_enable_6x_lp_quat(unsigned char enable);

	 /* Pedometer functions. */
	 uint8_t dmp_get_pedometer_step_count(unsigned long *count);
	 uint8_t dmp_set_pedometer_step_count(unsigned long count);
	 uint8_t dmp_get_pedometer_walk_time(unsigned long *time);
	 uint8_t dmp_set_pedometer_walk_time(unsigned long time);

	 /* DMP gyro calibration functions. */
	 uint8_t dmp_enable_gyro_cal(unsigned char enable);

	 void decode_gesture(unsigned char *gesture);

	 /* Read function. This function should be called whenever the MPU interrupt is
	  * detected.
	  */
	 uint8_t dmp_read_fifo(short *gyro, short *accel, long *quat,
	     unsigned long *timestamp, short *sensors, unsigned char *more);

};

#endif /* MPUDRIVER_H_ */
