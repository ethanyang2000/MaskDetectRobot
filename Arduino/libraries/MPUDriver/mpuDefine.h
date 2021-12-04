/*
 $License:
    Copyright (C) 2011 InvenSense Corporation, All Rights Reserved.
 $
 */
#ifndef MPUDEFINE_H__
#define MPUDEFINE_H__
#define MPU6500
#define SPIDATACOMMUNICATION
#define EMPL_TARGET_ARDUINO_UNO
#if !defined MPU6050 && !defined MPU9150 && !defined MPU6500 && !defined MPU9250
	#error  Which gyro are you using? Define MPUxxxx in your compiler options.
#endif
#include"Arduino.h"
/* Time for some messy macro work. =]
 * #define MPU9150
 * is equivalent to..
 * #define MPU6050
 * #define AK8975_SECONDARY
 *
 * #define MPU9250
 * is equivalent to..
 * #define MPU6500
 * #define AK8963_SECONDARY
 */
#if defined MPU9150
	#ifndef MPU6050
		#define MPU6050
	#endif                          /* #ifndef MPU6050 */
	#if defined AK8963_SECONDARY
		#error "MPU9150 and AK8963_SECONDARY cannot both be defined."
	#elif !defined AK8975_SECONDARY /* #if defined AK8963_SECONDARY */
		#define AK8975_SECONDARY
	#endif                          /* #if defined AK8963_SECONDARY */
#elif defined MPU9250           /* #if defined MPU9150 */
	#ifndef MPU6500
		#define MPU6500
	#endif                          /* #ifndef MPU6500 */
	#if defined AK8975_SECONDARY
		#error "MPU9250 and AK8975_SECONDARY cannot both be defined."
	#elif !defined AK8963_SECONDARY /* #if defined AK8975_SECONDARY */
		#define AK8963_SECONDARY
	#endif                          /* #if defined AK8975_SECONDARY */
#endif                          /* #if defined MPU9150 */

#if defined AK8975_SECONDARY || defined AK8963_SECONDARY
	#define AK89xx_SECONDARY
#else
	/* #warning "No compass = less profit for Invensense. Lame." */
#endif



/* When entering motion interrupt mode, the driver keeps track of the
 * previous state so that it can be restored at a later time.
 * TODO: This is tacky. Fix it.
 */
struct motion_int_cache_s {
    unsigned short gyro_fsr;
    unsigned char accel_fsr;
    unsigned short lpf;
    unsigned short sample_rate;
    unsigned char sensors_on;
    unsigned char fifo_sensors;
    unsigned char dmp_on;
};

/* Cached chip configuration data.
 * TODO: A lot of these can be handled with a bitmask.
 */
struct chip_cfg_s {
    /* Matches gyro_cfg >> 3 & 0x03 */
    unsigned char gyro_fsr;
    /* Matches accel_cfg >> 3 & 0x03 */
    unsigned char accel_fsr;
    /* Enabled sensors. Uses same masks as fifo_en, NOT pwr_mgmt_2. */
    unsigned char sensors;
    /* Matches config register. */
    unsigned char lpf;
    unsigned char clk_src;
    /* Sample rate, NOT rate divider. */
    unsigned short sample_rate;
    /* Matches fifo_en register. */
    unsigned char fifo_enable;
    /* Matches int enable register. */
    unsigned char int_enable;
    /* 1 if devices on auxiliary I2C bus appear on the primary. */
    unsigned char bypass_mode;
    /* 1 if half-sensitivity.
     * NOTE: This doesn't belong here, but everything else in hw_s is const,
     * and this allows us to save some precious RAM.
     */
    unsigned char accel_half;
    /* 1 if device in low-power accel-only mode. */
    unsigned char lp_accel_mode;
    /* 1 if interrupts are only triggered on motion events. */
    unsigned char int_motion_only;
    struct motion_int_cache_s cache;
    /* 1 for active low interrupts. */
    unsigned char active_low_int;
    /* 1 for latched interrupts. */
    unsigned char latched_int;
    /* 1 if DMP is enabled. */
    unsigned char dmp_on;
    /* Ensures that DMP will only be loaded once. */
    unsigned char dmp_loaded;
    /* Sampling rate used when DMP is enabled. */
    unsigned short dmp_sample_rate;
#ifdef AK89xx_SECONDARY
    /* Compass sample rate. */
    unsigned short compass_sample_rate;
    unsigned char compass_addr;
    short mag_sens_adj[3];
#endif
};


/* Filter configurations. */
enum lpf_e {
    INV_FILTER_256HZ_NOLPF2 = 0,
    INV_FILTER_188HZ,
    INV_FILTER_98HZ,
    INV_FILTER_42HZ,
    INV_FILTER_20HZ,
    INV_FILTER_10HZ,
    INV_FILTER_5HZ,
    INV_FILTER_2100HZ_NOLPF,
    NUM_FILTER
};

/* Full scale ranges. */
enum gyro_fsr_e {
    INV_FSR_250DPS = 0,
    INV_FSR_500DPS,
    INV_FSR_1000DPS,
    INV_FSR_2000DPS,
    NUM_GYRO_FSR
};

/* Full scale ranges. */
enum accel_fsr_e {
    INV_FSR_2G = 0,
    INV_FSR_4G,
    INV_FSR_8G,
    INV_FSR_16G,
    NUM_ACCEL_FSR
};

/* Clock sources. */
enum clock_sel_e {
    INV_CLK_INTERNAL = 0,
    INV_CLK_PLL,
    NUM_CLK
};

/* Low-power accel wakeup rates. */
enum lp_accel_rate_e {
#if defined MPU6050
    INV_LPA_1_25HZ,
    INV_LPA_5HZ,
    INV_LPA_20HZ,
    INV_LPA_40HZ
#elif defined MPU6500
    INV_LPA_0_24HZ,
    INV_LPA_0_49HZ,
    INV_LPA_0_98HZ,
    INV_LPA_1_95HZ,
    INV_LPA_3_91HZ,
    INV_LPA_7_81HZ,
    INV_LPA_15_63HZ,
    INV_LPA_31_25HZ,
    INV_LPA_62_50HZ,
    INV_LPA_125HZ,
    INV_LPA_250HZ,
    INV_LPA_500HZ
#endif
};

#define BIT_I2C_MST_VDDIO   (0x80)
#define BIT_FIFO_EN         (0x40)
#define BIT_DMP_EN          (0x80)
#define BIT_FIFO_RST        (0x04)
#define BIT_DMP_RST         (0x08)
#define BIT_FIFO_OVERFLOW   (0x10)
#define BIT_DATA_RDY_EN     (0x01)
#define BIT_DMP_INT_EN      (0x02)
#define BIT_MOT_INT_EN      (0x40)
#define BITS_FSR            (0x18)
#define BITS_LPF            (0x07)
#define BITS_HPF            (0x07)
#define BITS_CLK            (0x07)
#define BIT_FIFO_SIZE_1024  (0x40)
#define BIT_FIFO_SIZE_2048  (0x80)
#define BIT_FIFO_SIZE_4096  (0xC0)
#define BIT_RESET           (0x80)
#define BIT_SLEEP           (0x40)
#define BIT_S0_DELAY_EN     (0x01)
#define BIT_S2_DELAY_EN     (0x04)
#define BITS_SLAVE_LENGTH   (0x0F)
#define BIT_SLAVE_BYTE_SW   (0x40)
#define BIT_SLAVE_GROUP     (0x10)
#define BIT_SLAVE_EN        (0x80)
#define BIT_I2C_READ        (0x80)
#define BITS_I2C_MASTER_DLY (0x1F)
#define BIT_AUX_IF_EN       (0x20)
#define BIT_ACTL            (0x80)
#define BIT_LATCH_EN        (0x20)
#define BIT_ANY_RD_CLR      (0x10)
#define BIT_BYPASS_EN       (0x02)
#define BITS_WOM_EN         (0xC0)
#define BIT_LPA_CYCLE       (0x20)
#define BIT_STBY_XA         (0x20)
#define BIT_STBY_YA         (0x10)
#define BIT_STBY_ZA         (0x08)
#define BIT_STBY_XG         (0x04)
#define BIT_STBY_YG         (0x02)
#define BIT_STBY_ZG         (0x01)
#define BIT_STBY_XYZA       (BIT_STBY_XA | BIT_STBY_YA | BIT_STBY_ZA)
#define BIT_STBY_XYZG       (BIT_STBY_XG | BIT_STBY_YG | BIT_STBY_ZG)
#define BIT_ACCL_FC_B       (0x08)

#if defined AK8975_SECONDARY
	#define SUPPORTS_AK89xx_HIGH_SENS   (0x00)
	#define AK89xx_FSR                  (9830)
#elif defined AK8963_SECONDARY
	#define SUPPORTS_AK89xx_HIGH_SENS   (0x10)
	#define AK89xx_FSR                  (4915)
#endif

#ifdef AK89xx_SECONDARY
	#define AKM_REG_WHOAMI      (0x00)

	#define AKM_REG_ST1         (0x02)
	#define AKM_REG_HXL         (0x03)
	#define AKM_REG_ST2         (0x09)

	#define AKM_REG_CNTL        (0x0A)
	#define AKM_REG_ASTC        (0x0C)
	#define AKM_REG_ASAX        (0x10)
	#define AKM_REG_ASAY        (0x11)
	#define AKM_REG_ASAZ        (0x12)

	#define AKM_DATA_READY      (0x01)
	#define AKM_DATA_OVERRUN    (0x02)
	#define AKM_OVERFLOW        (0x80)
	#define AKM_DATA_ERROR      (0x40)

	#define AKM_BIT_SELF_TEST   (0x40)

	#define AKM_POWER_DOWN          (0x00 | SUPPORTS_AK89xx_HIGH_SENS)
	#define AKM_SINGLE_MEASUREMENT  (0x01 | SUPPORTS_AK89xx_HIGH_SENS)
	#define AKM_FUSE_ROM_ACCESS     (0x0F | SUPPORTS_AK89xx_HIGH_SENS)
	#define AKM_MODE_SELF_TEST      (0x08 | SUPPORTS_AK89xx_HIGH_SENS)

	#define AKM_WHOAMI      (0x48)
#endif

#if defined MPU6050
//const struct gyro_reg_s reg = {
	#define REG_who_am_i         0x75
	#define REG_rate_div         0x19
	#define REG_lpf              0x1A
	#define REG_prod_id          0x0C
	#define REG_user_ctrl        0x6A
	#define REG_fifo_en          0x23
	#define REG_gyro_cfg         0x1B
	#define REG_accel_cfg        0x1C
	#define REG_motion_thr       0x1F
	#define REG_motion_dur       0x20
	#define REG_fifo_count_h     0x72
	#define REG_fifo_r_w         0x74
	#define REG_raw_gyro         0x43
	#define REG_raw_accel        0x3B
	#define REG_temp             0x41
	#define REG_int_enable       0x38
	#define REG_dmp_int_status   0x39
	#define REG_int_status       0x3A
	#define REG_pwr_mgmt_1       0x6B
	#define REG_pwr_mgmt_2       0x6C
	#define REG_int_pin_cfg      0x37
	#define REG_mem_r_w          0x6F
	#define REG_accel_offs       0x06
	#define REG_i2c_mst          0x24
	#define REG_bank_sel         0x6D
	#define REG_mem_start_addr   0x6E
	#define REG_prgm_start_h     0x70
	#ifdef AK89xx_SECONDARY
		#define REG_raw_compass     0x49
		#define REG_yg_offs_tc       0x01
		#define REG_s0_addr          0x25
		#define REG_s0_reg           0x26
		#define REG_s0_ctrl          0x27
		#define REG_s1_addr          0x28
		#define REG_s1_reg           0x29
		#define REG_s1_ctrl          0x2A
		#define REG_s4_ctrl          0x34
		#define REG_s0_do            0x63
		#define REG_s1_do            0x64
		#define REG_i2c_delay_ctrl   0x67
	#endif
//};

//const struct hw_s hw = {
	#define HW_addr            0x68
	#define HW_max_fifo        1024
	#define HW_num_reg         118
	#define HW_temp_sens       340
	#define HW_temp_offset     -521
	#define HW_bank_size       256
	#if defined AK89xx_SECONDARY
		#define HW_compass_fsr     AK89xx_FSR
	#endif
//};

//const struct test_s test = {
	#define TEST_gyro_sens       32768/250
	#define TEST_accel_sens      32768/16
	#define TEST_reg_rate_div    0    /* 1kHz. */
	#define TEST_reg_lpf         1    /* 188Hz. */
	#define TEST_reg_gyro_fsr    0    /* 250dps. */
	#define TEST_reg_accel_fsr   0x18 /* 16g. */
	#define TEST_wait_ms         50
	#define TEST_packet_thresh   5    /* 5% */
	#define TEST_min_dps         10.f
	#define TEST_max_dps         105.f
	#define TEST_max_gyro_var    0.14f
	#define TEST_min_g           0.3f
	#define TEST_max_g           0.95f
	#define TEST_max_accel_var   0.14f
//};

//static struct gyro_state_s st = {
//    .reg = &reg,
//    .hw = &hw,
//    .test = &test
//};
#elif defined MPU6500
	//const struct gyro_reg_s reg = {
	#define REG_who_am_i         0x75
	#define REG_rate_div         0x19
	#define REG_lpf              0x1A
	#define REG_prod_id          0x0C
	#define REG_user_ctrl        0x6A
	#define REG_fifo_en          0x23
	#define REG_gyro_cfg         0x1B
	#define REG_accel_cfg        0x1C
	#define REG_accel_cfg2       0x1D
	#define REG_lp_accel_odr     0x1E
	#define REG_motion_thr       0x1F
	#define REG_motion_dur       0x20
	#define REG_fifo_count_h     0x72
	#define REG_fifo_r_w         0x74
	#define REG_raw_gyro         0x43
	#define REG_raw_accel        0x3B
	#define REG_temp             0x41
	#define REG_int_enable       0x38
	#define REG_dmp_int_status   0x39
	#define REG_int_status       0x3A
	#define REG_accel_intel      0x69
	#define REG_pwr_mgmt_1       0x6B
	#define REG_pwr_mgmt_2       0x6C
	#define REG_int_pin_cfg      0x37
	#define REG_mem_r_w          0x6F
	#define REG_accel_offs       0x77
	#define REG_i2c_mst          0x24
	#define REG_bank_sel         0x6D
	#define REG_mem_start_addr   0x6E
	#define REG_prgm_start_h     0x70
	#ifdef AK89xx_SECONDARY
		#define REG_raw_compass     0x49
		#define REG_s0_addr          0x25
		#define REG_s0_reg           0x26
		#define REG_s0_ctrl          0x27
		#define REG_s1_addr          0x28
		#define REG_s1_reg           0x29
		#define REG_s1_ctrl          0x2A
		#define REG_s4_ctrl          0x34
		#define REG_s0_do            0x63
		#define REG_s1_do            0x64
		#define REG_i2c_delay_ctrl   0x67
	#endif
	//};
	//const struct hw_s hw = {
	
	/*#ifdef SPIDATACOMMUNICATION
		extern unsigned char HW_addr = 10;
	#elif defined I2CDATACOMMUNICATION
		#define HW_addr            0x68
	#endif*/
	#ifdef I2CDATACOMMUNICATION
		#define HW_addr            0x68
	#endif
	
	#define HW_max_fifo        1024
	#define HW_num_reg         128
	#define HW_temp_sens       321
	#define HW_temp_offset     0
	#define HW_bank_size       256
	#if defined AK89xx_SECONDARY
		#define HW_compass_fsr     AK89xx_FSR
	#endif
	//};

	//const struct test_s test = {
	#define TEST_gyro_sens       32768/250
	#define TEST_accel_sens      32768/2  //FSR  +-2G  16384 LSB/G
	#define TEST_reg_rate_div    0    /* 1kHz#define TEST_ */
	#define TEST_reg_lpf         2    /* 92Hz low pass filter*/
	#define TEST_reg_gyro_fsr    0    /* 250dps#define TEST_ */
	#define TEST_reg_accel_fsr   0x0  /* Accel FSR setting  2g#define TEST_ */
	#define TEST_wait_ms         200   //200ms stabilization time
	#define TEST_packet_thresh   200    /* 200 samples */
	#define TEST_min_dps         20.f  //20 dps for Gyro Criteria C
	#define TEST_max_dps         60.f //Must exceed 60 dps threshold for Gyro Criteria B
	#define TEST_max_gyro_var    .5f //Must exceed +50% variation for Gyro Criteria A
	#define TEST_min_g           .225f //Accel must exceed Min 225 mg for Criteria B
	#define TEST_max_g           .675f //Accel cannot exceed Max 675 mg for Criteria B
	#define TEST_max_accel_var   .5f  //Accel must be within 50% variation for Criteria A
	#define TEST_max_g_offset    .5f   //500 mg for Accel Criteria C
	#define TEST_sample_wait_ms  10    //10ms sample time wait
	//};

	//static struct gyro_state_s st = {
	//    .reg = &reg,
	//    .hw = &hw,
	//    .test = &test
	//};
#endif

#define MAX_PACKET_LENGTH (12)
#ifdef MPU6500
	#define HWST_MAX_PACKET_LENGTH (512)
#endif

#ifdef AK89xx_SECONDARY
	static int setup_compass(void);
	#define MAX_COMPASS_SAMPLE_RATE (100)
#endif

struct int_param_s {
#if defined EMPL_TARGET_MSP430 || defined MOTION_DRIVER_TARGET_MSP430
	void (*cb)(void);
	unsigned short pin;
	unsigned char lp_exit;
	unsigned char active_low;
#elif defined EMPL_TARGET_UC3L0
	unsigned long pin;
	void (*cb)(volatile void*);
	void *arg;
#elif defined EMPL_TARGET_ARDUINO_UNO
	unsigned long pin;
	void (*cb)();
	int model; // LOW,CHANGE,RISING,FALLING
#endif
};

#ifdef EMPL_TARGET_ARDUINO_UNO
	
	#ifdef I2CDATACOMMUNICATION
		//#include "./I2Cdev/I2Cdev.h"
		#define I2CDEV_SERIAL_DEBUG
		#include "I2Cdev.h"
		//#define i2c_read(a, b, c, d) ~I2Cdev::readBytes(a, b, c, d, I2CDEV_DEFAULT_READ_TIMEOUT)
		//#define i2c_write(a, b, c, d) ~I2Cdev::writeBytes(a, b, c, d)
		static inline uint8_t i2c_write(unsigned char slave_addr, unsigned char reg_addr, unsigned char length, unsigned char *data)
		{
			if(I2Cdev::writeBytes((uint8_t)slave_addr, (uint8_t)reg_addr, (uint8_t)length, (uint8_t *)data))
				return 0; // success
			else
				return 1;
		}
		static inline uint8_t i2c_read(unsigned char slave_addr, unsigned char reg_addr, unsigned char length, unsigned char *data)
		{
			if(I2Cdev::readBytes((uint8_t)slave_addr, (uint8_t)reg_addr, (uint8_t)length, (uint8_t *)data, I2CDEV_DEFAULT_READ_TIMEOUT))
				return 0;
			else
				return 1;
		}	
	#endif

	#ifdef SPIDATACOMMUNICATION
		#include "SPI.h"
		extern unsigned char HW_addr;
		
		static inline uint8_t i2c_read(unsigned char slave_addr, unsigned char reg_addr, unsigned char length, unsigned char *data)
		{
			SPI.beginTransaction(SPISettings(10000000, MSBFIRST, SPI_MODE3));
			SPI.begin();
			digitalWrite(slave_addr, LOW);
			SPI.transfer(reg_addr | 0x80); // read
			for(uint8_t i = 0 ; i < length; i++)
			{
				data[i] = SPI.transfer(0x00);
			}
			digitalWrite(slave_addr, HIGH);
			SPI.endTransaction();
			return 0;
		}	
		static inline uint8_t i2c_write(unsigned char slave_addr, unsigned char reg_addr, unsigned char length, unsigned char *data)
		{
			SPI.beginTransaction(SPISettings(10000000, MSBFIRST, SPI_MODE3));
			SPI.begin();
			digitalWrite(slave_addr, LOW);
			SPI.transfer(reg_addr & 0x7F); // wirte
			for(uint8_t i = 0 ; i < length; i++)
			{
				SPI.transfer(data[i]);
			}
			
			digitalWrite(slave_addr, HIGH);
			SPI.endTransaction();
			return 0;
		}
	#endif
	
	#define delay_ms(a) delay(a)
	
	static inline void reg_int_cb(struct int_param_s *int_param)
	{
		// attachInterrupt(Inter No. , callback, Model) for Arduino;
		attachInterrupt(int_param->pin, int_param->cb, int_param->model);
	}

	static inline void get_ms(unsigned long *timestamp)
	{
		*timestamp = millis();
	}

#endif


#endif // MPUDEFINE_H__
