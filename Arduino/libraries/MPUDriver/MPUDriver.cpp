/*
 * MPUDriver.cpp
 *
 *  Created on: 2015��11��26��
 *      Author: Casey-NS-PC
 */

#include <MPUDriver.h>
#define MPU6500

MPUDriver::MPUDriver() {
	// TODO Auto-generated constructor stub
	dmp = {NULL, NULL, 0, 0, 0, 0};
}

MPUDriver::~MPUDriver() {
	// TODO Auto-generated destructor stub
}

// Implementation for DMP process.
/**
 *  @brief  Load the DMP with this image.
 *  @return 0 if successful.
 */
uint8_t MPUDriver::dmp_load_motion_driver_firmware(void)
{
    return mpu_load_firmware(DMP_CODE_SIZE, dmp_memory, sStartAddress, DMP_SAMPLE_RATE);
}

/**
 *  @brief      Push gyro and accel orientation to the DMP.
 *  The orientation is represented here as the output of
 *  @e inv_orientation_matrix_to_scalar.
 *  @param[in]  orient  Gyro and accel orientation in body frame.
 *  @return     0 if successful.
 */
uint8_t MPUDriver::dmp_set_orientation(unsigned short orient)
{
    unsigned char gyro_regs[3], accel_regs[3];
    const unsigned char gyro_axes[3] = {DINA4C, DINACD, DINA6C};
    const unsigned char accel_axes[3] = {DINA0C, DINAC9, DINA2C};
    const unsigned char gyro_sign[3] = {DINA36, DINA56, DINA76};
    const unsigned char accel_sign[3] = {DINA26, DINA46, DINA66};

    gyro_regs[0] = gyro_axes[orient & 3];
    gyro_regs[1] = gyro_axes[(orient >> 3) & 3];
    gyro_regs[2] = gyro_axes[(orient >> 6) & 3];
    accel_regs[0] = accel_axes[orient & 3];
    accel_regs[1] = accel_axes[(orient >> 3) & 3];
    accel_regs[2] = accel_axes[(orient >> 6) & 3];

    /* Chip-to-body, axes only. */
    if (mpu_write_mem(FCFG_1, 3, gyro_regs))
        return 1;
    if (mpu_write_mem(FCFG_2, 3, accel_regs))
        return 2;

    memcpy(gyro_regs, gyro_sign, 3);
    memcpy(accel_regs, accel_sign, 3);
    if (orient & 4) {
        gyro_regs[0] |= 1;
        accel_regs[0] |= 1;
    }
    if (orient & 0x20) {
        gyro_regs[1] |= 1;
        accel_regs[1] |= 1;
    }
    if (orient & 0x100) {
        gyro_regs[2] |= 1;
        accel_regs[2] |= 1;
    }

    /* Chip-to-body, sign only. */
    if (mpu_write_mem(FCFG_3, 3, gyro_regs))
        return 3;
    if (mpu_write_mem(FCFG_7, 3, accel_regs))
        return 4;
    dmp.orient = orient;
    return 0;
}

/**
 *  @brief      Push gyro biases to the DMP.
 *  Because the gyro integration is handled in the DMP, any gyro biases
 *  calculated by the MPL should be pushed down to DMP memory to remove
 *  3-axis quaternion drift.
 *  \n NOTE: If the DMP-based gyro calibration is enabled, the DMP will
 *  overwrite the biases written to this location once a new one is computed.
 *  @param[in]  bias    Gyro biases in q16.
 *  @return     0 if successful.
 */
uint8_t MPUDriver::dmp_set_gyro_bias(long *bias)
{
    long gyro_bias_body[3];
    unsigned char regs[4];

    gyro_bias_body[0] = bias[dmp.orient & 3];
    if (dmp.orient & 4)
        gyro_bias_body[0] *= -1;
    gyro_bias_body[1] = bias[(dmp.orient >> 3) & 3];
    if (dmp.orient & 0x20)
        gyro_bias_body[1] *= -1;
    gyro_bias_body[2] = bias[(dmp.orient >> 6) & 3];
    if (dmp.orient & 0x100)
        gyro_bias_body[2] *= -1;

#ifdef EMPL_NO_64BIT
    gyro_bias_body[0] = (long)(((float)gyro_bias_body[0] * GYRO_SF) / 1073741824.f);
    gyro_bias_body[1] = (long)(((float)gyro_bias_body[1] * GYRO_SF) / 1073741824.f);
    gyro_bias_body[2] = (long)(((float)gyro_bias_body[2] * GYRO_SF) / 1073741824.f);
#else
    gyro_bias_body[0] = (long)(((long long)gyro_bias_body[0] * GYRO_SF) >> 30);
    gyro_bias_body[1] = (long)(((long long)gyro_bias_body[1] * GYRO_SF) >> 30);
    gyro_bias_body[2] = (long)(((long long)gyro_bias_body[2] * GYRO_SF) >> 30);
#endif

    regs[0] = (unsigned char)((gyro_bias_body[0] >> 24) & 0xFF);
    regs[1] = (unsigned char)((gyro_bias_body[0] >> 16) & 0xFF);
    regs[2] = (unsigned char)((gyro_bias_body[0] >> 8) & 0xFF);
    regs[3] = (unsigned char)(gyro_bias_body[0] & 0xFF);
    if (mpu_write_mem(D_EXT_GYRO_BIAS_X, 4, regs))
        return 1;

    regs[0] = (unsigned char)((gyro_bias_body[1] >> 24) & 0xFF);
    regs[1] = (unsigned char)((gyro_bias_body[1] >> 16) & 0xFF);
    regs[2] = (unsigned char)((gyro_bias_body[1] >> 8) & 0xFF);
    regs[3] = (unsigned char)(gyro_bias_body[1] & 0xFF);
    if (mpu_write_mem(D_EXT_GYRO_BIAS_Y, 4, regs))
        return 2;

    regs[0] = (unsigned char)((gyro_bias_body[2] >> 24) & 0xFF);
    regs[1] = (unsigned char)((gyro_bias_body[2] >> 16) & 0xFF);
    regs[2] = (unsigned char)((gyro_bias_body[2] >> 8) & 0xFF);
    regs[3] = (unsigned char)(gyro_bias_body[2] & 0xFF);
    return mpu_write_mem(D_EXT_GYRO_BIAS_Z, 4, regs);
}

/**
 *  @brief      Push accel biases to the DMP.
 *  These biases will be removed from the DMP 6-axis quaternion.
 *  @param[in]  bias    Accel biases in q16.
 *  @return     0 if successful.
 */
uint8_t MPUDriver::dmp_set_accel_bias(long *bias)
{
    long accel_bias_body[3];
    unsigned char regs[12];
    long long accel_sf;
    unsigned short accel_sens;

    mpu_get_accel_sens(&accel_sens);
    accel_sf = (long long)accel_sens << 15;

    //__no_operation();
    __asm__("nop\n\t");

    accel_bias_body[0] = bias[dmp.orient & 3];
    if (dmp.orient & 4)
        accel_bias_body[0] *= -1;
    accel_bias_body[1] = bias[(dmp.orient >> 3) & 3];
    if (dmp.orient & 0x20)
        accel_bias_body[1] *= -1;
    accel_bias_body[2] = bias[(dmp.orient >> 6) & 3];
    if (dmp.orient & 0x100)
        accel_bias_body[2] *= -1;

#ifdef EMPL_NO_64BIT
    accel_bias_body[0] = (long)(((float)accel_bias_body[0] * accel_sf) / 1073741824.f);
    accel_bias_body[1] = (long)(((float)accel_bias_body[1] * accel_sf) / 1073741824.f);
    accel_bias_body[2] = (long)(((float)accel_bias_body[2] * accel_sf) / 1073741824.f);
#else
    accel_bias_body[0] = (long)(((long long)accel_bias_body[0] * accel_sf) >> 30);
    accel_bias_body[1] = (long)(((long long)accel_bias_body[1] * accel_sf) >> 30);
    accel_bias_body[2] = (long)(((long long)accel_bias_body[2] * accel_sf) >> 30);
#endif

    regs[0] = (unsigned char)((accel_bias_body[0] >> 24) & 0xFF);
    regs[1] = (unsigned char)((accel_bias_body[0] >> 16) & 0xFF);
    regs[2] = (unsigned char)((accel_bias_body[0] >> 8) & 0xFF);
    regs[3] = (unsigned char)(accel_bias_body[0] & 0xFF);
    regs[4] = (unsigned char)((accel_bias_body[1] >> 24) & 0xFF);
    regs[5] = (unsigned char)((accel_bias_body[1] >> 16) & 0xFF);
    regs[6] = (unsigned char)((accel_bias_body[1] >> 8) & 0xFF);
    regs[7] = (unsigned char)(accel_bias_body[1] & 0xFF);
    regs[8] = (unsigned char)((accel_bias_body[2] >> 24) & 0xFF);
    regs[9] = (unsigned char)((accel_bias_body[2] >> 16) & 0xFF);
    regs[10] = (unsigned char)((accel_bias_body[2] >> 8) & 0xFF);
    regs[11] = (unsigned char)(accel_bias_body[2] & 0xFF);
    return mpu_write_mem(D_ACCEL_BIAS, 12, regs);
}

/**
 *  @brief      Set DMP output rate.
 *  Only used when DMP is on.
 *  @param[in]  rate    Desired fifo rate (Hz).
 *  @return     0 if successful.
 */
uint8_t MPUDriver::dmp_set_fifo_rate(unsigned short rate)
{
    const unsigned char regs_end[12] = {DINAFE, DINAF2, DINAAB,
        0xc4, DINAAA, DINAF1, DINADF, DINADF, 0xBB, 0xAF, DINADF, DINADF};
    unsigned short div;
    unsigned char tmp[8];

    if (rate > DMP_SAMPLE_RATE)
        return 1;
    div = DMP_SAMPLE_RATE / rate - 1;
    tmp[0] = (unsigned char)((div >> 8) & 0xFF);
    tmp[1] = (unsigned char)(div & 0xFF);
    if (mpu_write_mem(D_0_22, 2, tmp))
        return 2;
    if (mpu_write_mem(CFG_6, 12, (unsigned char*)regs_end))
        return 3;

    dmp.fifo_rate = rate;
    return 0;
}

/**
 *  @brief      Get DMP output rate.
 *  @param[out] rate    Current fifo rate (Hz).
 *  @return     0 if successful.
 */
uint8_t MPUDriver::dmp_get_fifo_rate(unsigned short *rate)
{
    rate[0] = dmp.fifo_rate;
    return 0;
}

/**
 *  @brief      Set tap threshold for a specific axis.
 *  @param[in]  axis    1, 2, and 4 for XYZ accel, respectively.
 *  @param[in]  thresh  Tap threshold, in mg/ms.
 *  @return     0 if successful.
 */
uint8_t MPUDriver::dmp_set_tap_thresh(unsigned char axis, unsigned short thresh)
{
    unsigned char tmp[4], accel_fsr;
    float scaled_thresh;
    unsigned short dmp_thresh, dmp_thresh_2;
    if (!(axis & TAP_XYZ) || thresh > 1600)
        return 1;

    scaled_thresh = (float)thresh / DMP_SAMPLE_RATE;

    mpu_get_accel_fsr(&accel_fsr);
    switch (accel_fsr) {
    case 2:
        dmp_thresh = (unsigned short)(scaled_thresh * 16384);
        /* dmp_thresh * 0.75 */
        dmp_thresh_2 = (unsigned short)(scaled_thresh * 12288);
        break;
    case 4:
        dmp_thresh = (unsigned short)(scaled_thresh * 8192);
        /* dmp_thresh * 0.75 */
        dmp_thresh_2 = (unsigned short)(scaled_thresh * 6144);
        break;
    case 8:
        dmp_thresh = (unsigned short)(scaled_thresh * 4096);
        /* dmp_thresh * 0.75 */
        dmp_thresh_2 = (unsigned short)(scaled_thresh * 3072);
        break;
    case 16:
        dmp_thresh = (unsigned short)(scaled_thresh * 2048);
        /* dmp_thresh * 0.75 */
        dmp_thresh_2 = (unsigned short)(scaled_thresh * 1536);
        break;
    default:
        return 2;
    }
    tmp[0] = (unsigned char)(dmp_thresh >> 8);
    tmp[1] = (unsigned char)(dmp_thresh & 0xFF);
    tmp[2] = (unsigned char)(dmp_thresh_2 >> 8);
    tmp[3] = (unsigned char)(dmp_thresh_2 & 0xFF);

    if (axis & TAP_X) {
        if (mpu_write_mem(DMP_TAP_THX, 2, tmp))
            return 3;
        if (mpu_write_mem(D_1_36, 2, tmp+2))
            return 4;
    }
    if (axis & TAP_Y) {
        if (mpu_write_mem(DMP_TAP_THY, 2, tmp))
            return 5;
        if (mpu_write_mem(D_1_40, 2, tmp+2))
            return 6;
    }
    if (axis & TAP_Z) {
        if (mpu_write_mem(DMP_TAP_THZ, 2, tmp))
            return 7;
        if (mpu_write_mem(D_1_44, 2, tmp+2))
            return 8;
    }
    return 0;
}

/**
 *  @brief      Set which axes will register a tap.
 *  @param[in]  axis    1, 2, and 4 for XYZ, respectively.
 *  @return     0 if successful.
 */
uint8_t MPUDriver::dmp_set_tap_axes(unsigned char axis)
{
    unsigned char tmp = 0;

    if (axis & TAP_X)
        tmp |= 0x30;
    if (axis & TAP_Y)
        tmp |= 0x0C;
    if (axis & TAP_Z)
        tmp |= 0x03;
    return mpu_write_mem(D_1_72, 1, &tmp);
}

/**
 *  @brief      Set minimum number of taps needed for an interrupt.
 *  @param[in]  min_taps    Minimum consecutive taps (1-4).
 *  @return     0 if successful.
 */
uint8_t MPUDriver::dmp_set_tap_count(unsigned char min_taps)
{
    unsigned char tmp;

    if (min_taps < 1)
        min_taps = 1;
    else if (min_taps > 4)
        min_taps = 4;

    tmp = min_taps - 1;
    return mpu_write_mem(D_1_79, 1, &tmp);
}

/**
 *  @brief      Set length between valid taps.
 *  @param[in]  time    Milliseconds between taps.
 *  @return     0 if successful.
 */
uint8_t MPUDriver::dmp_set_tap_time(unsigned short time)
{
    unsigned short dmp_time;
    unsigned char tmp[2];

    dmp_time = time / (1000 / DMP_SAMPLE_RATE);
    tmp[0] = (unsigned char)(dmp_time >> 8);
    tmp[1] = (unsigned char)(dmp_time & 0xFF);
    return mpu_write_mem(DMP_TAPW_MIN, 2, tmp);
}

/**
 *  @brief      Set max time between taps to register as a multi-tap.
 *  @param[in]  time    Max milliseconds between taps.
 *  @return     0 if successful.
 */
uint8_t MPUDriver::dmp_set_tap_time_multi(unsigned short time)
{
    unsigned short dmp_time;
    unsigned char tmp[2];

    dmp_time = time / (1000 / DMP_SAMPLE_RATE);
    tmp[0] = (unsigned char)(dmp_time >> 8);
    tmp[1] = (unsigned char)(dmp_time & 0xFF);
    return mpu_write_mem(D_1_218, 2, tmp);
}

/**
 *  @brief      Set shake rejection threshold.
 *  If the DMP detects a gyro sample larger than @e thresh, taps are rejected.
 *  @param[in]  sf      Gyro scale factor.
 *  @param[in]  thresh  Gyro threshold in dps.
 *  @return     0 if successful.
 */
uint8_t MPUDriver::dmp_set_shake_reject_thresh(long sf, unsigned short thresh)
{
    unsigned char tmp[4];
    long thresh_scaled = sf / 1000 * thresh;
    tmp[0] = (unsigned char)(((long)thresh_scaled >> 24) & 0xFF);
    tmp[1] = (unsigned char)(((long)thresh_scaled >> 16) & 0xFF);
    tmp[2] = (unsigned char)(((long)thresh_scaled >> 8) & 0xFF);
    tmp[3] = (unsigned char)((long)thresh_scaled & 0xFF);
    return mpu_write_mem(D_1_92, 4, tmp);
}

/**
 *  @brief      Set shake rejection time.
 *  Sets the length of time that the gyro must be outside of the threshold set
 *  by @e gyro_set_shake_reject_thresh before taps are rejected. A mandatory
 *  60 ms is added to this parameter.
 *  @param[in]  time    Time in milliseconds.
 *  @return     0 if successful.
 */
uint8_t MPUDriver::dmp_set_shake_reject_time(unsigned short time)
{
    unsigned char tmp[2];

    time /= (1000 / DMP_SAMPLE_RATE);
    tmp[0] = time >> 8;
    tmp[1] = time & 0xFF;
    return mpu_write_mem(D_1_90,2,tmp);
}

/**
 *  @brief      Set shake rejection timeout.
 *  Sets the length of time after a shake rejection that the gyro must stay
 *  inside of the threshold before taps can be detected again. A mandatory
 *  60 ms is added to this parameter.
 *  @param[in]  time    Time in milliseconds.
 *  @return     0 if successful.
 */
uint8_t MPUDriver::dmp_set_shake_reject_timeout(unsigned short time)
{
    unsigned char tmp[2];

    time /= (1000 / DMP_SAMPLE_RATE);
    tmp[0] = time >> 8;
    tmp[1] = time & 0xFF;
    return mpu_write_mem(D_1_88,2,tmp);
}

/**
 *  @brief      Get current step count.
 *  @param[out] count   Number of steps detected.
 *  @return     0 if successful.
 */
uint8_t MPUDriver::dmp_get_pedometer_step_count(unsigned long *count)
{
    unsigned char tmp[4];
    if (!count)
        return 1;

    if (mpu_read_mem(D_PEDSTD_STEPCTR, 4, tmp))
        return 2;

    count[0] = ((unsigned long)tmp[0] << 24) | ((unsigned long)tmp[1] << 16) |
        ((unsigned long)tmp[2] << 8) | tmp[3];
    return 0;
}

/**
 *  @brief      Overwrite current step count.
 *  WARNING: This function writes to DMP memory and could potentially encounter
 *  a race condition if called while the pedometer is enabled.
 *  @param[in]  count   New step count.
 *  @return     0 if successful.
 */
uint8_t MPUDriver::dmp_set_pedometer_step_count(unsigned long count)
{
    unsigned char tmp[4];

    tmp[0] = (unsigned char)((count >> 24) & 0xFF);
    tmp[1] = (unsigned char)((count >> 16) & 0xFF);
    tmp[2] = (unsigned char)((count >> 8) & 0xFF);
    tmp[3] = (unsigned char)(count & 0xFF);
    return mpu_write_mem(D_PEDSTD_STEPCTR, 4, tmp);
}

/**
 *  @brief      Get duration of walking time.
 *  @param[in]  time    Walk time in milliseconds.
 *  @return     0 if successful.
 */
uint8_t MPUDriver::dmp_get_pedometer_walk_time(unsigned long *time)
{
    unsigned char tmp[4];
    if (!time)
        return 1;

    if (mpu_read_mem(D_PEDSTD_TIMECTR, 4, tmp))
        return 2;

    time[0] = (((unsigned long)tmp[0] << 24) | ((unsigned long)tmp[1] << 16) |
        ((unsigned long)tmp[2] << 8) | tmp[3]) * 20;
    return 0;
}

/**
 *  @brief      Overwrite current walk time.
 *  WARNING: This function writes to DMP memory and could potentially encounter
 *  a race condition if called while the pedometer is enabled.
 *  @param[in]  time    New walk time in milliseconds.
 */
uint8_t MPUDriver::dmp_set_pedometer_walk_time(unsigned long time)
{
    unsigned char tmp[4];

    time /= 20;

    tmp[0] = (unsigned char)((time >> 24) & 0xFF);
    tmp[1] = (unsigned char)((time >> 16) & 0xFF);
    tmp[2] = (unsigned char)((time >> 8) & 0xFF);
    tmp[3] = (unsigned char)(time & 0xFF);
    return mpu_write_mem(D_PEDSTD_TIMECTR, 4, tmp);
}

/**
 *  @brief      Enable DMP features.
 *  The following \#define's are used in the input mask:
 *  \n DMP_FEATURE_TAP
 *  \n DMP_FEATURE_ANDROID_ORIENT
 *  \n DMP_FEATURE_LP_QUAT
 *  \n DMP_FEATURE_6X_LP_QUAT
 *  \n DMP_FEATURE_GYRO_CAL
 *  \n DMP_FEATURE_SEND_RAW_ACCEL
 *  \n DMP_FEATURE_SEND_RAW_GYRO
 *  \n NOTE: DMP_FEATURE_LP_QUAT and DMP_FEATURE_6X_LP_QUAT are mutually
 *  exclusive.
 *  \n NOTE: DMP_FEATURE_SEND_RAW_GYRO and DMP_FEATURE_SEND_CAL_GYRO are also
 *  mutually exclusive.
 *  @param[in]  mask    Mask of features to enable.
 *  @return     0 if successful.
 */
void MPUDriver::dmp_enable_feature(unsigned short mask)
{
    unsigned char tmp[10];

    /* TODO: All of these settings can probably be integrated into the default
     * DMP image.
     */
    /* Set integration scale factor. */
    tmp[0] = (unsigned char)((GYRO_SF >> 24) & 0xFF);
    tmp[1] = (unsigned char)((GYRO_SF >> 16) & 0xFF);
    tmp[2] = (unsigned char)((GYRO_SF >> 8) & 0xFF);
    tmp[3] = (unsigned char)(GYRO_SF & 0xFF);
    mpu_write_mem(D_0_104, 4, tmp);

    /* Send sensor data to the FIFO. */
    tmp[0] = 0xA3;
    if (mask & DMP_FEATURE_SEND_RAW_ACCEL) {
        tmp[1] = 0xC0;
        tmp[2] = 0xC8;
        tmp[3] = 0xC2;
    } else {
        tmp[1] = 0xA3;
        tmp[2] = 0xA3;
        tmp[3] = 0xA3;
    }
    if (mask & DMP_FEATURE_SEND_ANY_GYRO) {
        tmp[4] = 0xC4;
        tmp[5] = 0xCC;
        tmp[6] = 0xC6;
    } else {
        tmp[4] = 0xA3;
        tmp[5] = 0xA3;
        tmp[6] = 0xA3;
    }
    tmp[7] = 0xA3;
    tmp[8] = 0xA3;
    tmp[9] = 0xA3;
    mpu_write_mem(CFG_15,10,tmp);

    /* Send gesture data to the FIFO. */
    if (mask & (DMP_FEATURE_TAP | DMP_FEATURE_ANDROID_ORIENT))
        tmp[0] = DINA20;
    else
        tmp[0] = 0xD8;
    mpu_write_mem(CFG_27,1,tmp);

    if (mask & DMP_FEATURE_GYRO_CAL)
        dmp_enable_gyro_cal(1);
    else
        dmp_enable_gyro_cal(0);

    if (mask & DMP_FEATURE_SEND_ANY_GYRO) {
        if (mask & DMP_FEATURE_SEND_CAL_GYRO) {
            tmp[0] = 0xB2;
            tmp[1] = 0x8B;
            tmp[2] = 0xB6;
            tmp[3] = 0x9B;
        } else {
            tmp[0] = DINAC0;
            tmp[1] = DINA80;
            tmp[2] = DINAC2;
            tmp[3] = DINA90;
        }
        mpu_write_mem(CFG_GYRO_RAW_DATA, 4, tmp);
    }

    if (mask & DMP_FEATURE_TAP) {
        /* Enable tap. */
        tmp[0] = 0xF8;
        mpu_write_mem(CFG_20, 1, tmp);
        dmp_set_tap_thresh(TAP_XYZ, 250);
        dmp_set_tap_axes(TAP_XYZ);
        dmp_set_tap_count(1);
        dmp_set_tap_time(100);
        dmp_set_tap_time_multi(500);

        dmp_set_shake_reject_thresh(GYRO_SF, 200);
        dmp_set_shake_reject_time(40);
        dmp_set_shake_reject_timeout(10);
    } else {
        tmp[0] = 0xD8;
        mpu_write_mem(CFG_20, 1, tmp);
    }

    if (mask & DMP_FEATURE_ANDROID_ORIENT) {
        tmp[0] = 0xD9;
    } else
        tmp[0] = 0xD8;
    mpu_write_mem(CFG_ANDROID_ORIENT_INT, 1, tmp);

    if (mask & DMP_FEATURE_LP_QUAT)
        dmp_enable_lp_quat(1);
    else
        dmp_enable_lp_quat(0);

    if (mask & DMP_FEATURE_6X_LP_QUAT)
        dmp_enable_6x_lp_quat(1);
    else
        dmp_enable_6x_lp_quat(0);

    /* Pedometer is always enabled. */
    dmp.feature_mask = mask | DMP_FEATURE_PEDOMETER;
    mpu_reset_fifo();

    dmp.packet_length = 0;
    if (mask & DMP_FEATURE_SEND_RAW_ACCEL)
        dmp.packet_length += 6;
    if (mask & DMP_FEATURE_SEND_ANY_GYRO)
        dmp.packet_length += 6;
    if (mask & (DMP_FEATURE_LP_QUAT | DMP_FEATURE_6X_LP_QUAT))
        dmp.packet_length += 16;
    if (mask & (DMP_FEATURE_TAP | DMP_FEATURE_ANDROID_ORIENT))
        dmp.packet_length += 4;
}

/**
 *  @brief      Get list of currently enabled DMP features.
 *  @param[out] Mask of enabled features.
 *  @return     0 if successful.
 */
void MPUDriver::dmp_get_enabled_features(unsigned short *mask)
{
    mask[0] = dmp.feature_mask;
}

/**
 *  @brief      Calibrate the gyro data in the DMP.
 *  After eight seconds of no motion, the DMP will compute gyro biases and
 *  subtract them from the quaternion output. If @e dmp_enable_feature is
 *  called with @e DMP_FEATURE_SEND_CAL_GYRO, the biases will also be
 *  subtracted from the gyro output.
 *  @param[in]  enable  1 to enable gyro calibration.
 *  @return     0 if successful.
 */
uint8_t MPUDriver::dmp_enable_gyro_cal(unsigned char enable)
{
    if (enable) {
        unsigned char regs[9] = {0xb8, 0xaa, 0xb3, 0x8d, 0xb4, 0x98, 0x0d, 0x35, 0x5d};
        return mpu_write_mem(CFG_MOTION_BIAS, 9, regs);
    } else {
        unsigned char regs[9] = {0xb8, 0xaa, 0xaa, 0xaa, 0xb0, 0x88, 0xc3, 0xc5, 0xc7};
        return mpu_write_mem(CFG_MOTION_BIAS, 9, regs);
    }
}

/**
 *  @brief      Generate 3-axis quaternions from the DMP.
 *  In this driver, the 3-axis and 6-axis DMP quaternion features are mutually
 *  exclusive.
 *  @param[in]  enable  1 to enable 3-axis quaternion.
 *  @return     0 if successful.
 */
uint8_t MPUDriver::dmp_enable_lp_quat(unsigned char enable)
{
    unsigned char regs[4];
    if (enable) {
        regs[0] = DINBC0;
        regs[1] = DINBC2;
        regs[2] = DINBC4;
        regs[3] = DINBC6;
    }
    else
        memset(regs, 0x8B, 4);

    mpu_write_mem(CFG_LP_QUAT, 4, regs);

    return mpu_reset_fifo();
}

/**
 *  @brief       Generate 6-axis quaternions from the DMP.
 *  In this driver, the 3-axis and 6-axis DMP quaternion features are mutually
 *  exclusive.
 *  @param[in]   enable  1 to enable 6-axis quaternion.
 *  @return      0 if successful.
 */
uint8_t MPUDriver::dmp_enable_6x_lp_quat(unsigned char enable)
{
    unsigned char regs[4];
    if (enable) {
        regs[0] = DINA20;
        regs[1] = DINA28;
        regs[2] = DINA30;
        regs[3] = DINA38;
    } else
        memset(regs, 0xA3, 4);

    mpu_write_mem(CFG_8, 4, regs);

    return mpu_reset_fifo();
}

/**
 *  @brief      Decode the four-byte gesture data and execute any callbacks.
 *  @param[in]  gesture Gesture data from DMP packet.
 *  @return     0 if successful.
 */
void MPUDriver::decode_gesture(unsigned char *gesture)
{
    unsigned char tap, android_orient;

    android_orient = gesture[3] & 0xC0;
    tap = 0x3F & gesture[3];

    if (gesture[1] & INT_SRC_TAP) {
        unsigned char direction, count;
        direction = tap >> 3;
        count = (tap % 8) + 1;
        if (dmp.tap_cb)
            dmp.tap_cb(direction, count);
    }

    if (gesture[1] & INT_SRC_ANDROID_ORIENT) {
        if (dmp.android_orient_cb)
            dmp.android_orient_cb(android_orient >> 6);
    }
}

/**
 *  @brief      Specify when a DMP interrupt should occur.
 *  A DMP interrupt can be configured to trigger on either of the two
 *  conditions below:
 *  \n a. One FIFO period has elapsed (set by @e mpu_set_sample_rate).
 *  \n b. A tap event has been detected.
 *  @param[in]  mode    DMP_INT_GESTURE or DMP_INT_CONTINUOUS.
 *  @return     0 if successful.
 */
uint8_t MPUDriver::dmp_set_interrupt_mode(unsigned char mode)
{
    const unsigned char regs_continuous[11] =
        {0xd8, 0xb1, 0xb9, 0xf3, 0x8b, 0xa3, 0x91, 0xb6, 0x09, 0xb4, 0xd9};
    const unsigned char regs_gesture[11] =
        {0xda, 0xb1, 0xb9, 0xf3, 0x8b, 0xa3, 0x91, 0xb6, 0xda, 0xb4, 0xda};

    switch (mode) {
    case DMP_INT_CONTINUOUS:
        return mpu_write_mem(CFG_FIFO_ON_EVENT, 11,
            (unsigned char*)regs_continuous);
    case DMP_INT_GESTURE:
        return mpu_write_mem(CFG_FIFO_ON_EVENT, 11,
            (unsigned char*)regs_gesture);
    default:
        return 1;
    }
}

/**
 *  @brief      Get one packet from the FIFO.
 *  If @e sensors does not contain a particular sensor, disregard the data
 *  returned to that pointer.
 *  \n @e sensors can contain a combination of the following flags:
 *  \n INV_X_GYRO, INV_Y_GYRO, INV_Z_GYRO
 *  \n INV_XYZ_GYRO
 *  \n INV_XYZ_ACCEL
 *  \n INV_WXYZ_QUAT
 *  \n If the FIFO has no new data, @e sensors will be zero.
 *  \n If the FIFO is disabled, @e sensors will be zero and this function will
 *  return a non-zero error code.
 *  @param[out] gyro        Gyro data in hardware units.
 *  @param[out] accel       Accel data in hardware units.
 *  @param[out] quat        3-axis quaternion data in hardware units.
 *  @param[out] timestamp   Timestamp in milliseconds.
 *  @param[out] sensors     Mask of sensors read from FIFO.
 *  @param[out] more        Number of remaining packets.
 *  @return     0 if successful.
 */
uint8_t MPUDriver::dmp_read_fifo(short *gyro, short *accel, long *quat,
    unsigned long *timestamp, short *sensors, unsigned char *more)
{
    unsigned char fifo_data[MAX_PACKET_LENGTH];
    unsigned char ii = 0;

    /* TODO: sensors[0] only changes when dmp_enable_feature is called. We can
     * cache this value and save some cycles.
     */
    sensors[0] = 0;

    /* Get a packet. */
    if (mpu_read_fifo_stream(dmp.packet_length, fifo_data, more))
        return mpu_read_fifo_stream(dmp.packet_length, fifo_data, more);

    /* Parse DMP packet. */
    if (dmp.feature_mask & (DMP_FEATURE_LP_QUAT | DMP_FEATURE_6X_LP_QUAT)) {
#ifdef FIFO_CORRUPTION_CHECK
        long quat_q14[4], quat_mag_sq;
#endif
        quat[0] = ((long)fifo_data[0] << 24) | ((long)fifo_data[1] << 16) |
            ((long)fifo_data[2] << 8) | fifo_data[3];
        quat[1] = ((long)fifo_data[4] << 24) | ((long)fifo_data[5] << 16) |
            ((long)fifo_data[6] << 8) | fifo_data[7];
        quat[2] = ((long)fifo_data[8] << 24) | ((long)fifo_data[9] << 16) |
            ((long)fifo_data[10] << 8) | fifo_data[11];
        quat[3] = ((long)fifo_data[12] << 24) | ((long)fifo_data[13] << 16) |
            ((long)fifo_data[14] << 8) | fifo_data[15];
        ii += 16;
#ifdef FIFO_CORRUPTION_CHECK
        /* We can detect a corrupted FIFO by monitoring the quaternion data and
         * ensuring that the magnitude is always normalized to one. This
         * shouldn't happen in normal operation, but if an I2C error occurs,
         * the FIFO reads might become misaligned.
         *
         * Let's start by scaling down the quaternion data to avoid long long
         * math.
         */
        quat_q14[0] = quat[0] >> 16;
        quat_q14[1] = quat[1] >> 16;
        quat_q14[2] = quat[2] >> 16;
        quat_q14[3] = quat[3] >> 16;
        quat_mag_sq = quat_q14[0] * quat_q14[0] + quat_q14[1] * quat_q14[1] +
            quat_q14[2] * quat_q14[2] + quat_q14[3] * quat_q14[3];
        if ((quat_mag_sq < QUAT_MAG_SQ_MIN) ||
            (quat_mag_sq > QUAT_MAG_SQ_MAX)) {
            /* Quaternion is outside of the acceptable threshold. */
            mpu_reset_fifo();
            sensors[0] = 0;
            return 2;
        }
        sensors[0] |= INV_WXYZ_QUAT;
#endif
    }

    if (dmp.feature_mask & DMP_FEATURE_SEND_RAW_ACCEL) {
        accel[0] = ((short)fifo_data[ii+0] << 8) | fifo_data[ii+1];
        accel[1] = ((short)fifo_data[ii+2] << 8) | fifo_data[ii+3];
        accel[2] = ((short)fifo_data[ii+4] << 8) | fifo_data[ii+5];
        ii += 6;
        sensors[0] |= INV_XYZ_ACCEL;
    }

    if (dmp.feature_mask & DMP_FEATURE_SEND_ANY_GYRO) {
        gyro[0] = ((short)fifo_data[ii+0] << 8) | fifo_data[ii+1];
        gyro[1] = ((short)fifo_data[ii+2] << 8) | fifo_data[ii+3];
        gyro[2] = ((short)fifo_data[ii+4] << 8) | fifo_data[ii+5];
        ii += 6;
        sensors[0] |= INV_XYZ_GYRO;
    }

    /* Gesture data is at the end of the DMP packet. Parse it and call
     * the gesture callbacks (if registered).
     */
    if (dmp.feature_mask & (DMP_FEATURE_TAP | DMP_FEATURE_ANDROID_ORIENT))
        decode_gesture(fifo_data + ii);

    get_ms(timestamp);

    return 0;
}

/**
 *  @brief      Register a function to be executed on a tap event.
 *  The tap direction is represented by one of the following:
 *  \n TAP_X_UP
 *  \n TAP_X_DOWN
 *  \n TAP_Y_UP
 *  \n TAP_Y_DOWN
 *  \n TAP_Z_UP
 *  \n TAP_Z_DOWN
 *  @param[in]  func    Callback function.
 *  @return     0 if successful.
 */
void MPUDriver::dmp_register_tap_cb(void (*func)(unsigned char, unsigned char))
{
    dmp.tap_cb = func;
}

/**
 *  @brief      Register a function to be executed on a android orientation event.
 *  @param[in]  func    Callback function.
 *  @return     0 if successful.
 */
void MPUDriver::dmp_register_android_orient_cb(void (*func)(unsigned char))
{
    dmp.android_orient_cb = func;
}



/**
 *  @brief      Read from a single register.
 *  NOTE: The memory and FIFO read/write registers cannot be accessed.
 *  @param[in]  reg     Register address.
 *  @param[out] data    Register data.
 *  @return     0 if successful.
 */
uint8_t MPUDriver::mpu_read_reg(unsigned char reg, unsigned char *data)
{
    if (reg == REG_fifo_r_w || reg == REG_mem_r_w)
        return 1;
    if (reg >= HW_num_reg)
        return 2;
    return i2c_read(HW_addr, reg, 1, data);
}

/**
 *  @brief      Initialize hardware.
 *  Initial configuration:\n
 *  Gyro FSR: +/- 2000DPS\n
 *  Accel FSR +/- 2G\n
 *  DLPF: 42Hz\n
 *  FIFO rate: 50Hz\n
 *  Clock source: Gyro PLL\n
 *  FIFO: Disabled.\n
 *  Data ready interrupt: Disabled, active low, unlatched.
 *  @param[in]  int_param   Platform-specific parameters to interrupt API.
 *  @return     0 if successful.
 */
uint8_t MPUDriver::mpu_init(struct int_param_s *int_param)
{

    unsigned char data[6];

    unsigned char testData[6];
    /* Reset device. */
    data[0] = BIT_RESET;
	
    if (i2c_write(HW_addr, REG_pwr_mgmt_1, 1, data))
        return 1;

    delay_ms(100);

	
    /* Wake up chip. */ // Strange thing happened??
    data[0] = 0x00;
    if (i2c_write(HW_addr, REG_pwr_mgmt_1, 1, data))
        return 2;

    chip_cfg.accel_half = 0;
	   
#ifdef MPU6500
    /* MPU6500 shares 4kB of memory between the DMP and the FIFO. Since the
     * first 3kB are needed by the DMP, we'll use the last 1kB for the FIFO.
     */
    data[0] = BIT_FIFO_SIZE_1024;
    if (i2c_write(HW_addr, REG_accel_cfg2, 1, data))
        return 3;
#endif

    
    /* Set to invalid values to ensure no I2C writes are skipped. */
    chip_cfg.sensors = 0xFF;
    chip_cfg.gyro_fsr = 0xFF;
    chip_cfg.accel_fsr = 0xFF;
    chip_cfg.lpf = 0xFF;
    chip_cfg.sample_rate = 0xFFFF;
    chip_cfg.fifo_enable = 0xFF;
    chip_cfg.bypass_mode = 0xFF;
#ifdef AK89xx_SECONDARY
    chip_cfg.compass_sample_rate = 0xFFFF;
#endif
    /* mpu_set_sensors always preserves this setting. */
    chip_cfg.clk_src = INV_CLK_PLL;
    /* Handled in next call to mpu_set_bypass. */
    chip_cfg.active_low_int = 1;
    chip_cfg.latched_int = 0;
    chip_cfg.int_motion_only = 0;
    chip_cfg.lp_accel_mode = 0;
    memset(&chip_cfg.cache, 0, sizeof(chip_cfg.cache));
    chip_cfg.dmp_on = 0;
    chip_cfg.dmp_loaded = 0;
    chip_cfg.dmp_sample_rate = 0;

    if (mpu_set_gyro_fsr(2000))
        return 4;
    if (mpu_set_accel_fsr(2))
        return 5;
    if (mpu_set_lpf(42))
        return 6;
    if (mpu_set_sample_rate(50))
        return 7;
    if (mpu_configure_fifo(0))
        return 8;

    if (int_param)
        reg_int_cb(int_param);

#ifdef AK89xx_SECONDARY
    setup_compass();
    if (mpu_set_compass_sample_rate(10))
        return 9;
#else
    /* Already disabled by setup_compass. */
    if (mpu_set_bypass(0))
        return 10;
#endif

    mpu_set_sensors(0);
    return 0;
}

/**
 *  @brief      Enter low-power accel-only mode.
 *  In low-power accel mode, the chip goes to sleep and only wakes up to sample
 *  the accelerometer at one of the following frequencies:
 *  \n MPU6050: 1.25Hz, 5Hz, 20Hz, 40Hz
 *  \n MPU6500: 0.24Hz, 0.49Hz, 0.98Hz, 1.95Hz, 3.91Hz, 7.81Hz, 15.63Hz, 31.25Hz, 62.5Hz, 125Hz, 250Hz, 500Hz
 *  \n If the requested rate is not one listed above, the device will be set to
 *  the next highest rate. Requesting a rate above the maximum supported
 *  frequency will result in an error.
 *  \n To select a fractional wake-up frequency, round down the value passed to
 *  @e rate.
 *  @param[in]  rate        Minimum sampling rate, or zero to disable LP
 *                          accel mode.
 *  @return     0 if successful.
 */
uint8_t MPUDriver::mpu_lp_accel_mode(unsigned short rate)
{
    unsigned char tmp[2];

#if defined MPU6500
    unsigned char data;
#endif

    if (!rate) {
        mpu_set_int_latched(0);
        tmp[0] = 0;
        tmp[1] = BIT_STBY_XYZG;
        if (i2c_write(HW_addr, REG_pwr_mgmt_1, 2, tmp))
            return 1;
        chip_cfg.lp_accel_mode = 0;
        return 0;
    }
    /* For LP accel, we automatically configure the hardware to produce latched
     * interrupts. In LP accel mode, the hardware cycles into sleep mode before
     * it gets a chance to deassert the interrupt pin; therefore, we shift this
     * responsibility over to the MCU.
     *
     * Any register read will clear the interrupt.
     */
    mpu_set_int_latched(1);
#if defined MPU6050
    tmp[0] = BIT_LPA_CYCLE;
    if (rate == 1) {
        tmp[1] = INV_LPA_1_25HZ;
        mpu_set_lpf(5);
    } else if (rate <= 5) {
        tmp[1] = INV_LPA_5HZ;
        mpu_set_lpf(5);
    } else if (rate <= 20) {
        tmp[1] = INV_LPA_20HZ;
        mpu_set_lpf(10);
    } else {
        tmp[1] = INV_LPA_40HZ;
        mpu_set_lpf(20);
    }
    tmp[1] = (tmp[1] << 6) | BIT_STBY_XYZG;
    if (i2c_write(HW_addr, REG_pwr_mgmt_1, 2, tmp))
        return 2;
#elif defined MPU6500
    /* Set wake frequency. */
    if (rate == 1)
    	data = INV_LPA_0_98HZ;
    else if (rate == 2)
    	data = INV_LPA_1_95HZ;
    else if (rate <= 5)
    	data = INV_LPA_3_91HZ;
    else if (rate <= 10)
    	data = INV_LPA_7_81HZ;
    else if (rate <= 20)
    	data = INV_LPA_15_63HZ;
    else if (rate <= 40)
    	data = INV_LPA_31_25HZ;
    else if (rate <= 70)
    	data = INV_LPA_62_50HZ;
    else if (rate <= 125)
    	data = INV_LPA_125HZ;
    else if (rate <= 250)
    	data = INV_LPA_250HZ;
    else
    	data = INV_LPA_500HZ;

    if (i2c_write(HW_addr, REG_lp_accel_odr, 1, &data))
        return 3;

    if (i2c_read(HW_addr, REG_accel_cfg2, 1, &data))
        return 4;

    data = data | BIT_ACCL_FC_B;
    if (i2c_write(HW_addr, REG_accel_cfg2, 1, &data))
            return 5;

    data = BIT_LPA_CYCLE;
    if (i2c_write(HW_addr, REG_pwr_mgmt_1, 1, &data))
        return 6;
#endif
    chip_cfg.sensors = INV_XYZ_ACCEL;
    chip_cfg.clk_src = 0;
    chip_cfg.lp_accel_mode = 1;
    mpu_configure_fifo(0);

    return 0;
}

/**
 *  @brief      Read raw gyro data directly from the registers.
 *  @param[out] data        Raw data in hardware units.
 *  @param[out] timestamp   Timestamp in milliseconds. Null if not needed.
 *  @return     0 if successful.
 */
uint8_t MPUDriver::mpu_get_gyro_reg(short *data, unsigned long *timestamp)
{
    unsigned char tmp[6];

    if (!(chip_cfg.sensors & INV_XYZ_GYRO))
        return 1;

    if (i2c_read(HW_addr, REG_raw_gyro, 6, tmp))
        return 2;
    data[0] = (tmp[0] << 8) | tmp[1];
    data[1] = (tmp[2] << 8) | tmp[3];
    data[2] = (tmp[4] << 8) | tmp[5];
    if (timestamp)
        get_ms(timestamp);
    return 0;
}

/**
 *  @brief      Read raw accel data directly from the registers.
 *  @param[out] data        Raw data in hardware units.
 *  @param[out] timestamp   Timestamp in milliseconds. Null if not needed.
 *  @return     0 if successful.
 */
uint8_t MPUDriver::mpu_get_accel_reg(short *data, unsigned long *timestamp)
{
    unsigned char tmp[6];

    if (!(chip_cfg.sensors & INV_XYZ_ACCEL))
        return 1;

    if (i2c_read(HW_addr, REG_raw_accel, 6, tmp))
        return 2;
    data[0] = (tmp[0] << 8) | tmp[1];
    data[1] = (tmp[2] << 8) | tmp[3];
    data[2] = (tmp[4] << 8) | tmp[5];
    if (timestamp)
        get_ms(timestamp);
    return 0;
}

/**
 *  @brief      Read temperature data directly from the registers.
 *  @param[out] data        Data in q16 format.
 *  @param[out] timestamp   Timestamp in milliseconds. Null if not needed.
 *  @return     0 if successful.
 */
uint8_t MPUDriver::mpu_get_temperature(long *data, unsigned long *timestamp)
{
    unsigned char tmp[2];
    short raw;

    if (!(chip_cfg.sensors))
        return 1;

    if (i2c_read(HW_addr, REG_temp, 2, tmp))
        return 2;
    raw = (tmp[0] << 8) | tmp[1];
    if (timestamp)
        get_ms(timestamp);

    data[0] = (long)((35 + ((raw - (float)HW_temp_offset) / HW_temp_sens)) * 65536L);
    return 0;
}

/**
 *  @brief      Read biases to the accel bias 6500 registers.
 *  This function reads from the MPU6500 accel offset cancellations registers.
 *  The format are G in +-8G format. The register is initialized with OTP
 *  factory trim values.
 *  @param[in]  accel_bias  returned structure with the accel bias
 *  @return     0 if successful.
 */
uint8_t MPUDriver::mpu_read_6500_accel_bias(long *accel_bias) {
	unsigned char data[6];
	if (i2c_read(HW_addr, 0x77, 2, &data[0]))
		return 1;
	if (i2c_read(HW_addr, 0x7A, 2, &data[2]))
		return 2;
	if (i2c_read(HW_addr, 0x7D, 2, &data[4]))
		return 3;
	accel_bias[0] = ((long)data[0]<<8) | data[1];
	accel_bias[1] = ((long)data[2]<<8) | data[3];
	accel_bias[2] = ((long)data[4]<<8) | data[5];
	return 0;
}

/**
 *  @brief      Read biases to the accel bias 6050 registers.
 *  This function reads from the MPU6050 accel offset cancellations registers.
 *  The format are G in +-8G format. The register is initialized with OTP
 *  factory trim values.
 *  @param[in]  accel_bias  returned structure with the accel bias
 *  @return     0 if successful.
 */
uint8_t MPUDriver::mpu_read_6050_accel_bias(long *accel_bias) {
	unsigned char data[6];
	if (i2c_read(HW_addr, 0x06, 2, &data[0]))
		return 1;
	if (i2c_read(HW_addr, 0x08, 2, &data[2]))
		return 2;
	if (i2c_read(HW_addr, 0x0A, 2, &data[4]))
		return 3;
	accel_bias[0] = ((long)data[0]<<8) | data[1];
	accel_bias[1] = ((long)data[2]<<8) | data[3];
	accel_bias[2] = ((long)data[4]<<8) | data[5];
	return 0;
}

uint8_t MPUDriver::mpu_read_6500_gyro_bias(long *gyro_bias) {
	unsigned char data[6];
	if (i2c_read(HW_addr, 0x13, 2, &data[0]))
		return 1;
	if (i2c_read(HW_addr, 0x15, 2, &data[2]))
		return 2;
	if (i2c_read(HW_addr, 0x17, 2, &data[4]))
		return 3;
	gyro_bias[0] = ((long)data[0]<<8) | data[1];
	gyro_bias[1] = ((long)data[2]<<8) | data[3];
	gyro_bias[2] = ((long)data[4]<<8) | data[5];
	return 0;
}

/**
 *  @brief      Push biases to the gyro bias 6500/6050 registers.
 *  This function expects biases relative to the current sensor output, and
 *  these biases will be added to the factory-supplied values. Bias inputs are LSB
 *  in +-1000dps format.
 *  @param[in]  gyro_bias  New biases.
 *  @return     0 if successful.
 */
uint8_t MPUDriver::mpu_set_gyro_bias_reg(long *gyro_bias)
{
    unsigned char data[6] = {0, 0, 0, 0, 0, 0};
    long gyro_reg_bias[3] = {0, 0, 0};
    int i=0;

    if(mpu_read_6500_gyro_bias(gyro_reg_bias))
        return 1;

    for(i=0;i<3;i++) {
        gyro_reg_bias[i]-= gyro_bias[i];
    }

    data[0] = (gyro_reg_bias[0] >> 8) & 0xff;
    data[1] = (gyro_reg_bias[0]) & 0xff;
    data[2] = (gyro_reg_bias[1] >> 8) & 0xff;
    data[3] = (gyro_reg_bias[1]) & 0xff;
    data[4] = (gyro_reg_bias[2] >> 8) & 0xff;
    data[5] = (gyro_reg_bias[2]) & 0xff;

    if (i2c_write(HW_addr, 0x13, 2, &data[0]))
        return 2;
    if (i2c_write(HW_addr, 0x15, 2, &data[2]))
        return 3;
    if (i2c_write(HW_addr, 0x17, 2, &data[4]))
        return 4;
    return 0;
}

/**
 *  @brief      Push biases to the accel bias 6050 registers.
 *  This function expects biases relative to the current sensor output, and
 *  these biases will be added to the factory-supplied values. Bias inputs are LSB
 *  in +-8G format.
 *  @param[in]  accel_bias  New biases.
 *  @return     0 if successful.
 */
uint8_t MPUDriver::mpu_set_accel_bias_6050_reg(const long *accel_bias)
{
    unsigned char data[6] = {0, 0, 0, 0, 0, 0};
    long accel_reg_bias[3] = {0, 0, 0};

    if(mpu_read_6050_accel_bias(accel_reg_bias))
    	return 1;

    accel_reg_bias[0] -= (accel_bias[0] & ~1);
    accel_reg_bias[1] -= (accel_bias[1] & ~1);
    accel_reg_bias[2] -= (accel_bias[2] & ~1);

    data[0] = (accel_reg_bias[0] >> 8) & 0xff;
    data[1] = (accel_reg_bias[0]) & 0xff;
    data[2] = (accel_reg_bias[1] >> 8) & 0xff;
    data[3] = (accel_reg_bias[1]) & 0xff;
    data[4] = (accel_reg_bias[2] >> 8) & 0xff;
    data[5] = (accel_reg_bias[2]) & 0xff;

    if (i2c_write(HW_addr, 0x06, 2, &data[0]))
        return 2;
    if (i2c_write(HW_addr, 0x08, 2, &data[2]))
        return 3;
    if (i2c_write(HW_addr, 0x0A, 2, &data[4]))
        return 4;

    return 0;
}

/**
 *  @brief      Push biases to the accel bias 6500 registers.
 *  This function expects biases relative to the current sensor output, and
 *  these biases will be added to the factory-supplied values. Bias inputs are LSB
 *  in +-8G format.
 *  @param[in]  accel_bias  New biases.
 *  @return     0 if successful.
 */
uint8_t MPUDriver::mpu_set_accel_bias_6500_reg(const long *accel_bias) {
    unsigned char data[6] = {0, 0, 0, 0, 0, 0};
    long accel_reg_bias[3] = {0, 0, 0};

    if(mpu_read_6500_accel_bias(accel_reg_bias))
        return 1;

    // Preserve bit 0 of factory value (for temperature compensation)
    accel_reg_bias[0] -= (accel_bias[0] & ~1);
    accel_reg_bias[1] -= (accel_bias[1] & ~1);
    accel_reg_bias[2] -= (accel_bias[2] & ~1);

    data[0] = (accel_reg_bias[0] >> 8) & 0xff;
    data[1] = (accel_reg_bias[0]) & 0xff;
    data[2] = (accel_reg_bias[1] >> 8) & 0xff;
    data[3] = (accel_reg_bias[1]) & 0xff;
    data[4] = (accel_reg_bias[2] >> 8) & 0xff;
    data[5] = (accel_reg_bias[2]) & 0xff;

    if (i2c_write(HW_addr, 0x77, 2, &data[0]))
        return 2;
    if (i2c_write(HW_addr, 0x7A, 2, &data[2]))
        return 3;
    if (i2c_write(HW_addr, 0x7D, 2, &data[4]))
        return 4;

    return 0;
}

/**
 *  @brief  Reset FIFO read/write pointers.
 *  @return 0 if successful.
 */
uint8_t MPUDriver::mpu_reset_fifo(void)
{
    unsigned char data;

    if (!(chip_cfg.sensors))
        return 1;

    data = 0;
    if (i2c_write(HW_addr, REG_int_enable, 1, &data))
        return 2;
    if (i2c_write(HW_addr, REG_fifo_en, 1, &data))
        return 3;
    if (i2c_write(HW_addr, REG_user_ctrl, 1, &data))
        return 4;

    if (chip_cfg.dmp_on) {
        data = BIT_FIFO_RST | BIT_DMP_RST;
        if (i2c_write(HW_addr, REG_user_ctrl, 1, &data))
            return 5;
        delay_ms(50);
        data = BIT_DMP_EN | BIT_FIFO_EN;
        if (chip_cfg.sensors & INV_XYZ_COMPASS)
            data |= BIT_AUX_IF_EN;
        if (i2c_write(HW_addr, REG_user_ctrl, 1, &data))
            return 6;
        if (chip_cfg.int_enable)
            data = BIT_DMP_INT_EN;
        else
            data = 0;
        if (i2c_write(HW_addr, REG_int_enable, 1, &data))
            return 7;
        data = 0;
        if (i2c_write(HW_addr, REG_fifo_en, 1, &data))
            return 8;
    } else {
        data = BIT_FIFO_RST;
        if (i2c_write(HW_addr, REG_user_ctrl, 1, &data))
            return 9;
        if (chip_cfg.bypass_mode || !(chip_cfg.sensors & INV_XYZ_COMPASS))
            data = BIT_FIFO_EN;
        else
            data = BIT_FIFO_EN | BIT_AUX_IF_EN;
        if (i2c_write(HW_addr, REG_user_ctrl, 1, &data))
            return 10;
        delay_ms(50);
        if (chip_cfg.int_enable)
            data = BIT_DATA_RDY_EN;
        else
            data = 0;
        if (i2c_write(HW_addr, REG_int_enable, 1, &data))
            return 11;
        if (i2c_write(HW_addr, REG_fifo_en, 1, &chip_cfg.fifo_enable))
            return 12;
    }
    return 0;
}

/**
 *  @brief      Get the gyro full-scale range.
 *  @param[out] fsr Current full-scale range.
 *  @return     0 if successful.
 */
void MPUDriver::mpu_get_gyro_fsr(unsigned short *fsr)
{
    switch (chip_cfg.gyro_fsr) {
    case INV_FSR_250DPS:
        fsr[0] = 250;
        break;
    case INV_FSR_500DPS:
        fsr[0] = 500;
        break;
    case INV_FSR_1000DPS:
        fsr[0] = 1000;
        break;
    case INV_FSR_2000DPS:
        fsr[0] = 2000;
        break;
    default:
        fsr[0] = 0;
        break;
    }
}

/**
 *  @brief      Set the gyro full-scale range.
 *  @param[in]  fsr Desired full-scale range.
 *  @return     0 if successful.
 */
uint8_t MPUDriver::mpu_set_gyro_fsr(unsigned short fsr)
{
    unsigned char data;

    if (!(chip_cfg.sensors))
        return 1;

    switch (fsr) {
    case 250:
        data = INV_FSR_250DPS << 3;
        break;
    case 500:
        data = INV_FSR_500DPS << 3;
        break;
    case 1000:
        data = INV_FSR_1000DPS << 3;
        break;
    case 2000:
        data = INV_FSR_2000DPS << 3;
        break;
    default:
        return 2;
    }

    if (chip_cfg.gyro_fsr == (data >> 3))
        return 0;
    if (i2c_write(HW_addr, REG_gyro_cfg, 1, &data))
        return 3;
    chip_cfg.gyro_fsr = data >> 3;
    return 0;
}

/**
 *  @brief      Get the accel full-scale range.
 *  @param[out] fsr Current full-scale range.
 *  @return     0 if successful.
 */
uint8_t MPUDriver::mpu_get_accel_fsr(unsigned char *fsr)
{
    switch (chip_cfg.accel_fsr) {
    case INV_FSR_2G:
        fsr[0] = 2;
        break;
    case INV_FSR_4G:
        fsr[0] = 4;
        break;
    case INV_FSR_8G:
        fsr[0] = 8;
        break;
    case INV_FSR_16G:
        fsr[0] = 16;
        break;
    default:
        return 1;
    }
    if (chip_cfg.accel_half)
        fsr[0] <<= 1;
    return 0;
}

/**
 *  @brief      Set the accel full-scale range.
 *  @param[in]  fsr Desired full-scale range.
 *  @return     0 if successful.
 */
uint8_t MPUDriver::mpu_set_accel_fsr(unsigned char fsr)
{
    unsigned char data;

    if (!(chip_cfg.sensors))
        return 1;

    switch (fsr) {
    case 2:
        data = INV_FSR_2G << 3;
        break;
    case 4:
        data = INV_FSR_4G << 3;
        break;
    case 8:
        data = INV_FSR_8G << 3;
        break;
    case 16:
        data = INV_FSR_16G << 3;
        break;
    default:
        return 2;
    }

    if (chip_cfg.accel_fsr == (data >> 3))
        return 0;
    if (i2c_write(HW_addr, REG_accel_cfg, 1, &data))
        return 3;
    chip_cfg.accel_fsr = data >> 3;
    return 0;
}

/**
 *  @brief      Get the current DLPF setting.
 *  @param[out] lpf Current LPF setting.
 *  0 if successful.
 */
void MPUDriver::mpu_get_lpf(unsigned short *lpf)
{
    switch (chip_cfg.lpf) {
    case INV_FILTER_188HZ:
        lpf[0] = 188;
        break;
    case INV_FILTER_98HZ:
        lpf[0] = 98;
        break;
    case INV_FILTER_42HZ:
        lpf[0] = 42;
        break;
    case INV_FILTER_20HZ:
        lpf[0] = 20;
        break;
    case INV_FILTER_10HZ:
        lpf[0] = 10;
        break;
    case INV_FILTER_5HZ:
        lpf[0] = 5;
        break;
    case INV_FILTER_256HZ_NOLPF2:
    case INV_FILTER_2100HZ_NOLPF:
    default:
        lpf[0] = 0;
        break;
    }
}

/**
 *  @brief      Set digital low pass filter.
 *  The following LPF settings are supported: 188, 98, 42, 20, 10, 5.
 *  @param[in]  lpf Desired LPF setting.
 *  @return     0 if successful.
 */
uint8_t MPUDriver::mpu_set_lpf(unsigned short lpf)
{
    unsigned char data;

    if (!(chip_cfg.sensors))
        return 1;

    if (lpf >= 188)
        data = INV_FILTER_188HZ;
    else if (lpf >= 98)
        data = INV_FILTER_98HZ;
    else if (lpf >= 42)
        data = INV_FILTER_42HZ;
    else if (lpf >= 20)
        data = INV_FILTER_20HZ;
    else if (lpf >= 10)
        data = INV_FILTER_10HZ;
    else
        data = INV_FILTER_5HZ;

    if (chip_cfg.lpf == data)
        return 0;

    if (i2c_write(HW_addr, REG_lpf, 1, &data))
        return 2;

#ifdef MPU6500 //MPU6500 accel/gyro dlpf separately
    data = BIT_FIFO_SIZE_1024 | data;
    if (i2c_write(HW_addr, REG_accel_cfg2, 1, &data))
            return 3;
#endif

    chip_cfg.lpf = data;
    return 0;
}

/**
 *  @brief      Get sampling rate.
 *  @param[out] rate    Current sampling rate (Hz).
 *  @return     0 if successful.
 */
uint8_t MPUDriver::mpu_get_sample_rate(unsigned short *rate)
{
    if (chip_cfg.dmp_on)
        return 1;
    else
        rate[0] = chip_cfg.sample_rate;
    return 0;
}

/**
 *  @brief      Set sampling rate.
 *  Sampling rate must be between 4Hz and 1kHz.
 *  @param[in]  rate    Desired sampling rate (Hz).
 *  @return     0 if successful.
 */
uint8_t MPUDriver::mpu_set_sample_rate(unsigned short rate)
{
    unsigned char data;

    if (!(chip_cfg.sensors))
        return 1;

    if (chip_cfg.dmp_on)
        return 2;
    else {
        if (chip_cfg.lp_accel_mode) {
            if (rate && (rate <= 40)) {
                /* Just stay in low-power accel mode. */
                mpu_lp_accel_mode(rate);
                return 0;
            }
            /* Requested rate exceeds the allowed frequencies in LP accel mode,
             * switch back to full-power mode.
             */
            mpu_lp_accel_mode(0);
        }
        if (rate < 4)
            rate = 4;
        else if (rate > 1000)
            rate = 1000;

        data = 1000 / rate - 1;
        if (i2c_write(HW_addr, REG_rate_div, 1, &data))
            return 3;

        chip_cfg.sample_rate = 1000 / (1 + data);

#ifdef AK89xx_SECONDARY
        mpu_set_compass_sample_rate(min(st.chip_cfg.compass_sample_rate, MAX_COMPASS_SAMPLE_RATE));
#endif

        /* Automatically set LPF to 1/2 sampling rate. */
        mpu_set_lpf(chip_cfg.sample_rate >> 1);
        return 0;
    }
}

/**
 *  @brief      Get compass sampling rate.
 *  @param[out] rate    Current compass sampling rate (Hz).
 *  @return     0 if successful.
 */
 /*
int MPUDriver::mpu_get_compass_sample_rate(unsigned short *rate)
{
#ifdef AK89xx_SECONDARY
    rate[0] = chip_cfg.compass_sample_rate;
    return 0;
#else
    rate[0] = 0;
    return -1;
#endif
}*/

/**
 *  @brief      Set compass sampling rate.
 *  The compass on the auxiliary I2C bus is read by the MPU hardware at a
 *  maximum of 100Hz. The actual rate can be set to a fraction of the gyro
 *  sampling rate.
 *
 *  \n WARNING: The new rate may be different than what was requested. Call
 *  mpu_get_compass_sample_rate to check the actual setting.
 *  @param[in]  rate    Desired compass sampling rate (Hz).
 *  @return     0 if successful.
 */
/*int MPUDriver::mpu_set_compass_sample_rate(unsigned short rate)
{
#ifdef AK89xx_SECONDARY
    unsigned char div;
    if (!rate || rate > chip_cfg.sample_rate || rate > MAX_COMPASS_SAMPLE_RATE)
        return -1;

    div = chip_cfg.sample_rate / rate - 1;
    if (i2c_write(HW_addr, REG_s4_ctrl, 1, &div))
        return -1;
    chip_cfg.compass_sample_rate = chip_cfg.sample_rate / (div + 1);
    return 0;
#else
    return -1;
#endif
}*/

/**
 *  @brief      Get gyro sensitivity scale factor.
 *  @param[out] sens    Conversion from hardware units to dps.
 *  @return     0 if successful.
 */
uint8_t MPUDriver::mpu_get_gyro_sens(float *sens)
{
    switch (chip_cfg.gyro_fsr) {
    case INV_FSR_250DPS:
        sens[0] = 131.f;
        break;
    case INV_FSR_500DPS:
        sens[0] = 65.5f;
        break;
    case INV_FSR_1000DPS:
        sens[0] = 32.8f;
        break;
    case INV_FSR_2000DPS:
        sens[0] = 16.4f;
        break;
    default:
        return 1;
    }
    return 0;
}

/**
 *  @brief      Get accel sensitivity scale factor.
 *  @param[out] sens    Conversion from hardware units to g's.
 *  @return     0 if successful.
 */
uint8_t MPUDriver::mpu_get_accel_sens(unsigned short *sens)
{
    switch (chip_cfg.accel_fsr) {
    case INV_FSR_2G:
        sens[0] = 16384;
        break;
    case INV_FSR_4G:
        sens[0] = 8092;
        break;
    case INV_FSR_8G:
        sens[0] = 4096;
        break;
    case INV_FSR_16G:
        sens[0] = 2048;
        break;
    default:
        return 1;
    }
    if (chip_cfg.accel_half)
        sens[0] >>= 1;
    return 0;
}

/**
 *  @brief      Get current FIFO configuration.
 *  @e sensors can contain a combination of the following flags:
 *  \n INV_X_GYRO, INV_Y_GYRO, INV_Z_GYRO
 *  \n INV_XYZ_GYRO
 *  \n INV_XYZ_ACCEL
 *  @param[out] sensors Mask of sensors in FIFO.
 *  @return     0 if successful.
 */
void MPUDriver::mpu_get_fifo_config(unsigned char *sensors)
{
    sensors[0] = chip_cfg.fifo_enable;
}

/**
 *  @brief      Select which sensors are pushed to FIFO.
 *  @e sensors can contain a combination of the following flags:
 *  \n INV_X_GYRO, INV_Y_GYRO, INV_Z_GYRO
 *  \n INV_XYZ_GYRO
 *  \n INV_XYZ_ACCEL
 *  @param[in]  sensors Mask of sensors to push to FIFO.
 *  @return     0 if successful.
 */
uint8_t MPUDriver::mpu_configure_fifo(unsigned char sensors)
{
    unsigned char prev;
    int result = 0;

    /* Compass data isn't going into the FIFO. Stop trying. */
    sensors &= ~INV_XYZ_COMPASS;

    if (chip_cfg.dmp_on)
        return 0;
    else {
        if (!(chip_cfg.sensors))
            return 1;
        prev = chip_cfg.fifo_enable;
        chip_cfg.fifo_enable = sensors & chip_cfg.sensors;
        if (chip_cfg.fifo_enable != sensors)
            /* You're not getting what you asked for. Some sensors are
             * asleep.
             */
            result = -1;
        else
            result = 0;
        if (sensors || chip_cfg.lp_accel_mode)
            set_int_enable(1);
        else
            set_int_enable(0);
        if (sensors) {
            if (mpu_reset_fifo()) {
                chip_cfg.fifo_enable = prev;
                return 2;
            }
        }
    }

    return result;
}

/**
 *  @brief      Get current power state.
 *  @param[in]  power_on    1 if turned on, 0 if suspended.
 *  @return     0 if successful.
 */
void MPUDriver::mpu_get_power_state(unsigned char *power_on)
{
    if (chip_cfg.sensors)
        power_on[0] = 1;
    else
        power_on[0] = 0;
}

/**
 *  @brief      Turn specific sensors on/off.
 *  @e sensors can contain a combination of the following flags:
 *  \n INV_X_GYRO, INV_Y_GYRO, INV_Z_GYRO
 *  \n INV_XYZ_GYRO
 *  \n INV_XYZ_ACCEL
 *  \n INV_XYZ_COMPASS
 *  @param[in]  sensors    Mask of sensors to wake.
 *  @return     0 if successful.
 */
uint8_t MPUDriver::mpu_set_sensors(unsigned char sensors)
{
    unsigned char data;
#ifdef AK89xx_SECONDARY
    unsigned char user_ctrl;
#endif

    if (sensors & INV_XYZ_GYRO)
        data = INV_CLK_PLL;
    else if (sensors)
        data = 0;
    else
        data = BIT_SLEEP;
    if (i2c_write(HW_addr, REG_pwr_mgmt_1, 1, &data)) {
        chip_cfg.sensors = 0;
        return 1;
    }
    chip_cfg.clk_src = data & ~BIT_SLEEP;

    data = 0;
    if (!(sensors & INV_X_GYRO))
        data |= BIT_STBY_XG;
    if (!(sensors & INV_Y_GYRO))
        data |= BIT_STBY_YG;
    if (!(sensors & INV_Z_GYRO))
        data |= BIT_STBY_ZG;
    if (!(sensors & INV_XYZ_ACCEL))
        data |= BIT_STBY_XYZA;
    if (i2c_write(HW_addr, REG_pwr_mgmt_2, 1, &data)) {
        chip_cfg.sensors = 0;
        return 2;
    }

    if (sensors && (sensors != INV_XYZ_ACCEL))
        /* Latched interrupts only used in LP accel mode. */
        mpu_set_int_latched(0);

#ifdef AK89xx_SECONDARY
#ifdef AK89xx_BYPASS
    if (sensors & INV_XYZ_COMPASS)
        mpu_set_bypass(1);
    else
        mpu_set_bypass(0);
#else
    if (i2c_read(HW_addr, REG_user_ctrl, 1, &user_ctrl))
        return 3;
    /* Handle AKM power management. */
    if (sensors & INV_XYZ_COMPASS) {
        data = AKM_SINGLE_MEASUREMENT;
        user_ctrl |= BIT_AUX_IF_EN;
    } else {
        data = AKM_POWER_DOWN;
        user_ctrl &= ~BIT_AUX_IF_EN;
    }
    if (st.chip_cfg.dmp_on)
        user_ctrl |= BIT_DMP_EN;
    else
        user_ctrl &= ~BIT_DMP_EN;
    if (i2c_write(HW_addr, REG_s1_do, 1, &data))
        return 4;
    /* Enable/disable I2C master mode. */
    if (i2c_write(HW_addr, REG_user_ctrl, 1, &user_ctrl))
        return 5;
#endif
#endif

    chip_cfg.sensors = sensors;
    chip_cfg.lp_accel_mode = 0;
    delay_ms(50);
    return 0;
}

/**
 *  @brief      Read the MPU interrupt status registers.
 *  @param[out] status  Mask of interrupt bits.
 *  @return     0 if successful.
 */
uint8_t MPUDriver::mpu_get_int_status(short *status)
{
    unsigned char tmp[2];
    if (!chip_cfg.sensors)
        return 1;
    if (i2c_read(HW_addr, REG_dmp_int_status, 2, tmp))
        return 2;
    status[0] = (tmp[0] << 8) | tmp[1];
    return 0;
}

/**
 *  @brief      Get one packet from the FIFO.
 *  If @e sensors does not contain a particular sensor, disregard the data
 *  returned to that pointer.
 *  \n @e sensors can contain a combination of the following flags:
 *  \n INV_X_GYRO, INV_Y_GYRO, INV_Z_GYRO
 *  \n INV_XYZ_GYRO
 *  \n INV_XYZ_ACCEL
 *  \n If the FIFO has no new data, @e sensors will be zero.
 *  \n If the FIFO is disabled, @e sensors will be zero and this function will
 *  return a non-zero error code.
 *  @param[out] gyro        Gyro data in hardware units.
 *  @param[out] accel       Accel data in hardware units.
 *  @param[out] timestamp   Timestamp in milliseconds.
 *  @param[out] sensors     Mask of sensors read from FIFO.
 *  @param[out] more        Number of remaining packets.
 *  @return     0 if successful.
 */
uint8_t MPUDriver::mpu_read_fifo(short *gyro, short *accel, unsigned long *timestamp,
        unsigned char *sensors, unsigned char *more)
{
    /* Assumes maximum packet size is gyro (6) + accel (6). */
    unsigned char data[MAX_PACKET_LENGTH];
    unsigned char packet_size = 0;
    unsigned short fifo_count, index = 0;

    if (chip_cfg.dmp_on)
        return 1;

    sensors[0] = 0;
    if (!chip_cfg.sensors)
        return 2;
    if (!chip_cfg.fifo_enable)
        return 3;

    if (chip_cfg.fifo_enable & INV_X_GYRO)
        packet_size += 2;
    if (chip_cfg.fifo_enable & INV_Y_GYRO)
        packet_size += 2;
    if (chip_cfg.fifo_enable & INV_Z_GYRO)
        packet_size += 2;
    if (chip_cfg.fifo_enable & INV_XYZ_ACCEL)
        packet_size += 6;

    if (i2c_read(HW_addr, REG_fifo_count_h, 2, data))
        return 4;
    fifo_count = (data[0] << 8) | data[1];
    if (fifo_count < packet_size)
        return 0;
//    log_i("FIFO count: %hd\n", fifo_count);
    if (fifo_count > (HW_max_fifo >> 1)) {
        /* FIFO is 50% full, better check overflow bit. */
        if (i2c_read(HW_addr, REG_int_status, 1, data))
            return 5;
        if (data[0] & BIT_FIFO_OVERFLOW) {
            mpu_reset_fifo();
            return 6;
        }
    }
    get_ms((unsigned long*)timestamp);

    if (i2c_read(HW_addr, REG_fifo_r_w, packet_size, data))
        return 7;
    more[0] = fifo_count / packet_size - 1;
    sensors[0] = 0;

    if ((index != packet_size) && (chip_cfg.fifo_enable & INV_XYZ_ACCEL)) {
        accel[0] = (data[index+0] << 8) | data[index+1];
        accel[1] = (data[index+2] << 8) | data[index+3];
        accel[2] = (data[index+4] << 8) | data[index+5];
        sensors[0] |= INV_XYZ_ACCEL;
        index += 6;
    }
    if ((index != packet_size) && (chip_cfg.fifo_enable & INV_X_GYRO)) {
        gyro[0] = (data[index+0] << 8) | data[index+1];
        sensors[0] |= INV_X_GYRO;
        index += 2;
    }
    if ((index != packet_size) && (chip_cfg.fifo_enable & INV_Y_GYRO)) {
        gyro[1] = (data[index+0] << 8) | data[index+1];
        sensors[0] |= INV_Y_GYRO;
        index += 2;
    }
    if ((index != packet_size) && (chip_cfg.fifo_enable & INV_Z_GYRO)) {
        gyro[2] = (data[index+0] << 8) | data[index+1];
        sensors[0] |= INV_Z_GYRO;
        index += 2;
    }

    return 0;
}

unsigned short MPUDriver::mpu_get_fifo_count()
{
	unsigned short fifo_count;
	unsigned char tmp[2];
	if (i2c_read(HW_addr, REG_fifo_count_h, 2, tmp))
        return 3;
    fifo_count = (tmp[0] << 8) | tmp[1];
	return fifo_count;
}

/**
 *  @brief      Get one unparsed packet from the FIFO.
 *  This function should be used if the packet is to be parsed elsewhere.
 *  @param[in]  length  Length of one FIFO packet.
 *  @param[in]  data    FIFO packet.
 *  @param[in]  more    Number of remaining packets.
 */
uint8_t MPUDriver::mpu_read_fifo_stream(unsigned short length, unsigned char *data,
    unsigned char *more)
{
    unsigned char tmp[2];
    unsigned short fifo_count;
    if (!chip_cfg.dmp_on)
        return 1;
    if (!chip_cfg.sensors)
        return 2;

    if (i2c_read(HW_addr, REG_fifo_count_h, 2, tmp))
        return 3;
    fifo_count = (tmp[0] << 8) | tmp[1];
    if (fifo_count < length) {
        more[0] = 0;
        return 4;
    }
    if (fifo_count > (HW_max_fifo >> 1)) {
        /* FIFO is 50% full, better check overflow bit. */
        if (i2c_read(HW_addr, REG_int_status, 1, tmp))
            return 5;
        if (tmp[0] & BIT_FIFO_OVERFLOW) {
            mpu_reset_fifo();
            return 6;
        }
    }

    if (i2c_read(HW_addr, REG_fifo_r_w, length, data))
        return 7;
    more[0] = fifo_count / length - 1;
    return 0;
}

/**
 *  @brief      Set device to bypass mode.
 *  @param[in]  bypass_on   1 to enable bypass mode.
 *  @return     0 if successful.
 */
uint8_t MPUDriver::mpu_set_bypass(unsigned char bypass_on)
{
    unsigned char tmp;

    if (chip_cfg.bypass_mode == bypass_on)
        return 0;

    if (bypass_on) {
        if (i2c_read(HW_addr, REG_user_ctrl, 1, &tmp))
            return 1;
        tmp &= ~BIT_AUX_IF_EN;
        if (i2c_write(HW_addr, REG_user_ctrl, 1, &tmp))
            return 2;
        delay_ms(3);
        tmp = BIT_BYPASS_EN;
        if (chip_cfg.active_low_int)
            tmp |= BIT_ACTL;
        if (chip_cfg.latched_int)
            tmp |= BIT_LATCH_EN | BIT_ANY_RD_CLR;
        if (i2c_write(HW_addr, REG_int_pin_cfg, 1, &tmp))
            return 3;
    } else {
        /* Enable I2C master mode if compass is being used. */
        if (i2c_read(HW_addr, REG_user_ctrl, 1, &tmp))
            return 4;
        if (chip_cfg.sensors & INV_XYZ_COMPASS)
            tmp |= BIT_AUX_IF_EN;
        else
            tmp &= ~BIT_AUX_IF_EN;
        if (i2c_write(HW_addr, REG_user_ctrl, 1, &tmp))
            return 5;
        delay_ms(3);
        if (chip_cfg.active_low_int)
            tmp = BIT_ACTL;
        else
            tmp = 0;
        if (chip_cfg.latched_int)
            tmp |= BIT_LATCH_EN | BIT_ANY_RD_CLR;
        if (i2c_write(HW_addr, REG_int_pin_cfg, 1, &tmp))
            return 6;
    }
    chip_cfg.bypass_mode = bypass_on;
    return 0;
}

/**
 *  @brief      Set interrupt level.
 *  @param[in]  active_low  1 for active low, 0 for active high.
 *  @return     0 if successful.
 */
void MPUDriver::mpu_set_int_level(unsigned char active_low)
{
    chip_cfg.active_low_int = active_low;
}

/**
 *  @brief      Enable latched interrupts.
 *  Any MPU register will clear the interrupt.
 *  @param[in]  enable  1 to enable, 0 to disable.
 *  @return     0 if successful.
 */
uint8_t MPUDriver::mpu_set_int_latched(unsigned char enable)
{
    unsigned char tmp;
    if (chip_cfg.latched_int == enable)
        return 0;

    if (enable)
        tmp = BIT_LATCH_EN | BIT_ANY_RD_CLR;
    else
        tmp = 0;
    if (chip_cfg.bypass_mode)
        tmp |= BIT_BYPASS_EN;
    if (chip_cfg.active_low_int)
        tmp |= BIT_ACTL;
    if (i2c_write(HW_addr, REG_int_pin_cfg, 1, &tmp))
        return 1;
    chip_cfg.latched_int = enable;
    return 0;
}

#ifdef MPU6050
static uint8_t get_accel_prod_shift(float *st_shift)
{
    unsigned char tmp[4], shift_code[3], ii;

    if (i2c_read(HW_addr, 0x0D, 4, tmp))
        return 0x07;

    shift_code[0] = ((tmp[0] & 0xE0) >> 3) | ((tmp[3] & 0x30) >> 4);
    shift_code[1] = ((tmp[1] & 0xE0) >> 3) | ((tmp[3] & 0x0C) >> 2);
    shift_code[2] = ((tmp[2] & 0xE0) >> 3) | (tmp[3] & 0x03);
    for (ii = 0; ii < 3; ii++) {
        if (!shift_code[ii]) {
            st_shift[ii] = 0.f;
            continue;
        }
        /* Equivalent to..
         * st_shift[ii] = 0.34f * powf(0.92f/0.34f, (shift_code[ii]-1) / 30.f)
         */
        st_shift[ii] = 0.34f;
        while (--shift_code[ii])
            st_shift[ii] *= 1.034f;
    }
    return 0;
}

static uint8_t accel_self_test(long *bias_regular, long *bias_st)
{
    int jj, result = 0;
    float st_shift[3], st_shift_cust, st_shift_var;

    get_accel_prod_shift(st_shift);
    for(jj = 0; jj < 3; jj++) {
        st_shift_cust = labs(bias_regular[jj] - bias_st[jj]) / 65536.f;
        if (st_shift[jj]) {
            st_shift_var = st_shift_cust / st_shift[jj] - 1.f;
            if (fabs(st_shift_var) > TEST_max_accel_var)
                result |= 1 << jj;
        } else if ((st_shift_cust < TEST_min_g) ||
            (st_shift_cust > TEST_max_g))
            result |= 1 << jj;
    }

    return result;
}

static uint8_t gyro_self_test(long *bias_regular, long *bias_st)
{
    int jj, result = 0;
    unsigned char tmp[3];
    float st_shift, st_shift_cust, st_shift_var;

    if (i2c_read(HW_addr, 0x0D, 3, tmp))
        return 0x07;

    tmp[0] &= 0x1F;
    tmp[1] &= 0x1F;
    tmp[2] &= 0x1F;

    for (jj = 0; jj < 3; jj++) {
        st_shift_cust = labs(bias_regular[jj] - bias_st[jj]) / 65536.f;
        if (tmp[jj]) {
            st_shift = 3275.f / TEST_gyro_sens;
            while (--tmp[jj])
                st_shift *= 1.046f;
            st_shift_var = st_shift_cust / st_shift - 1.f;
            if (fabs(st_shift_var) > TEST_max_gyro_var)
                result |= 1 << jj;
        } else if ((st_shift_cust < TEST_min_dps) ||
            (st_shift_cust > TEST_max_dps))
            result |= 1 << jj;
    }
    return result;
}
#endif

#ifdef AK89xx_SECONDARY
//static int compass_self_test(void){}
#endif

static int get_st_biases(long *gyro, long *accel, unsigned char hw_test)
{
    unsigned char data[MAX_PACKET_LENGTH];
    unsigned char packet_count, ii;
    unsigned short fifo_count;

    data[0] = 0x01;
    data[1] = 0;
    if (i2c_write(HW_addr, REG_pwr_mgmt_1, 2, data))
        return 1;
    delay_ms(200);
    data[0] = 0;
    if (i2c_write(HW_addr, REG_int_enable, 1, data))
        return 2;
    if (i2c_write(HW_addr, REG_fifo_en, 1, data))
        return 3;
    if (i2c_write(HW_addr, REG_pwr_mgmt_1, 1, data))
        return 4;
    if (i2c_write(HW_addr, REG_i2c_mst, 1, data))
        return 5;
    if (i2c_write(HW_addr, REG_user_ctrl, 1, data))
        return 6;
    data[0] = BIT_FIFO_RST | BIT_DMP_RST;
    if (i2c_write(HW_addr, REG_user_ctrl, 1, data))
        return 7;
    delay_ms(15);
    data[0] = TEST_reg_lpf;
    if (i2c_write(HW_addr, REG_lpf, 1, data))
        return 8;
    data[0] = TEST_reg_rate_div;
    if (i2c_write(HW_addr, REG_rate_div, 1, data))
        return 9;
    if (hw_test)
        data[0] = TEST_reg_gyro_fsr | 0xE0;
    else
        data[0] = TEST_reg_gyro_fsr;
    if (i2c_write(HW_addr, REG_gyro_cfg, 1, data))
        return 10;

    if (hw_test)
        data[0] = TEST_reg_accel_fsr | 0xE0;
    else
        data[0] = TEST_reg_accel_fsr;
    if (i2c_write(HW_addr, REG_accel_cfg, 1, data))
        return 11;
    if (hw_test)
        delay_ms(200);

    /* Fill FIFO for test.wait_ms milliseconds. */
    data[0] = BIT_FIFO_EN;
    if (i2c_write(HW_addr, REG_user_ctrl, 1, data))
        return 12;

    data[0] = INV_XYZ_GYRO | INV_XYZ_ACCEL;
    if (i2c_write(HW_addr, REG_fifo_en, 1, data))
        return 13;
    delay_ms(TEST_wait_ms);
    data[0] = 0;
    if (i2c_write(HW_addr, REG_fifo_en, 1, data))
        return 14;

    if (i2c_read(HW_addr, REG_fifo_count_h, 2, data))
        return 15;

    fifo_count = (data[0] << 8) | data[1];
    packet_count = fifo_count / MAX_PACKET_LENGTH;
    gyro[0] = gyro[1] = gyro[2] = 0;
    accel[0] = accel[1] = accel[2] = 0;

    for (ii = 0; ii < packet_count; ii++) {
        short accel_cur[3], gyro_cur[3];
        if (i2c_read(HW_addr, REG_fifo_r_w, MAX_PACKET_LENGTH, data))
            return 16;
        accel_cur[0] = ((short)data[0] << 8) | data[1];
        accel_cur[1] = ((short)data[2] << 8) | data[3];
        accel_cur[2] = ((short)data[4] << 8) | data[5];
        accel[0] += (long)accel_cur[0];
        accel[1] += (long)accel_cur[1];
        accel[2] += (long)accel_cur[2];
        gyro_cur[0] = (((short)data[6] << 8) | data[7]);
        gyro_cur[1] = (((short)data[8] << 8) | data[9]);
        gyro_cur[2] = (((short)data[10] << 8) | data[11]);
        gyro[0] += (long)gyro_cur[0];
        gyro[1] += (long)gyro_cur[1];
        gyro[2] += (long)gyro_cur[2];
    }
#ifdef EMPL_NO_64BIT
    gyro[0] = (long)(((float)gyro[0]*65536.f) / TEST_gyro_sens / packet_count);
    gyro[1] = (long)(((float)gyro[1]*65536.f) / TEST_gyro_sens / packet_count);
    gyro[2] = (long)(((float)gyro[2]*65536.f) / TEST_gyro_sens / packet_count);
    if (has_accel) {
        accel[0] = (long)(((float)accel[0]*65536.f) / TEST_accel_sens /
            packet_count);
        accel[1] = (long)(((float)accel[1]*65536.f) / TEST_accel_sens /
            packet_count);
        accel[2] = (long)(((float)accel[2]*65536.f) / TEST_accel_sens /
            packet_count);
        /* Don't remove gravity! */
        accel[2] -= 65536L;
    }
#else
    gyro[0] = (long)(((long long)gyro[0]<<16) / TEST_gyro_sens / packet_count);
    gyro[1] = (long)(((long long)gyro[1]<<16) / TEST_gyro_sens / packet_count);
    gyro[2] = (long)(((long long)gyro[2]<<16) / TEST_gyro_sens / packet_count);
    accel[0] = (long)(((long long)accel[0]<<16) / TEST_accel_sens /
        packet_count);
    accel[1] = (long)(((long long)accel[1]<<16) / TEST_accel_sens /
        packet_count);
    accel[2] = (long)(((long long)accel[2]<<16) / TEST_accel_sens /
        packet_count);
    /* Don't remove gravity! */
    if (accel[2] > 0L)
        accel[2] -= 65536L;
    else
        accel[2] += 65536L;
#endif

    return 0;
}

#ifdef MPU6500
#define REG_6500_XG_ST_DATA     0x0
#define REG_6500_XA_ST_DATA     0xD
const unsigned short mpu_6500_st_tb[256] PROGMEM = {
//static const unsigned short mpu_6500_st_tb[256] = {
	2620,2646,2672,2699,2726,2753,2781,2808, //7
	2837,2865,2894,2923,2952,2981,3011,3041, //15
	3072,3102,3133,3165,3196,3228,3261,3293, //23
	3326,3359,3393,3427,3461,3496,3531,3566, //31
	3602,3638,3674,3711,3748,3786,3823,3862, //39
	3900,3939,3979,4019,4059,4099,4140,4182, //47
	4224,4266,4308,4352,4395,4439,4483,4528, //55
	4574,4619,4665,4712,4759,4807,4855,4903, //63
	4953,5002,5052,5103,5154,5205,5257,5310, //71
	5363,5417,5471,5525,5581,5636,5693,5750, //79
	5807,5865,5924,5983,6043,6104,6165,6226, //87
	6289,6351,6415,6479,6544,6609,6675,6742, //95
	6810,6878,6946,7016,7086,7157,7229,7301, //103
	7374,7448,7522,7597,7673,7750,7828,7906, //111
	7985,8065,8145,8227,8309,8392,8476,8561, //119
	8647,8733,8820,8909,8998,9088,9178,9270,
	9363,9457,9551,9647,9743,9841,9939,10038,
	10139,10240,10343,10446,10550,10656,10763,10870,
	10979,11089,11200,11312,11425,11539,11654,11771,
	11889,12008,12128,12249,12371,12495,12620,12746,
	12874,13002,13132,13264,13396,13530,13666,13802,
	13940,14080,14221,14363,14506,14652,14798,14946,
	15096,15247,15399,15553,15709,15866,16024,16184,
	16346,16510,16675,16842,17010,17180,17352,17526,
	17701,17878,18057,18237,18420,18604,18790,18978,
	19167,19359,19553,19748,19946,20145,20347,20550,
	20756,20963,21173,21385,21598,21814,22033,22253,
	22475,22700,22927,23156,23388,23622,23858,24097,
	24338,24581,24827,25075,25326,25579,25835,26093,
	26354,26618,26884,27153,27424,27699,27976,28255,
	28538,28823,29112,29403,29697,29994,30294,30597,
	30903,31212,31524,31839,32157,32479,32804,33132
};

static uint8_t accel_6500_self_test(long *bias_regular, long *bias_st, int debug)
{
    int i, result = 0, otp_value_zero = 0;
    float accel_st_al_min, accel_st_al_max;
    float st_shift_cust[3], st_shift_ratio[3], ct_shift_prod[3], accel_offset_max;
    unsigned char regs[3];
    if (i2c_read(HW_addr, REG_6500_XA_ST_DATA, 3, regs)) {
    	/*if(debug)
    		log_i("Reading OTP Register Error.\n");*/
    	return 0x07;
    }
    /*if(debug)
    	log_i("Accel OTP:%d, %d, %d\n", regs[0], regs[1], regs[2]);*/
	for (i = 0; i < 3; i++) {
		if (regs[i] != 0) {
			ct_shift_prod[i] = mpu_6500_st_tb[regs[i] - 1];
			ct_shift_prod[i] *= 65536.f;
			ct_shift_prod[i] /= TEST_accel_sens;
		}
		else {
			ct_shift_prod[i] = 0;
			otp_value_zero = 1;
		}
	}
	if(otp_value_zero == 0) {
		/*if(debug)
			log_i("ACCEL:CRITERIA A\n");*/
		for (i = 0; i < 3; i++) {
			st_shift_cust[i] = bias_st[i] - bias_regular[i];
			//if(debug) {
				/*log_i("Bias_Shift=%7.4f, Bias_Reg=%7.4f, Bias_HWST=%7.4f\r\n",
						st_shift_cust[i]/1.f, bias_regular[i]/1.f,
						bias_st[i]/1.f);
				log_i("OTP value: %7.4f\r\n", ct_shift_prod[i]/1.f); */
			//}

			st_shift_ratio[i] = st_shift_cust[i] / ct_shift_prod[i] - 1.f;

			//if(debug)
				/*log_i("ratio=%7.4f, threshold=%7.4f\r\n", st_shift_ratio[i]/1.f,
							test.max_accel_var/1.f); */

			if (fabs(st_shift_ratio[i]) > TEST_max_accel_var) {
				/*if(debug)
					log_i("ACCEL Fail Axis = %d\n", i);*/
				result |= 1 << i;	//Error condition
			}
		}
	}
	else {
		/* Self Test Pass/Fail Criteria B */
		accel_st_al_min = TEST_min_g * 65536.f;
		accel_st_al_max = TEST_max_g * 65536.f;

		//if(debug) {
			/*log_i("ACCEL:CRITERIA B\r\n");
			log_i("Min MG: %7.4f\r\n", accel_st_al_min/1.f);
			log_i("Max MG: %7.4f\r\n", accel_st_al_max/1.f);*/
		//}

		for (i = 0; i < 3; i++) {
			st_shift_cust[i] = bias_st[i] - bias_regular[i];

			//if(debug)
			//	log_i("Bias_shift=%7.4f, st=%7.4f, reg=%7.4f\n", st_shift_cust[i]/1.f, bias_st[i]/1.f, bias_regular[i]/1.f);
			if(st_shift_cust[i] < accel_st_al_min || st_shift_cust[i] > accel_st_al_max) {
				//if(debug)
				//	log_i("Accel FAIL axis:%d <= 225mg or >= 675mg\n", i);
				result |= 1 << i;	//Error condition
			}
		}
	}

	if(result == 0) {
	/* Self Test Pass/Fail Criteria C */
		accel_offset_max = TEST_max_g_offset * 65536.f;
		//if(debug)
		//	log_i("Accel:CRITERIA C: bias less than %7.4f\n", accel_offset_max/1.f);
		for (i = 0; i < 3; i++) {
			if(fabs(bias_regular[i]) > accel_offset_max) {
				//if(debug)
				//	log_i("FAILED: Accel axis:%d = %d > 500mg\n", i, bias_regular[i]);
				result |= 1 << i;	//Error condition
			}
		}
	}

    return result;
}

static uint8_t gyro_6500_self_test(long *bias_regular, long *bias_st, int debug)
{
    int i, result = 0, otp_value_zero = 0;
    float gyro_st_al_max;
    float st_shift_cust[3], st_shift_ratio[3], ct_shift_prod[3], gyro_offset_max;
    unsigned char regs[3];

    if (i2c_read(HW_addr, REG_6500_XG_ST_DATA, 3, regs)) {
    	//if(debug)
    		//log_i("Reading OTP Register Error.\n");
        return 0x07;
    }

    //if(debug)
    	//log_i("Gyro OTP:%d, %d, %d\r\n", regs[0], regs[1], regs[2]);

	for (i = 0; i < 3; i++) {
		if (regs[i] != 0) {
			ct_shift_prod[i] = mpu_6500_st_tb[regs[i] - 1];
			ct_shift_prod[i] *= 65536.f;
			ct_shift_prod[i] /= TEST_gyro_sens;
		}
		else {
			ct_shift_prod[i] = 0;
			otp_value_zero = 1;
		}
	}

	if(otp_value_zero == 0) {
		//if(debug)
			//log_i("GYRO:CRITERIA A\n");
		/* Self Test Pass/Fail Criteria A */
		for (i = 0; i < 3; i++) {
			st_shift_cust[i] = bias_st[i] - bias_regular[i];

			//if(debug) {
				/*log_i("Bias_Shift=%7.4f, Bias_Reg=%7.4f, Bias_HWST=%7.4f\r\n",
						st_shift_cust[i]/1.f, bias_regular[i]/1.f,
						bias_st[i]/1.f);
				log_i("OTP value: %7.4f\r\n", ct_shift_prod[i]/1.f);*/
			//}

			st_shift_ratio[i] = st_shift_cust[i] / ct_shift_prod[i];

			//if(debug)
				/*log_i("ratio=%7.4f, threshold=%7.4f\r\n", st_shift_ratio[i]/1.f,
							test.max_gyro_var/1.f);*/

			if (fabs(st_shift_ratio[i]) < TEST_max_gyro_var) {
				//if(debug)
					//log_i("Gyro Fail Axis = %d\n", i);
				result |= 1 << i;	//Error condition
			}
		}
	}
	else {
		/* Self Test Pass/Fail Criteria B */
		gyro_st_al_max = TEST_max_dps * 65536.f;

		/*if(debug) {
			log_i("GYRO:CRITERIA B\r\n");
			log_i("Max DPS: %7.4f\r\n", gyro_st_al_max/1.f);
		}*/

		for (i = 0; i < 3; i++) {
			st_shift_cust[i] = bias_st[i] - bias_regular[i];

			/*if(debug)
				log_i("Bias_shift=%7.4f, st=%7.4f, reg=%7.4f\n", st_shift_cust[i]/1.f, bias_st[i]/1.f, bias_regular[i]/1.f);*/
			if(st_shift_cust[i] < gyro_st_al_max) {
				/*if(debug)
					log_i("GYRO FAIL axis:%d greater than 60dps\n", i);*/
				result |= 1 << i;	//Error condition
			}
		}
	}

	if(result == 0) {
	/* Self Test Pass/Fail Criteria C */
		gyro_offset_max = TEST_min_dps * 65536.f;
		/*if(debug)
			log_i("Gyro:CRITERIA C: bias less than %7.4f\n", gyro_offset_max/1.f);*/
		for (i = 0; i < 3; i++) {
			if(fabs(bias_regular[i]) > gyro_offset_max) {
				/*if(debug)
					log_i("FAILED: Gyro axis:%d = %d > 20dps\n", i, bias_regular[i]);*/
				result |= 1 << i;	//Error condition
			}
		}
	}
    return result;
}

static uint8_t get_st_6500_biases(long *gyro, long *accel, unsigned char hw_test, int debug)
{
    unsigned char data[HWST_MAX_PACKET_LENGTH];
    unsigned char packet_count, ii;
    unsigned short fifo_count;
    int s = 0, read_size = 0, ind;

    data[0] = 0x01;
    data[1] = 0;
    if (i2c_write(HW_addr, REG_pwr_mgmt_1, 2, data))
        return 1;
    delay_ms(200);
    data[0] = 0;
    if (i2c_write(HW_addr, REG_int_enable, 1, data))
        return 2;
    if (i2c_write(HW_addr, REG_fifo_en, 1, data))
        return 3;
    if (i2c_write(HW_addr, REG_pwr_mgmt_1, 1, data))
        return 4;
    if (i2c_write(HW_addr, REG_i2c_mst, 1, data))
        return 5;
    if (i2c_write(HW_addr, REG_user_ctrl, 1, data))
        return 6;
    data[0] = BIT_FIFO_RST | BIT_DMP_RST;
    if (i2c_write(HW_addr, REG_user_ctrl, 1, data))
        return 7;
    delay_ms(15);
    data[0] = TEST_reg_lpf;
    if (i2c_write(HW_addr, REG_lpf, 1, data))
        return 8;
    data[0] = TEST_reg_rate_div;
    if (i2c_write(HW_addr, REG_rate_div, 1, data))
        return 9;
    if (hw_test)
        data[0] = TEST_reg_gyro_fsr | 0xE0;
    else
        data[0] = TEST_reg_gyro_fsr;
    if (i2c_write(HW_addr, REG_gyro_cfg, 1, data))
        return 10;

    if (hw_test)
        data[0] = TEST_reg_accel_fsr | 0xE0;
    else
        data[0] = TEST_reg_accel_fsr;
    if (i2c_write(HW_addr, REG_accel_cfg, 1, data))
        return 11;

    delay_ms(TEST_wait_ms);  //wait 200ms for sensors to stabilize

    /* Enable FIFO */
    data[0] = BIT_FIFO_EN;
    if (i2c_write(HW_addr, REG_user_ctrl, 1, data))
        return 12;
    data[0] = INV_XYZ_GYRO | INV_XYZ_ACCEL;
    if (i2c_write(HW_addr, REG_fifo_en, 1, data))
        return 13;

    //initialize the bias return values
    gyro[0] = gyro[1] = gyro[2] = 0;
    accel[0] = accel[1] = accel[2] = 0;

    if(debug)
    	//log_i("Starting Bias Loop Reads\n");

    //start reading samples
    while (s < TEST_packet_thresh) {
    	delay_ms(TEST_sample_wait_ms); //wait 10ms to fill FIFO
		if (i2c_read(HW_addr, REG_fifo_count_h, 2, data))
			return 14;
		fifo_count = (data[0] << 8) | data[1];
		packet_count = fifo_count / MAX_PACKET_LENGTH;
		if ((TEST_packet_thresh - s) < packet_count)
		            packet_count = TEST_packet_thresh - s;
		read_size = packet_count * MAX_PACKET_LENGTH;

		//burst read from FIFO
		if (i2c_read(HW_addr, REG_fifo_r_w, read_size, data))
						return 15;
		ind = 0;
		for (ii = 0; ii < packet_count; ii++) {
			short accel_cur[3], gyro_cur[3];
			accel_cur[0] = ((short)data[ind + 0] << 8) | data[ind + 1];
			accel_cur[1] = ((short)data[ind + 2] << 8) | data[ind + 3];
			accel_cur[2] = ((short)data[ind + 4] << 8) | data[ind + 5];
			accel[0] += (long)accel_cur[0];
			accel[1] += (long)accel_cur[1];
			accel[2] += (long)accel_cur[2];
			gyro_cur[0] = (((short)data[ind + 6] << 8) | data[ind + 7]);
			gyro_cur[1] = (((short)data[ind + 8] << 8) | data[ind + 9]);
			gyro_cur[2] = (((short)data[ind + 10] << 8) | data[ind + 11]);
			gyro[0] += (long)gyro_cur[0];
			gyro[1] += (long)gyro_cur[1];
			gyro[2] += (long)gyro_cur[2];
			ind += MAX_PACKET_LENGTH;
		}
		s += packet_count;
    }

    if(debug)
    	//log_i("Samples: %d\n", s);

    //stop FIFO
    data[0] = 0;
    if (i2c_write(HW_addr, REG_fifo_en, 1, data))
        return 16;

    gyro[0] = (long)(((long long)gyro[0]<<16) / TEST_gyro_sens / s);
    gyro[1] = (long)(((long long)gyro[1]<<16) / TEST_gyro_sens / s);
    gyro[2] = (long)(((long long)gyro[2]<<16) / TEST_gyro_sens / s);
    accel[0] = (long)(((long long)accel[0]<<16) / TEST_accel_sens / s);
    accel[1] = (long)(((long long)accel[1]<<16) / TEST_accel_sens / s);
    accel[2] = (long)(((long long)accel[2]<<16) / TEST_accel_sens / s);
    /* remove gravity from bias calculation */
    if (accel[2] > 0L)
        accel[2] -= 65536L;
    else
        accel[2] += 65536L;


    if(debug) {
    	//log_i("Accel offset data HWST bit=%d: %7.4f %7.4f %7.4f\r\n", hw_test, accel[0]/65536.f, accel[1]/65536.f, accel[2]/65536.f);
    	//log_i("Gyro offset data HWST bit=%d: %7.4f %7.4f %7.4f\r\n", hw_test, gyro[0]/65536.f, gyro[1]/65536.f, gyro[2]/65536.f);
    }

    return 0;
}

/**
 *  @brief      Trigger gyro/accel/compass self-test for MPU6500/MPU9250
 *  On success/error, the self-test returns a mask representing the sensor(s)
 *  that failed. For each bit, a one (1) represents a "pass" case; conversely,
 *  a zero (0) indicates a failure.
 *
 *  \n The mask is defined as follows:
 *  \n Bit 0:   Gyro.
 *  \n Bit 1:   Accel.
 *  \n Bit 2:   Compass.
 *
 *  @param[out] gyro        Gyro biases in q16 format.
 *  @param[out] accel       Accel biases (if applicable) in q16 format.
 *  @param[in]  debug       Debug flag used to print out more detailed logs. Must first set up logging in Motion Driver.
 *  @return     Result mask (see above).
 */
uint8_t MPUDriver::mpu_run_6500_self_test(long *gyro, long *accel, unsigned char debug)
{
    const unsigned char tries = 2;
    long gyro_st[3], accel_st[3];
    unsigned char accel_result, gyro_result;
#ifdef AK89xx_SECONDARY
    unsigned char compass_result;
#endif
    uint8_t ii;

    uint8_t result;
    unsigned char accel_fsr, fifo_sensors, sensors_on;
    unsigned short gyro_fsr, sample_rate, lpf;
    unsigned char dmp_was_on;



    if(debug)
    	//log_i("Starting MPU6500 HWST!\r\n");

    if (chip_cfg.dmp_on) {
        mpu_set_dmp_state(0);
        dmp_was_on = 1;
    } else
        dmp_was_on = 0;

    /* Get initial settings. */
    mpu_get_gyro_fsr(&gyro_fsr);
    mpu_get_accel_fsr(&accel_fsr);
    mpu_get_lpf(&lpf);
    mpu_get_sample_rate(&sample_rate);
    sensors_on = chip_cfg.sensors;
    mpu_get_fifo_config(&fifo_sensors);

    if(debug)
    	//log_i("Retrieving Biases\r\n");

    for (ii = 0; ii < tries; ii++)
        if (!get_st_6500_biases(gyro, accel, 0, debug))
            break;
    if (ii == tries) {
        /* If we reach this point, we most likely encountered an I2C error.
         * We'll just report an error for all three sensors.
         */
        if(debug)
        	//log_i("Retrieving Biases Error - possible I2C error\n");

        result = 0;
        goto restore;
    }

    if(debug)
    	//log_i("Retrieving ST Biases\n");

    for (ii = 0; ii < tries; ii++)
        if (!get_st_6500_biases(gyro_st, accel_st, 1, debug))
            break;
    if (ii == tries) {

        if(debug)
        	//log_i("Retrieving ST Biases Error - possible I2C error\n");

        /* Again, probably an I2C error. */
        result = 0;
        goto restore;
    }

    accel_result = accel_6500_self_test(accel, accel_st, debug);
    if(debug)
    	//log_i("Accel Self Test Results: %d\n", accel_result);

    gyro_result = gyro_6500_self_test(gyro, gyro_st, debug);
    if(debug)
    	//log_i("Gyro Self Test Results: %d\n", gyro_result);

    result = 0;
    if (!gyro_result)
        result |= 0x01;
    if (!accel_result)
        result |= 0x02;

#ifdef AK89xx_SECONDARY
    compass_result = compass_self_test();
    if(debug)
    	log_i("Compass Self Test Results: %d\n", compass_result);
    if (!compass_result)
        result |= 0x04;
#else
    result |= 0x04;
#endif
restore:
	if(debug)
		//log_i("Exiting HWST\n");
	/* Set to invalid values to ensure no I2C writes are skipped. */
	chip_cfg.gyro_fsr = 0xFF;
	chip_cfg.accel_fsr = 0xFF;
	chip_cfg.lpf = 0xFF;
	chip_cfg.sample_rate = 0xFFFF;
	chip_cfg.sensors = 0xFF;
	chip_cfg.fifo_enable = 0xFF;
	chip_cfg.clk_src = INV_CLK_PLL;
	mpu_set_gyro_fsr(gyro_fsr);
	mpu_set_accel_fsr(accel_fsr);
	mpu_set_lpf(lpf);
	mpu_set_sample_rate(sample_rate);
	mpu_set_sensors(sensors_on);
	mpu_configure_fifo(fifo_sensors);

	if (dmp_was_on)
		mpu_set_dmp_state(1);

	return result;
}
#endif // end #ifdef MPU6500

/*
*  \n This function must be called with the device either face-up or face-down
*  (z-axis is parallel to gravity).
*  @param[out] gyro        Gyro biases in q16 format.
*  @param[out] accel       Accel biases (if applicable) in q16 format.
*  @return     Result mask (see above).
*/
uint8_t MPUDriver::mpu_run_self_test(long *gyro, long *accel)
{
#ifdef MPU6050
   const unsigned char tries = 2;
   long gyro_st[3], accel_st[3];
   unsigned char accel_result, gyro_result;
#ifdef AK89xx_SECONDARY
   unsigned char compass_result;
#endif
   uint8_t ii;
#endif
   uint8_t result;
   unsigned char accel_fsr, fifo_sensors, sensors_on;
   unsigned short gyro_fsr, sample_rate, lpf;
   unsigned char dmp_was_on;

   if (chip_cfg.dmp_on) {
       mpu_set_dmp_state(0);
       dmp_was_on = 1;
   } else
       dmp_was_on = 0;

   /* Get initial settings. */
   mpu_get_gyro_fsr(&gyro_fsr);
   mpu_get_accel_fsr(&accel_fsr);
   mpu_get_lpf(&lpf);
   mpu_get_sample_rate(&sample_rate);
   sensors_on = chip_cfg.sensors;
   mpu_get_fifo_config(&fifo_sensors);

   /* For older chips, the self-test will be different. */
#if defined MPU6050
   for (ii = 0; ii < tries; ii++)
       if (!get_st_biases(gyro, accel, 0))
           break;
   if (ii == tries) {
       /* If we reach this point, we most likely encountered an I2C error.
        * We'll just report an error for all three sensors.
        */
       result = 0;
       goto restore;
   }
   for (ii = 0; ii < tries; ii++)
       if (!get_st_biases(gyro_st, accel_st, 1))
           break;
   if (ii == tries) {
       /* Again, probably an I2C error. */
       result = 0;
       goto restore;
   }
   accel_result = accel_self_test(accel, accel_st);
   gyro_result = gyro_self_test(gyro, gyro_st);

   result = 0;
   if (!gyro_result)
       result |= 0x01;
   if (!accel_result)
       result |= 0x02;

#ifdef AK89xx_SECONDARY
   compass_result = compass_self_test();
   if (!compass_result)
       result |= 0x04;
#else
       result |= 0x04;
#endif
restore:
#elif defined MPU6500
   /* For now, this function will return a "pass" result for all three sensors
    * for compatibility with current test applications.
    */
   get_st_biases(gyro, accel, 0);
   result = 0x7;
#endif
   /* Set to invalid values to ensure no I2C writes are skipped. */
   chip_cfg.gyro_fsr = 0xFF;
   chip_cfg.accel_fsr = 0xFF;
   chip_cfg.lpf = 0xFF;
   chip_cfg.sample_rate = 0xFFFF;
   chip_cfg.sensors = 0xFF;
   chip_cfg.fifo_enable = 0xFF;
   chip_cfg.clk_src = INV_CLK_PLL;
   mpu_set_gyro_fsr(gyro_fsr);
   mpu_set_accel_fsr(accel_fsr);
   mpu_set_lpf(lpf);
   mpu_set_sample_rate(sample_rate);
   mpu_set_sensors(sensors_on);
   mpu_configure_fifo(fifo_sensors);

   if (dmp_was_on)
       mpu_set_dmp_state(1);

   return result;
}

/**
 *  @brief      Write to the DMP memory.
 *  This function prevents I2C writes past the bank boundaries. The DMP memory
 *  is only accessible when the chip is awake.
 *  @param[in]  mem_addr    Memory location (bank << 8 | start address)
 *  @param[in]  length      Number of bytes to write.
 *  @param[in]  data        Bytes to write to memory.
 *  @return     0 if successful.
 */
uint8_t MPUDriver::mpu_write_mem(unsigned short mem_addr, unsigned short length,
        unsigned char *data)
{
    unsigned char tmp[2];
	
    if (!data)
        return 4;
    if (!chip_cfg.sensors)
        return 5;

	//return data[2];
	
    tmp[0] = (unsigned char)(mem_addr >> 8);
    tmp[1] = (unsigned char)(mem_addr & 0xFF);
	
	//return data[2];
	
    /* Check bank boundaries. */
    if (tmp[1] + length > HW_bank_size)
        return 1;
	
    if (i2c_write(HW_addr, REG_bank_sel, 2, tmp))
        return 2;
    if (i2c_write(HW_addr, REG_mem_r_w, length, data))
        return 3;
	
    return 0;
}

/**
 *  @brief      Read from the DMP memory.
 *  This function prevents I2C reads past the bank boundaries. The DMP memory
 *  is only accessible when the chip is awake.
 *  @param[in]  mem_addr    Memory location (bank << 8 | start address)
 *  @param[in]  length      Number of bytes to read.
 *  @param[out] data        Bytes read from memory.
 *  @return     0 if successful.
 */
uint8_t MPUDriver::mpu_read_mem(unsigned short mem_addr, unsigned short length,
        unsigned char *data)
{
    unsigned char tmp[2];

    if (!data)
        return 1;
    if (!chip_cfg.sensors)
        return 2;

    tmp[0] = (unsigned char)(mem_addr >> 8);
    tmp[1] = (unsigned char)(mem_addr & 0xFF);

    /* Check bank boundaries. */
    if (tmp[1] + length > HW_bank_size)
        return 3;

    if (i2c_write(HW_addr, REG_bank_sel, 2, tmp))
        return 4;
    if (i2c_read(HW_addr, REG_mem_r_w, length, data))
        return 5;
    return 0;
}

/**
 *  @brief      Load and verify DMP image.
 *  @param[in]  length      Length of DMP image.
 *  @param[in]  firmware    DMP code.
 *  @param[in]  start_addr  Starting address of DMP code memory.
 *  @param[in]  sample_rate Fixed sampling rate used when DMP is enabled.
 *  @return     0 if successful.
 */
uint8_t MPUDriver::mpu_load_firmware(unsigned short length, const unsigned char *firmware,
    unsigned short start_addr, unsigned short sample_rate)
{	

    unsigned short ii;
	unsigned short jj;
    unsigned short this_write;
    /* Must divide evenly into st.hw->bank_size to avoid bank crossings. */
#define LOAD_CHUNK  (16)
    unsigned char cur[LOAD_CHUNK], dmpBuf[LOAD_CHUNK], tmp[2];
	
    if (chip_cfg.dmp_loaded)
        /* DMP should only be loaded once. */
        return 1;

    if (!firmware)
        return 2;
	
    for (ii = 0; ii < length; ii += this_write) {
        this_write = min(LOAD_CHUNK, length - ii);
		
		for(jj = 0 ; jj < this_write; jj++){dmpBuf[jj] = pgm_read_byte_near(firmware + ii + jj);}
		
		
        if (mpu_write_mem(ii, this_write,  dmpBuf))  //(unsigned char*)&firmware[ii])
            return 3;
		
        if (mpu_read_mem(ii, this_write, cur))
            return 4;
		
        if (memcmp(dmpBuf, cur, this_write))
            return 5;
		
    }

    /* Set program start address. */
    tmp[0] = start_addr >> 8;
    tmp[1] = start_addr & 0xFF;
    if (i2c_write(HW_addr, REG_prgm_start_h, 2, tmp))
        return 6;

    chip_cfg.dmp_loaded = 1;
    chip_cfg.dmp_sample_rate = sample_rate;
    return 0;
}

/**
 *  @brief      Enable/disable DMP support.
 *  @param[in]  enable  1 to turn on the DMP.
 *  @return     0 if successful.
 */
uint8_t MPUDriver::mpu_set_dmp_state(unsigned char enable)
{
    unsigned char tmp;
    if (chip_cfg.dmp_on == enable)
        return 0;

    if (enable) {
        if (!chip_cfg.dmp_loaded)
            return 1;
        /* Disable data ready interrupt. */
        set_int_enable(0);
        /* Disable bypass mode. */
        mpu_set_bypass(0);
        /* Keep constant sample rate, FIFO rate controlled by DMP. */
        mpu_set_sample_rate(chip_cfg.dmp_sample_rate);
        /* Remove FIFO elements. */
        tmp = 0;
        i2c_write(HW_addr, 0x23, 1, &tmp);
        chip_cfg.dmp_on = 1;
        /* Enable DMP interrupt. */
        set_int_enable(1);
        mpu_reset_fifo();
    } else {
        /* Disable DMP interrupt. */
        set_int_enable(0);
        /* Restore FIFO settings. */
        tmp = chip_cfg.fifo_enable;
        i2c_write(HW_addr, 0x23, 1, &tmp);
        chip_cfg.dmp_on = 0;
        mpu_reset_fifo();
    }
    return 0;
}

/**
 *  @brief      Get DMP state.
 *  @param[out] enabled 1 if enabled.
 *  @return     0 if successful.
 */
void MPUDriver::mpu_get_dmp_state(unsigned char *enabled)
{
    enabled[0] = chip_cfg.dmp_on;
}

#ifdef AK89xx_SECONDARY
/* This initialization is similar to the one in ak8975.c. */
static int setup_compass(void)
{
    unsigned char data[4], akm_addr;

    mpu_set_bypass(1);

    /* Find compass. Possible addresses range from 0x0C to 0x0F. */
    for (akm_addr = 0x0C; akm_addr <= 0x0F; akm_addr++) {
        int result;
        result = i2c_read(akm_addr, AKM_REG_WHOAMI, 1, data);
        if (!result && (data[0] == AKM_WHOAMI))
            break;
    }

    if (akm_addr > 0x0F) {
        /* TODO: Handle this case in all compass-related functions. */
        //log_e("Compass not found.\n");
        return -1;
    }

    chip_cfg.compass_addr = akm_addr;

    data[0] = AKM_POWER_DOWN;
    if (i2c_write(chip_cfg.compass_addr, AKM_REG_CNTL, 1, data))
        return -1;
    delay_ms(1);

    data[0] = AKM_FUSE_ROM_ACCESS;
    if (i2c_write(chip_cfg.compass_addr, AKM_REG_CNTL, 1, data))
        return -1;
    delay_ms(1);

    /* Get sensitivity adjustment data from fuse ROM. */
    if (i2c_read(chip_cfg.compass_addr, AKM_REG_ASAX, 3, data))
        return -1;
    chip_cfg.mag_sens_adj[0] = (long)data[0] + 128;
    chip_cfg.mag_sens_adj[1] = (long)data[1] + 128;
    chip_cfg.mag_sens_adj[2] = (long)data[2] + 128;

    data[0] = AKM_POWER_DOWN;
    if (i2c_write(chip_cfg.compass_addr, AKM_REG_CNTL, 1, data))
        return -1;
    delay_ms(1);

    mpu_set_bypass(0);

    /* Set up master mode, master clock, and ES bit. */
    data[0] = 0x40;
    if (i2c_write(HW_addr, REG_i2c_mst, 1, data))
        return -1;

    /* Slave 0 reads from AKM data registers. */
    data[0] = BIT_I2C_READ | chip_cfg.compass_addr;
    if (i2c_write(HW_addr, REG_s0_addr, 1, data))
        return -1;

    /* Compass reads start at this register. */
    data[0] = AKM_REG_ST1;
    if (i2c_write(HW_addr, REG_s0_reg, 1, data))
        return -1;

    /* Enable slave 0, 8-byte reads. */
    data[0] = BIT_SLAVE_EN | 8;
    if (i2c_write(HW_addr, REG_s0_ctrl, 1, data))
        return -1;

    /* Slave 1 changes AKM measurement mode. */
    data[0] = chip_cfg.compass_addr;
    if (i2c_write(HW_addr, REG_s1_addr, 1, data))
        return -1;

    /* AKM measurement mode register. */
    data[0] = AKM_REG_CNTL;
    if (i2c_write(HW_addr, REG_s1_reg, 1, data))
        return -1;

    /* Enable slave 1, 1-byte writes. */
    data[0] = BIT_SLAVE_EN | 1;
    if (i2c_write(HW_addr, REG_s1_ctrl, 1, data))
        return -1;

    /* Set slave 1 data. */
    data[0] = AKM_SINGLE_MEASUREMENT;
    if (i2c_write(HW_addr, REG_s1_do, 1, data))
        return -1;

    /* Trigger slave 0 and slave 1 actions at each sample. */
    data[0] = 0x03;
    if (i2c_write(HW_addr, REG_i2c_delay_ctrl, 1, data))
        return -1;

#ifdef MPU9150
    /* For the MPU9150, the auxiliary I2C bus needs to be set to VDD. */
    data[0] = BIT_I2C_MST_VDDIO;
    if (i2c_write(st.hw->addr, st.reg->yg_offs_tc, 1, data))
        return -1;
#endif

    return 0;
}
#endif //end #ifdef AK89xx_SECONDARY

/**
 *  @brief      Read raw compass data.
 *  @param[out] data        Raw data in hardware units.
 *  @param[out] timestamp   Timestamp in milliseconds. Null if not needed.
 *  @return     0 if successful.
 */
// void MPUDriver::mpu_get_compass_reg(short *data, unsigned long *timestamp)
// {
// #ifdef AK89xx_SECONDARY
    // unsigned char tmp[9];

    // if (!(st.chip_cfg.sensors & INV_XYZ_COMPASS))
        // return -1;

// #ifdef AK89xx_BYPASS
    // if (i2c_read(st.chip_cfg.compass_addr, AKM_REG_ST1, 8, tmp))
        // return -1;
    // tmp[8] = AKM_SINGLE_MEASUREMENT;
    // if (i2c_write(st.chip_cfg.compass_addr, AKM_REG_CNTL, 1, tmp+8))
        // return -1;
// #else
    // if (i2c_read(st.hw->addr, st.reg->raw_compass, 8, tmp))
        // return -1;
// #endif

// #if defined AK8975_SECONDARY
    // /* AK8975 doesn't have the overrun error bit. */
    // if (!(tmp[0] & AKM_DATA_READY))
        // return -2;
    // if ((tmp[7] & AKM_OVERFLOW) || (tmp[7] & AKM_DATA_ERROR))
        // return -3;
// #elif defined AK8963_SECONDARY
    // /* AK8963 doesn't have the data read error bit. */
    // if (!(tmp[0] & AKM_DATA_READY) || (tmp[0] & AKM_DATA_OVERRUN))
        // return -2;
    // if (tmp[7] & AKM_OVERFLOW)
        // return -3;
// #endif
    // data[0] = (tmp[2] << 8) | tmp[1];
    // data[1] = (tmp[4] << 8) | tmp[3];
    // data[2] = (tmp[6] << 8) | tmp[5];

    // data[0] = ((long)data[0] * st.chip_cfg.mag_sens_adj[0]) >> 8;
    // data[1] = ((long)data[1] * st.chip_cfg.mag_sens_adj[1]) >> 8;
    // data[2] = ((long)data[2] * st.chip_cfg.mag_sens_adj[2]) >> 8;

    // if (timestamp)
        // get_ms(timestamp);
    // return 0;
// #else
    // return -1;
// #endif
// }

/**
 *  @brief      Get the compass full-scale range.
 *  @param[out] fsr Current full-scale range.
 *  @return     0 if successful.
 */
// void MPUDriver::mpu_get_compass_fsr(unsigned short *fsr)
// {
// #ifdef AK89xx_SECONDARY
    // fsr[0] = st.hw->compass_fsr;
    // return 0;
// #else
    // return -1;
// #endif
// }

/**
 *  @brief      Enters LP accel motion interrupt mode.
 *  The behaviour of this feature is very different between the MPU6050 and the
 *  MPU6500. Each chip's version of this feature is explained below.
 *
 *  \n The hardware motion threshold can be between 32mg and 8160mg in 32mg
 *  increments.
 *
 *  \n Low-power accel mode supports the following frequencies:
 *  \n 1.25Hz, 5Hz, 20Hz, 40Hz
 *
 *  \n MPU6500:
 *  \n Unlike the MPU6050 version, the hardware does not "lock in" a reference
 *  sample. The hardware monitors the accel data and detects any large change
 *  over a short period of time.
 *
 *  \n The hardware motion threshold can be between 4mg and 1020mg in 4mg
 *  increments.
 *
 *  \n MPU6500 Low-power accel mode supports the following frequencies:
 *  \n 0.24Hz, 0.49Hz, 0.98Hz, 1.95Hz, 3.91Hz, 7.81Hz, 15.63Hz, 31.25Hz, 62.5Hz, 125Hz, 250Hz, 500Hz
 *
 *  \n\n NOTES:
 *  \n The driver will round down @e thresh to the nearest supported value if
 *  an unsupported threshold is selected.
 *  \n To select a fractional wake-up frequency, round down the value passed to
 *  @e lpa_freq.
 *  \n The MPU6500 does not support a delay parameter. If this function is used
 *  for the MPU6500, the value passed to @e time will be ignored.
 *  \n To disable this mode, set @e lpa_freq to zero. The driver will restore
 *  the previous configuration.
 *
 *  @param[in]  thresh      Motion threshold in mg.
 *  @param[in]  time        Duration in milliseconds that the accel data must
 *                          exceed @e thresh before motion is reported.
 *  @param[in]  lpa_freq    Minimum sampling rate, or zero to disable.
 *  @return     0 if successful.
 */
uint8_t MPUDriver::mpu_lp_motion_interrupt(unsigned short thresh, unsigned char time,
    unsigned char lpa_freq)
{

#if defined MPU6500
    unsigned char data[3];
#endif
    if (lpa_freq) {
#if defined MPU6500
    	unsigned char thresh_hw;

        /* 1LSb = 4mg. */
        if (thresh > 1020)
            thresh_hw = 255;
        else if (thresh < 4)
            thresh_hw = 1;
        else
            thresh_hw = thresh >> 2;
#endif

        if (!time)
            /* Minimum duration must be 1ms. */
            time = 1;

#if defined MPU6500
        if (lpa_freq > 500)
            /* At this point, the chip has not been re-configured, so the
             * function can safely exit.
             */
            return 1;
#endif

        if (!chip_cfg.int_motion_only) {
            /* Store current settings for later. */
            if (chip_cfg.dmp_on) {
                mpu_set_dmp_state(0);
                chip_cfg.cache.dmp_on = 1;
            } else
                chip_cfg.cache.dmp_on = 0;
            mpu_get_gyro_fsr(&chip_cfg.cache.gyro_fsr);
            mpu_get_accel_fsr(&chip_cfg.cache.accel_fsr);
            mpu_get_lpf(&chip_cfg.cache.lpf);
            mpu_get_sample_rate(&chip_cfg.cache.sample_rate);
            chip_cfg.cache.sensors_on = chip_cfg.sensors;
            mpu_get_fifo_config(&chip_cfg.cache.fifo_sensors);
        }

#if defined MPU6500
        /* Disable hardware interrupts. */
        set_int_enable(0);

        /* Enter full-power accel-only mode, no FIFO/DMP. */
        data[0] = 0;
        data[1] = 0;
        data[2] = BIT_STBY_XYZG;
        if (i2c_write(HW_addr, REG_user_ctrl, 3, data))
            goto lp_int_restore;

        /* Set motion threshold. */
        data[0] = thresh_hw;
        if (i2c_write(HW_addr, REG_motion_thr, 1, data))
            goto lp_int_restore;

        /* Set wake frequency. */
        if (lpa_freq == 1)
            data[0] = INV_LPA_0_98HZ;
        else if (lpa_freq == 2)
            data[0] = INV_LPA_1_95HZ;
        else if (lpa_freq <= 5)
            data[0] = INV_LPA_3_91HZ;
        else if (lpa_freq <= 10)
            data[0] = INV_LPA_7_81HZ;
        else if (lpa_freq <= 20)
            data[0] = INV_LPA_15_63HZ;
        else if (lpa_freq <= 40)
            data[0] = INV_LPA_31_25HZ;
        else if (lpa_freq <= 70)
            data[0] = INV_LPA_62_50HZ;
        else if (lpa_freq <= 125)
            data[0] = INV_LPA_125HZ;
        else if (lpa_freq <= 250)
            data[0] = INV_LPA_250HZ;
        else
            data[0] = INV_LPA_500HZ;
        if (i2c_write(HW_addr, REG_lp_accel_odr, 1, data))
            goto lp_int_restore;

        /* Enable motion interrupt (MPU6500 version). */
        data[0] = BITS_WOM_EN;
        if (i2c_write(HW_addr, REG_accel_intel, 1, data))
            goto lp_int_restore;

        /* Bypass DLPF ACCEL_FCHOICE_B=1*/
        data[0] = BIT_ACCL_FC_B | 0x01;
        if (i2c_write(HW_addr, REG_accel_cfg2, 1, data))
            goto lp_int_restore;

        /* Enable interrupt. */
        data[0] = BIT_MOT_INT_EN;
        if (i2c_write(HW_addr, REG_int_enable, 1, data))
            goto lp_int_restore;

        /* Enable cycle mode. */
        data[0] = BIT_LPA_CYCLE;
        if (i2c_write(HW_addr, REG_pwr_mgmt_1, 1, data))
            goto lp_int_restore;

        chip_cfg.int_motion_only = 1;
        return 0;
#endif
    } else {
        /* Don't "restore" the previous state if no state has been saved. */
        int ii;
        char *cache_ptr = (char*)&chip_cfg.cache;
        for (ii = 0; ii < sizeof(chip_cfg.cache); ii++) {
            if (cache_ptr[ii] != 0)
                goto lp_int_restore;
        }
        /* If we reach this point, motion interrupt mode hasn't been used yet. */
        return 2;
    }
lp_int_restore:
    /* Set to invalid values to ensure no I2C writes are skipped. */
    chip_cfg.gyro_fsr = 0xFF;
    chip_cfg.accel_fsr = 0xFF;
    chip_cfg.lpf = 0xFF;
    chip_cfg.sample_rate = 0xFFFF;
    chip_cfg.sensors = 0xFF;
    chip_cfg.fifo_enable = 0xFF;
    chip_cfg.clk_src = INV_CLK_PLL;
    mpu_set_sensors(chip_cfg.cache.sensors_on);
    mpu_set_gyro_fsr(chip_cfg.cache.gyro_fsr);
    mpu_set_accel_fsr(chip_cfg.cache.accel_fsr);
    mpu_set_lpf(chip_cfg.cache.lpf);
    mpu_set_sample_rate(chip_cfg.cache.sample_rate);
    mpu_configure_fifo(chip_cfg.cache.fifo_sensors);

    if (chip_cfg.cache.dmp_on)
        mpu_set_dmp_state(1);

#ifdef MPU6500
    /* Disable motion interrupt (MPU6500 version). */
    data[0] = 0;
    if (i2c_write(HW_addr, REG_accel_intel, 1, data))
        goto lp_int_restore;
#endif

    chip_cfg.int_motion_only = 0;
    return 0;
}

uint8_t MPUDriver::set_int_enable(unsigned char enable)
{
    unsigned char tmp;

    if (chip_cfg.dmp_on) {
        if (enable)
            tmp = BIT_DMP_INT_EN;
        else
            tmp = 0x00;
        if (i2c_write(HW_addr, REG_int_enable, 1, &tmp))
            return 1;
        chip_cfg.int_enable = tmp;
    } else {
        if (!chip_cfg.sensors)
            return 2;
        if (enable && chip_cfg.int_enable)
            return 0;
        if (enable)
            tmp = BIT_DATA_RDY_EN;
        else
            tmp = 0x00;
        if (i2c_write(HW_addr, REG_int_enable, 1, &tmp))
            return 3;
        chip_cfg.int_enable = tmp;
    }
    return 0;
}

