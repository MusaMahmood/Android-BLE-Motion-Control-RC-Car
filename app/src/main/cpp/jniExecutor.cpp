//
// Created by mahmoodms on 4/3/2017.
//

#include "rt_nonfinite.h"
#include "ssvep_filter_f32.h"
#include "downsample_250Hz.h"
#include "ecg_bandstop_250Hz.h"

/*Additional Includes*/
#include <jni.h>
#include <android/log.h>

#define  LOG_TAG "jniExecutor-cpp"
#define  LOGE(...)  __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

extern "C" {
JNIEXPORT jint JNICALL
Java_com_yeolabgt_mahmoodms_wearablemotioncontrol_DeviceControlActivity_jmainInitialization(
        JNIEnv *env, jobject obj, jboolean initialize) {
    if (!(bool) initialize) {
//        downsample_250Hz_initialize();
//        ecg_bandstop_250Hz_initialize();
        return 0;
    } else {
        return -1;
    }
}
}
