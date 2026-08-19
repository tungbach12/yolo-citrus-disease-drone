// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2021 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include <android/asset_manager_jni.h>
#include <android/native_window_jni.h>
#include <android/native_window.h>

#include <android/log.h>

#include <jni.h>

#include <string>
#include <vector>

#include <platform.h>
#include <benchmark.h>

#include "yolov8.h"

#include "ndkcamera.h"

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#if __ARM_NEON
#include <arm_neon.h>
#endif // __ARM_NEON

static int draw_unsupported(cv::Mat& rgb)
{
    const char text[] = "unsupported";

    int baseLine = 0;
    cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 1.0, 1, &baseLine);

    int y = (rgb.rows - label_size.height) / 2;
    int x = (rgb.cols - label_size.width) / 2;

    cv::rectangle(rgb, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
                    cv::Scalar(255, 255, 255), -1);

    cv::putText(rgb, text, cv::Point(x, y + label_size.height),
                cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 0));

    return 0;
}

static int draw_fps(cv::Mat& rgb)
{
    // resolve moving average
    float avg_fps = 0.f;
    {
        static double t0 = 0.f;
        static float fps_history[10] = {0.f};

        double t1 = ncnn::get_current_time();
        if (t0 == 0.f)
        {
            t0 = t1;
            return 0;
        }

        float fps = 1000.f / (t1 - t0);
        t0 = t1;

        for (int i = 9; i >= 1; i--)
        {
            fps_history[i] = fps_history[i - 1];
        }
        fps_history[0] = fps;

        if (fps_history[9] == 0.f)
        {
            return 0;
        }

        for (int i = 0; i < 10; i++)
        {
            avg_fps += fps_history[i];
        }
        avg_fps /= 10.f;
    }

    char text[32];
    sprintf(text, "FPS=%.2f", avg_fps);

    int baseLine = 0;
    cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

    int y = 0;
    int x = rgb.cols - label_size.width;

    cv::rectangle(rgb, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
                    cv::Scalar(255, 255, 255), -1);

    cv::putText(rgb, text, cv::Point(x, y + label_size.height),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));

    return 0;
}

static YOLOv8* g_yolov8 = 0;
static ncnn::Mutex lock;

class MyNdkCamera : public NdkCameraWindow
{
public:
    virtual void on_image_render(cv::Mat& rgb) const;
};

void MyNdkCamera::on_image_render(cv::Mat& rgb) const
{
    // yolov8
    {
        ncnn::MutexLockGuard g(lock);

        if (g_yolov8)
        {
            std::vector<Object> objects;
            g_yolov8->detect(rgb, objects);

            g_yolov8->draw(rgb, objects);
        }
        else
        {
            draw_unsupported(rgb);
        }
    }

    draw_fps(rgb);
}

static MyNdkCamera* g_camera = 0;

extern "C" {

JNIEXPORT jint JNI_OnLoad(JavaVM* vm, void* reserved)
{
    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "JNI_OnLoad");

    g_camera = new MyNdkCamera;

    ncnn::create_gpu_instance();

    return JNI_VERSION_1_4;
}

JNIEXPORT void JNI_OnUnload(JavaVM* vm, void* reserved)
{
    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "JNI_OnUnload");

    {
        ncnn::MutexLockGuard g(lock);

        delete g_yolov8;
        g_yolov8 = 0;
    }

    ncnn::destroy_gpu_instance();

    delete g_camera;
    g_camera = 0;
}

// public native boolean loadModel(AssetManager mgr, int modelver, int sahi, int cpugpu);
// modelver: 0=v1, 1=v2, 2=v3, 3=v4, 4=v5
// sahi:     0=off (full-frame), 1=on (SAHI 640/0.25/IOS NMS 0.5)
// cpugpu:   0=CPU, 1=GPU (Vulkan), 2=GPU (Turnip)
JNIEXPORT jboolean JNICALL Java_com_tencent_yolov8ncnn_YOLOv8Ncnn_loadModel(JNIEnv* env, jobject thiz, jobject assetManager, jint modelver, jint sahi, jint cpugpu)
{
    if (modelver < 0 || modelver > 4 || sahi < 0 || sahi > 1 || cpugpu < 0 || cpugpu > 2)
    {
        return JNI_FALSE;
    }

    AAssetManager* mgr = AAssetManager_fromJava(env, assetManager);

    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "loadModel modelver=%d sahi=%d cpugpu=%d mgr=%p", (int)modelver, (int)sahi, (int)cpugpu, mgr);

    // citrus v1..v4 all share the 39-class detect head with imgsz=640
    std::string parampath = "v" + std::to_string((int)modelver + 1) + ".ncnn.param";
    std::string modelpath = "v" + std::to_string((int)modelver + 1) + ".ncnn.bin";
    bool use_gpu = (int)cpugpu == 1;
    bool use_turnip = (int)cpugpu == 2;

    // reload
    {
        ncnn::MutexLockGuard g(lock);

        {
            static int old_modelver = -1;
            static int old_sahi = -1;
            static int old_cpugpu = -1;
            if (modelver != old_modelver || sahi != old_sahi || cpugpu != old_cpugpu)
            {
                delete g_yolov8;
                g_yolov8 = 0;
            }
            old_modelver = modelver;
            old_sahi = sahi;
            old_cpugpu = cpugpu;

            ncnn::destroy_gpu_instance();

            if (use_turnip)
            {
                ncnn::create_gpu_instance("libvulkan_freedreno.so");
            }
            else if (use_gpu)
            {
                ncnn::create_gpu_instance();
            }

            if (!g_yolov8)
            {
                g_yolov8 = new YOLOv8_det_coco;
                g_yolov8->load(mgr, parampath.c_str(), modelpath.c_str(), use_gpu || use_turnip);
            }
            g_yolov8->set_det_target_size(640);
            g_yolov8->set_sahi(sahi != 0);
        }
    }

    return JNI_TRUE;
}

// public native boolean openCamera(int facing);
JNIEXPORT jboolean JNICALL Java_com_tencent_yolov8ncnn_YOLOv8Ncnn_openCamera(JNIEnv* env, jobject thiz, jint facing)
{
    if (facing < 0 || facing > 1)
        return JNI_FALSE;

    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "openCamera %d", facing);

    g_camera->open((int)facing);

    return JNI_TRUE;
}

// public native boolean closeCamera();
JNIEXPORT jboolean JNICALL Java_com_tencent_yolov8ncnn_YOLOv8Ncnn_closeCamera(JNIEnv* env, jobject thiz)
{
    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "closeCamera");

    g_camera->close();

    return JNI_TRUE;
}

// public native boolean setResolution(int width, int height);
// Re-opens camera at the requested size. Returns true if camera opened
// successfully and the actual resolution is available (checked after a
// short delay to let AImageReader negotiate with the camera HAL).
JNIEXPORT jboolean JNICALL Java_com_tencent_yolov8ncnn_YOLOv8Ncnn_setResolution(JNIEnv* env, jobject thiz, jint width, jint height)
{
    if (width < 320 || width > 3840 || height < 240 || height > 2160)
        return JNI_FALSE;

    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "setResolution %d x %d", (int)width, (int)height);

    g_camera->close();
    g_camera->set_imagereader_size((int)width, (int)height);
    g_camera->open(g_camera->camera_facing);

    // Query actual image_reader size (Android may adjust to nearest supported)
    int actual_w = g_camera->get_actual_width();
    int actual_h = g_camera->get_actual_height();
    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "actual resolution %d x %d", actual_w, actual_h);

    return (actual_w > 0 && actual_h > 0) ? JNI_TRUE : JNI_FALSE;
}

// public native int[] getResolution() -> returns [width, height] of the
// actual camera preview (may differ from requested if not supported).
JNIEXPORT jintArray JNICALL Java_com_tencent_yolov8ncnn_YOLOv8Ncnn_getResolution(JNIEnv* env, jobject thiz)
{
    int w = g_camera->get_actual_width();
    int h = g_camera->get_actual_height();

    jintArray result = env->NewIntArray(2);
    if (result)
    {
        jint vals[2] = {w, h};
        env->SetIntArrayRegion(result, 0, 2, vals);
    }
    return result;
}

// public native int[] getMaxPreviewResolution() -> [width, height] of the
// highest preview stream size the back camera supports (for 50MP sensors the
// phone still only streams ~4K; 50MP is still-capture only).
JNIEXPORT jintArray JNICALL Java_com_tencent_yolov8ncnn_YOLOv8Ncnn_getMaxPreviewResolution(JNIEnv* env, jobject thiz)
{
    int w = 0, h = 0;
    bool ok = g_camera ? g_camera->get_max_preview_size(w, h) : false;

    jintArray result = env->NewIntArray(2);
    if (result)
    {
        jint vals[2] = {ok ? (jint)w : 0, ok ? (jint)h : 0};
        env->SetIntArrayRegion(result, 0, 2, vals);
    }
    return result;
}

// public native boolean setSahiTileSize(int tileSize);
// 0 = auto (=640, full-frame equivalent), 320 = smaller tiles for small objects.
JNIEXPORT jboolean JNICALL Java_com_tencent_yolov8ncnn_YOLOv8Ncnn_setSahiTileSize(JNIEnv* env, jobject thiz, jint tileSize)
{
    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "setSahiTileSize %d", (int)tileSize);

    if (g_yolov8)
        g_yolov8->set_sahi_tile_size((int)tileSize);

    return JNI_TRUE;
}

// public native boolean setOutputWindow(Surface surface);
JNIEXPORT jboolean JNICALL Java_com_tencent_yolov8ncnn_YOLOv8Ncnn_setOutputWindow(JNIEnv* env, jobject thiz, jobject surface)
{
    ANativeWindow* win = ANativeWindow_fromSurface(env, surface);

    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "setOutputWindow %p", win);

    g_camera->set_window(win);

    return JNI_TRUE;
}

}
