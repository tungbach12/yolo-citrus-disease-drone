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

package com.tencent.yolov8ncnn;

import android.Manifest;
import android.app.Activity;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.PixelFormat;
import android.net.Uri;
import android.os.Bundle;
import android.util.Log;
import android.view.Surface;
import android.view.SurfaceHolder;
import android.view.SurfaceView;
import android.view.View;
import android.view.WindowManager;
import android.widget.AdapterView;
import android.widget.Button;
import android.widget.Spinner;
import android.widget.Toast;

import android.support.v4.app.ActivityCompat;
import android.support.v4.content.ContextCompat;

import java.io.InputStream;

public class MainActivity extends Activity implements SurfaceHolder.Callback
{
    public static final int REQUEST_CAMERA = 100;
    public static final int REQUEST_PICK_IMAGE = 101;

    private YOLOv8Ncnn yolov8ncnn = new YOLOv8Ncnn();
    private int facing = 0;

    // modelver: 0=v1, 1=v2, 2=v3, 3=v4, 4=v5 (P2 head)
    private int current_modelver = 4;
    // sahi: 0=off (full-frame), 1=on (SAHI 640/0.25/IOS NMS 0.5)
    private int current_sahi = 0;
    private int current_cpugpu = 0;
    // resolution: 0="Max" (query camera), 1=640x480, 2=1280x720, 3=1920x1080, 4=2560x1440
    private int current_resolution = 0;

    private static final int[] RES_W = {0, 640, 1280, 1920, 2560};
    private static final int[] RES_H = {0, 480, 720, 1080, 1440};

    // sahi tile: 0=auto(640), 1=320
    private int current_sahi_tile = 0;

    // whether the default "Max" resolution was already applied on first resume
    private boolean firstMaxApplied = false;

    // when true, a still image is shown on the surface instead of the live
    // camera preview. The camera is closed so it doesn't overwrite the surface.
    private boolean imageMode = false;

    private static final int[] SAHI_TILES = {0, 320};

    private SurfaceView cameraView;

    /** Called when the activity is first created. */
    @Override
    public void onCreate(Bundle savedInstanceState)
    {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.main);

        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        cameraView = (SurfaceView) findViewById(R.id.cameraview);

        cameraView.getHolder().setFormat(PixelFormat.RGBA_8888);
        cameraView.getHolder().addCallback(this);

        Button buttonSwitchCamera = (Button) findViewById(R.id.buttonSwitchCamera);
        buttonSwitchCamera.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View arg0) {

                int new_facing = 1 - facing;

                imageMode = false;
                yolov8ncnn.closeCamera();

                yolov8ncnn.openCamera(new_facing);

                facing = new_facing;
            }
        });

        Button buttonUpload = (Button) findViewById(R.id.buttonUpload);
        buttonUpload.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View arg0) {
                // pause the camera so it doesn't draw over the picked photo
                imageMode = true;
                yolov8ncnn.closeCamera();

                Intent intent = new Intent(Intent.ACTION_PICK);
                intent.setType("image/*");
                startActivityForResult(intent, REQUEST_PICK_IMAGE);
            }
        });

        Spinner spinnerModel = (Spinner) findViewById(R.id.spinnerModel);
        spinnerModel.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {
            @Override
            public void onItemSelected(AdapterView<?> arg0, View arg1, int position, long id)
            {
                if (position != current_modelver)
                {
                    current_modelver = position;
                    reload();
                }
            }

            @Override
            public void onNothingSelected(AdapterView<?> arg0)
            {
            }
        });

        Spinner spinnerSAHI = (Spinner) findViewById(R.id.spinnerSAHI);
        spinnerSAHI.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {
            @Override
            public void onItemSelected(AdapterView<?> arg0, View arg1, int position, long id)
            {
                if (position != current_sahi)
                {
                    current_sahi = position;
                    reload();
                }
            }

            @Override
            public void onNothingSelected(AdapterView<?> arg0)
            {
            }
        });

        Spinner spinnerCPUGPU = (Spinner) findViewById(R.id.spinnerCPUGPU);
        spinnerCPUGPU.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {
            @Override
            public void onItemSelected(AdapterView<?> arg0, View arg1, int position, long id)
            {
                if (position != current_cpugpu)
                {
                    current_cpugpu = position;
                    reload();
                }
            }

            @Override
            public void onNothingSelected(AdapterView<?> arg0)
            {
            }
        });

        Spinner spinnerSahiTile = (Spinner) findViewById(R.id.spinnerSahiTile);
        spinnerSahiTile.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {
            @Override
            public void onItemSelected(AdapterView<?> arg0, View arg1, int position, long id)
            {
                if (position != current_sahi_tile)
                {
                    current_sahi_tile = position;
                    yolov8ncnn.setSahiTileSize(SAHI_TILES[position]);
                    reload();
                }
            }

            @Override
            public void onNothingSelected(AdapterView<?> arg0)
            {
            }
        });

        Spinner spinnerResolution = (Spinner) findViewById(R.id.spinnerResolution);
        spinnerResolution.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {
            @Override
            public void onItemSelected(AdapterView<?> arg0, View arg1, int position, long id)
            {
                if (position != current_resolution)
                {
                    current_resolution = position;
                    if (position == 0)
                    {
                        // "Max" — query the back camera's highest preview stream size
                        int[] max = yolov8ncnn.getMaxPreviewResolution();
                        if (max != null && max.length == 2 && max[0] > 0 && max[1] > 0)
                        {
                            boolean ok = yolov8ncnn.setResolution(max[0], max[1]);
                            int[] actual = yolov8ncnn.getResolution();
                            if (actual != null && actual.length == 2 && actual[0] > 0 && actual[1] > 0)
                            {
                                String msg = ok
                                    ? "Max: " + actual[0] + "x" + actual[1]
                                    : "Max " + actual[0] + "x" + actual[1];
                                Toast.makeText(MainActivity.this, msg, Toast.LENGTH_LONG).show();
                            }
                        }
                        else
                        {
                            Toast.makeText(MainActivity.this, "Không query được max, dùng 1080p",
                                Toast.LENGTH_LONG).show();
                            yolov8ncnn.setResolution(1920, 1080);
                        }
                    }
                    else
                    {
                        yolov8ncnn.setResolution(RES_W[position], RES_H[position]);

                        // query actual camera size (may differ from requested)
                        int[] actual = yolov8ncnn.getResolution();
                        if (actual != null && actual.length == 2 && actual[0] > 0 && actual[1] > 0)
                        {
                            boolean ok = actual[0] == RES_W[position] && actual[1] == RES_H[position];
                            String msg = ok
                                ? "Đã chuyển sang " + actual[0] + "x" + actual[1]
                                : "Lỗi: yêu cầu " + RES_W[position] + "x" + RES_H[position] + " → camera " + actual[0] + "x" + actual[1];
                            Toast.makeText(MainActivity.this, msg,
                                ok ? Toast.LENGTH_SHORT : Toast.LENGTH_LONG).show();
                        }
                    }
                }
            }

            @Override
            public void onNothingSelected(AdapterView<?> arg0)
            {
            }
        });

        reload();
    }

    private void reload()
    {
        boolean ret_init = yolov8ncnn.loadModel(getAssets(), current_modelver, current_sahi, current_cpugpu);
        if (!ret_init)
        {
            Log.e("MainActivity", "yolov8ncnn loadModel failed");
        }
    }

    @Override
    public void surfaceChanged(SurfaceHolder holder, int format, int width, int height)
    {
        yolov8ncnn.setOutputWindow(holder.getSurface());
    }

    @Override
    public void surfaceCreated(SurfaceHolder holder)
    {
    }

    @Override
    public void surfaceDestroyed(SurfaceHolder holder)
    {
    }

    @Override
    public void onResume()
    {
        super.onResume();

        if (ContextCompat.checkSelfPermission(getApplicationContext(), Manifest.permission.CAMERA) == PackageManager.PERMISSION_DENIED)
        {
            ActivityCompat.requestPermissions(this, new String[] {Manifest.permission.CAMERA}, REQUEST_CAMERA);
        }

        // showing a picked photo: keep the camera off so it doesn't overwrite it
        if (imageMode)
            return;

        if (!firstMaxApplied)
        {
            // first launch: default resolution "Max" = highest the camera will
            // stream. setResolution() also (re)opens the camera, so skip the
            // openCamera() below to avoid opening the device twice.
            firstMaxApplied = true;
            int[] max = yolov8ncnn.getMaxPreviewResolution();
            if (max != null && max.length == 2 && max[0] > 0 && max[1] > 0)
            {
                yolov8ncnn.setResolution(max[0], max[1]);
                return;
            }
        }

        yolov8ncnn.openCamera(facing);
    }

    @Override
    public void onPause()
    {
        super.onPause();

        yolov8ncnn.closeCamera();
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data)
    {
        super.onActivityResult(requestCode, resultCode, data);

        if (requestCode == REQUEST_PICK_IMAGE)
        {
            if (resultCode == RESULT_OK && data != null && data.getData() != null)
            {
                Uri uri = data.getData();
                try
                {
                    InputStream is = getContentResolver().openInputStream(uri);
                    BitmapFactory.Options opts = new BitmapFactory.Options();
                    opts.inSampleSize = 1;
                    Bitmap bmp = BitmapFactory.decodeStream(is, null, opts);
                    if (is != null) is.close();

                    if (bmp != null)
                    {
                        // ensure RGBA_8888 for the JNI bridge
                        if (bmp.getConfig() != Bitmap.Config.ARGB_8888)
                            bmp = bmp.copy(Bitmap.Config.ARGB_8888, false);

                        yolov8ncnn.detectBitmap(bmp);
                        return;
                    }
                }
                catch (Exception e)
                {
                    Log.e("MainActivity", "pick image failed", e);
                }
                Toast.makeText(this, "Không đọc được ảnh", Toast.LENGTH_LONG).show();
                // fall back to camera
                imageMode = false;
                onResumeCamera();
            }
            else
            {
                // cancelled -> back to live camera
                imageMode = false;
                onResumeCamera();
            }
        }
    }

    // reopen the live camera (used after leaving image mode)
    private void onResumeCamera()
    {
        if (ContextCompat.checkSelfPermission(getApplicationContext(), Manifest.permission.CAMERA) == PackageManager.PERMISSION_DENIED)
            return;
        yolov8ncnn.openCamera(facing);
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, String[] permissions, int[] grantResults)
    {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
    }
}
