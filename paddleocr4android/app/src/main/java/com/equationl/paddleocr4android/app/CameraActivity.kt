package com.equationl.paddleocr4android.app

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.*
import android.hardware.camera2.*
import android.media.Image
import android.media.ImageReader
import android.os.Bundle
import android.os.Handler
import android.os.HandlerThread
import android.util.Log
import android.util.Size
import android.view.Surface
import android.view.TextureView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.equationl.paddleocr4android.CpuPowerMode
import com.equationl.paddleocr4android.OCR
import com.equationl.paddleocr4android.OcrConfig
import com.equationl.paddleocr4android.bean.OcrResult
import java.io.ByteArrayOutputStream
import java.util.concurrent.Executors

class CameraActivity : AppCompatActivity() {

    // UI
    private lateinit var viewFinder: TextureView
    private lateinit var overlayView: OverlayView

    // Camera2
    private lateinit var cameraManager: CameraManager
    private var cameraDevice: CameraDevice? = null
    private var captureSession: CameraCaptureSession? = null
    private lateinit var imageReader: ImageReader
    private lateinit var previewSize: Size
    private lateinit var cameraId: String
    private lateinit var backgroundThread: HandlerThread
    private lateinit var backgroundHandler: Handler

    // OCR
    private lateinit var ocr: OCR
    @Volatile private var isOcrReady = false
    
    // Throttling
    @Volatile private var isProcessing = false
    private var lastProcessedTime = 0L
    private val PROCESS_INTERVAL_MS = 300L
    
    // FPS tracking
    private var lastFrameTime = System.currentTimeMillis()
    private var fps = 0.0

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_camera)

        viewFinder = findViewById(R.id.viewFinder)
        overlayView = findViewById(R.id.overlayView)
        cameraManager = getSystemService(Context.CAMERA_SERVICE) as CameraManager

        initOCR()
    }

    private fun initOCR() {
        ocr = OCR(this)
        val config = OcrConfig()
        config.modelPath = "models/ch_PP-OCRv4"
        config.clsModelFilename = "cls.nb"
        config.detModelFilename = "det.nb"
        config.recModelFilename = "rec.nb"
        config.isRunDet = true
        config.isRunCls = true
        config.isRunRec = true
        config.cpuPowerMode = CpuPowerMode.LITE_POWER_FULL
        config.isDrwwTextPositionBox = false

        ocr.initModel(config, object : com.equationl.paddleocr4android.callback.OcrInitCallback {
            override fun onSuccess() {
                Log.d(TAG, "OCR Model Initialized")
                isOcrReady = true
                runOnUiThread {
                    Toast.makeText(this@CameraActivity, "OCR Ready", Toast.LENGTH_SHORT).show()
                    
                    // Start camera after OCR is ready
                    if (allPermissionsGranted()) {
                        startBackgroundThread()
                        setupTextureListener()
                    } else {
                        ActivityCompat.requestPermissions(
                            this@CameraActivity, REQUIRED_PERMISSIONS, REQUEST_CODE_PERMISSIONS
                        )
                    }
                }
            }

            override fun onFail(e: Throwable) {
                Log.e(TAG, "OCR Model Init Failed", e)
                runOnUiThread {
                    Toast.makeText(this@CameraActivity, "OCR Init Failed", Toast.LENGTH_SHORT).show()
                    finish()
                }
            }
        })
    }

    private fun startBackgroundThread() {
        backgroundThread = HandlerThread("CameraBackground")
        backgroundThread.start()
        backgroundHandler = Handler(backgroundThread.looper)
    }

    private fun stopBackgroundThread() {
        try {
            backgroundThread.quitSafely()
            backgroundThread.join()
        } catch (e: InterruptedException) {
            Log.e(TAG, "Error stopping background thread", e)
        }
    }

    private fun setupTextureListener() {
        if (viewFinder.isAvailable) {
            startCamera()
        } else {
            viewFinder.surfaceTextureListener = object : TextureView.SurfaceTextureListener {
                override fun onSurfaceTextureAvailable(surface: SurfaceTexture, width: Int, height: Int) {
                    startCamera()
                }
                override fun onSurfaceTextureSizeChanged(surface: SurfaceTexture, width: Int, height: Int) {}
                override fun onSurfaceTextureDestroyed(surface: SurfaceTexture) = true
                override fun onSurfaceTextureUpdated(surface: SurfaceTexture) {}
            }
        }
    }

    private fun startCamera() {
        try {
            chooseCamera()
            
            if (ActivityCompat.checkSelfPermission(this, Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
                return
            }

            cameraManager.openCamera(cameraId, object : CameraDevice.StateCallback() {
                override fun onOpened(device: CameraDevice) {
                    Log.d(TAG, "Camera opened successfully")
                    cameraDevice = device
                    createCameraSession()
                }
                override fun onDisconnected(device: CameraDevice) {
                    Log.w(TAG, "Camera disconnected")
                    device.close()
                    cameraDevice = null
                }
                override fun onError(device: CameraDevice, error: Int) {
                    Log.e(TAG, "Camera error: $error")
                    device.close()
                    cameraDevice = null
                }
            }, backgroundHandler)
        } catch (e: Exception) {
            Log.e(TAG, "Error starting camera", e)
        }
    }

    private fun chooseCamera() {
        for (id in cameraManager.cameraIdList) {
            val characteristics = cameraManager.getCameraCharacteristics(id)
            val facing = characteristics.get(CameraCharacteristics.LENS_FACING)
            
            if (facing == CameraCharacteristics.LENS_FACING_BACK) {
                cameraId = id
                previewSize = Size(1280, 720)
                
                imageReader = ImageReader.newInstance(1280, 720, ImageFormat.YUV_420_888, 2)
                imageReader.setOnImageAvailableListener({ reader ->
                    val image = reader.acquireLatestImage() ?: return@setOnImageAvailableListener
                    val currentTime = System.currentTimeMillis()

                    if (!isOcrReady || isProcessing || (currentTime - lastProcessedTime) < PROCESS_INTERVAL_MS) {
                        image.close()
                        return@setOnImageAvailableListener
                    }

                    isProcessing = true
                    lastProcessedTime = currentTime

                    // --- All processing happens on this single background thread ---
                    processImageSingleThread(image)
                }, backgroundHandler)  // Only this backgroundHandler is used
                return
            }
        }
    }

    private fun createCameraSession() {
        try {
            val texture = viewFinder.surfaceTexture ?: return
            texture.setDefaultBufferSize(previewSize.width, previewSize.height)
            
            val previewSurface = Surface(texture)
            val imageSurface = imageReader.surface
            
            val captureRequestBuilder = cameraDevice!!.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW)
            captureRequestBuilder.addTarget(previewSurface)
            captureRequestBuilder.addTarget(imageSurface)
            
            cameraDevice!!.createCaptureSession(
                listOf(previewSurface, imageSurface),
                object : CameraCaptureSession.StateCallback() {
                    override fun onConfigured(session: CameraCaptureSession) {
                        Log.d(TAG, "Camera session configured")
                        captureSession = session
                        captureSession?.setRepeatingRequest(captureRequestBuilder.build(), null, backgroundHandler)
                    }
                    override fun onConfigureFailed(session: CameraCaptureSession) {
                        Log.e(TAG, "Camera session configuration failed")
                    }
                },
                backgroundHandler
            )
        } catch (e: Exception) {
            Log.e(TAG, "Error creating camera session", e)
        }
    }

    private fun processImageSingleThread(image: Image) {
        var bitmap: Bitmap? = null
        try {
            val totalStartTime = System.currentTimeMillis()

            // 1️⃣ Convert YUV → RGB and scale down
            bitmap = yuvToRgbScaledBitmap(image, 640, 480) ?: return

            val predictor = ocr.getPredictor()
            predictor.setInputImage(bitmap)

            // 2️⃣ Run OCR (detect + cls + rec)
            val startTime = System.currentTimeMillis()
            predictor.runModel(true, true, true)
            val totalTime = System.currentTimeMillis() - startTime

            // 3️⃣ Prepare result
            val ocrResult = OcrResult(
                predictor.outputResult(),
                predictor.inferenceTime(),
                predictor.outputImage(),
                predictor.outputRawResult()
            )

            // 4️⃣ FPS calculation
            fps = 1000.0 / (System.currentTimeMillis() - lastFrameTime)
            lastFrameTime = System.currentTimeMillis()
            Log.d(TAG, "OCR Total Time: ${totalTime}ms | FPS: ${"%.2f".format(fps)}")

            // 5️⃣ Update overlay (UI must be updated on main thread)
            runOnUiThread {
                overlayView.setResult(
                    ocrResult,
                    bitmap.width,
                    bitmap.height,
                    0,
                    viewFinder.width,
                    viewFinder.height
                )
            }

        } catch (e: Exception) {
            Log.e(TAG, "Error processing image", e)
        } finally {
            bitmap?.recycle()
            image.close()
            isProcessing = false
        }
    }
    
    private fun yuvToRgbScaledBitmap(image: Image, maxWidth: Int, maxHeight: Int): Bitmap? {
        return try {
            val yBuffer = image.planes[0].buffer
            val uBuffer = image.planes[1].buffer
            val vBuffer = image.planes[2].buffer

            val ySize = yBuffer.remaining()
            val uSize = uBuffer.remaining()
            val vSize = vBuffer.remaining()

            val nv21 = ByteArray(ySize + uSize + vSize)
            yBuffer.get(nv21, 0, ySize)
            vBuffer.get(nv21, ySize, vSize)
            uBuffer.get(nv21, ySize + vSize, uSize)

            val yuvImage = YuvImage(nv21, ImageFormat.NV21, image.width, image.height, null)
            val out = ByteArrayOutputStream()
            yuvImage.compressToJpeg(Rect(0, 0, image.width, image.height), 80, out)
            val imageBytes = out.toByteArray()
            out.close()

            var bitmap = BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.size)

            // Rotate to portrait
            val matrix = Matrix()
            matrix.postRotate(90f)
            bitmap = Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)

            // Scale down if needed
            val scale = Math.min(maxWidth.toFloat() / bitmap.width, maxHeight.toFloat() / bitmap.height)
            if (scale < 1f) {
                val scaledWidth = (bitmap.width * scale).toInt()
                val scaledHeight = (bitmap.height * scale).toInt()
                val scaledBitmap = Bitmap.createScaledBitmap(bitmap, scaledWidth, scaledHeight, true)
                bitmap.recycle()
                scaledBitmap
            } else {
                bitmap
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error converting YUV to bitmap", e)
            null
        }
    }

    private fun allPermissionsGranted() = REQUIRED_PERMISSIONS.all {
        ContextCompat.checkSelfPermission(baseContext, it) == PackageManager.PERMISSION_GRANTED
    }

    override fun onResume() {
        super.onResume()
        if (isOcrReady && allPermissionsGranted() && !::backgroundThread.isInitialized) {
            startBackgroundThread()
            if (viewFinder.isAvailable) {
                startCamera()
            } else {
                setupTextureListener()
            }
        }
    }

    override fun onPause() {
        closeCamera()
        stopBackgroundThread()
        super.onPause()
    }

    override fun onDestroy() {
        super.onDestroy()
        ocr.releaseModel()
    }

    private fun closeCamera() {
        try {
            captureSession?.close()
            captureSession = null
            cameraDevice?.close()
            cameraDevice = null
            imageReader.close()
        } catch (e: Exception) {
            Log.e(TAG, "Error closing camera", e)
        }
    }

    override fun onRequestPermissionsResult(
        requestCode: Int, permissions: Array<String>, grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == REQUEST_CODE_PERMISSIONS) {
            if (allPermissionsGranted()) {
                startBackgroundThread()
                setupTextureListener()
            } else {
                Toast.makeText(this, "Camera permission required", Toast.LENGTH_SHORT).show()
                finish()
            }
        }
    }

    companion object {
        private const val TAG = "CameraActivity"
        private const val REQUEST_CODE_PERMISSIONS = 10
        private val REQUIRED_PERMISSIONS = arrayOf(Manifest.permission.CAMERA)
    }
}