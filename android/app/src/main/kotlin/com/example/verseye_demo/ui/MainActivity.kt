package com.example.verseye_demo.ui

import android.annotation.SuppressLint
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.*
import android.hardware.camera2.CameraCaptureSession
import android.hardware.camera2.CameraDevice
import android.hardware.camera2.CameraManager
import androidx.appcompat.app.AppCompatActivity
import androidx.appcompat.app.AlertDialog
import android.os.Bundle
import android.os.Handler
import android.os.HandlerThread
import android.util.Log
import android.view.Surface
import android.view.SurfaceView
import android.view.TextureView
import android.view.View
import android.widget.TextView
import android.widget.LinearLayout
import android.widget.ImageButton
import android.widget.Button
import android.widget.ImageView
import android.content.Intent
import android.net.Uri
import android.provider.MediaStore
import androidx.activity.result.contract.ActivityResultContracts
import androidx.core.content.ContextCompat
import android.view.Gravity
import android.os.Build
import android.view.WindowInsets
import android.view.WindowInsetsController
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.DataType
import org.tensorflow.lite.gpu.CompatibilityList
import org.tensorflow.lite.gpu.GpuDelegate
import org.tensorflow.lite.Interpreter
import java.io.*
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import kotlinx.coroutines.*
import java.net.URL
import org.json.JSONArray
import java.nio.ByteBuffer
import java.nio.ByteOrder
import com.example.verseye_demo.models.ModelInfo
//import com.example.verseye_demo.helpers.LicensePlateDetectionHelper
import com.example.verseye_demo.helpers.PoseEstimationHelper
import com.example.verseye_demo.helpers.SegmentationHelper
//import com.example.verseye_demo.helpers.BlinkDrowseDetectionHelper
//import com.example.verseye_demo.helpers.DentDetectionHelper
import com.example.verseye_demo.helpers.FaceDetectionHelper
//import com.example.verseye_demo.helpers.FireSmokeDetectionHelper
//import com.example.verseye_demo.helpers.GroceryDetectionHelper
//import com.example.verseye_demo.helpers.HelmetDetectionHelper
import com.example.verseye_demo.R
import org.json.JSONObject

class MainActivity : AppCompatActivity() {

    lateinit var labels:List<String>
    var colors = listOf<Int>(
        Color.BLUE, Color.GREEN, Color.RED, Color.CYAN, Color.GRAY, Color.BLACK,
        Color.DKGRAY, Color.MAGENTA, Color.YELLOW, Color.RED)
    val paint = Paint()
    lateinit var imageProcessor: ImageProcessor
    lateinit var bitmap:Bitmap
    lateinit var overlayView: SurfaceView
    lateinit var cameraDevice: CameraDevice
    lateinit var handler: Handler
    lateinit var cameraManager: CameraManager
    lateinit var textureView: TextureView
    private var gpuDelegate: GpuDelegate? = null
    private var loadingDialog: AlertDialog? = null
    lateinit var settingsButton: ImageButton
    lateinit var fpsTextView: TextView
    private lateinit var cameraButton: Button
    private lateinit var uploadButton: Button
    private lateinit var staticImageView: ImageView
    
    // Download progress UI
    private var downloadProgressBar: android.widget.ProgressBar? = null
    private var downloadProgressText: TextView? = null
    private var isInCameraMode = true
    private var uploadedImageUri: Uri? = null
    
    private val pickMediaLauncher = registerForActivityResult(ActivityResultContracts.GetContent()) { uri: Uri? ->
        uri?.let {
            uploadedImageUri = it
            handleUploadedMedia(it)
        }
    }
    
    // ModelInfo is now in models package
    private val availableModels = mutableListOf<ModelInfo>()
    private var currentModelIndex: Int? = null
    private val API_URL = "http://ai-srv.qbscocloud.net:8006/models"
    
    // Predefined list of available models
    
    
    lateinit var bottomDashboard: LinearLayout
    lateinit var dashboardEmoji1: TextView
    lateinit var dashboardCount1: TextView
    lateinit var dashboardEmoji2: TextView
    lateinit var dashboardCount2: TextView

    private val labelToEmoji: Map<String, String> = mapOf(
        "person" to "🧍", "bicycle" to "🚲", "car" to "🚗", "motorcycle" to "🏍️",
        "airplane" to "✈️", "bus" to "🚌", "train" to "🚆", "truck" to "🚚",
        "boat" to "🚤", "traffic light" to "🚦", "fire hydrant" to "🚒",
        "stop sign" to "🛑", "parking meter" to "🅿️", "bench" to "🪑",
        "bird" to "🐦", "cat" to "🐱", "dog" to "🐶", "horse" to "🐴",
        "sheep" to "🐑", "cow" to "🐄", "elephant" to "🐘", "bear" to "🐻",
        "zebra" to "🦓", "giraffe" to "🦒", "backpack" to "🎒", "umbrella" to "☂️",
        "handbag" to "👜", "tie" to "👔", "suitcase" to "🧳", "frisbee" to "🥏",
        "skis" to "🎿", "snowboard" to "🏂", "sports ball" to "⚽", "kite" to "🪁",
        "baseball bat" to "⚾", "baseball glove" to "🥎", "skateboard" to "🛹",
        "surfboard" to "🏄", "tennis racket" to "🎾", "bottle" to "🍼",
        "wine glass" to "🍷", "cup" to "☕", "fork" to "🍴", "knife" to "🔪",
        "spoon" to "🥄", "bowl" to "🥣", "banana" to "🍌", "apple" to "🍎",
        "sandwich" to "🥪", "orange" to "🍊", "broccoli" to "🥦", "carrot" to "🥕",
        "hot dog" to "🌭", "pizza" to "🍕", "donut" to "🍩", "cake" to "🎂",
        "chair" to "🪑", "couch" to "🛋️", "potted plant" to "🪴", "bed" to "🛏️",
        "dining table" to "🍽️", "toilet" to "🚽", "tv" to "📺", "laptop" to "💻",
        "mouse" to "🖱️", "remote" to "🕹️", "keyboard" to "⌨️", "cell phone" to "📱",
        "microwave" to "📡", "oven" to "🔥", "toaster" to "🍞", "sink" to "🚰",
        "refrigerator" to "🧊", "book" to "📚", "clock" to "🕒", "vase" to "🏺",
        "scissors" to "✂️", "teddy bear" to "🧸", "hair drier" to "💨", "toothbrush" to "🪥"
    )
    private val counts: MutableMap<String, Int> = mutableMapOf()

    private var frameCounter = 0
    private var lastFpsTimestamp = System.currentTimeMillis()
    private var currentFps = 0.0
    
    private var isUsingYolox: Boolean
        get() = currentModelIndex?.let { availableModels.getOrNull(it)?.type == "yolox" } ?: false
        set(value) { }
    
    private var isProcessing = false
    private var lastProcessTime = 0L
    private val MIN_PROCESS_INTERVAL = 50L
    
    private var INPUT_SIZE = 300
    private var INPUT_WIDTH = 300
    private var INPUT_HEIGHT = 300
    private var OUTPUT_SHAPE: IntArray? = null
    private val CONFIDENCE_THRESHOLD = 0.5f
    private val NMS_THRESHOLD = 0.45f
    
    private val modelInterpreters = mutableMapOf<String, Interpreter>()
    private var currentInterpreter: Interpreter? = null
    
    // Detection helpers for specialized models
    private var poseHelper: PoseEstimationHelper? = null
    private var segmentationHelper: SegmentationHelper? = null
    private var faceHelper: FaceDetectionHelper? = null
    
    // Store annotated bitmap from pose/segmentation helpers
    private var annotatedBitmap: Bitmap? = null
    
    // Camera state management
    private var isCameraOpen = false
    private var captureSession: CameraCaptureSession? = null

    // Track last processed frame dimensions for correct overlay scaling
    private var lastFrameWidth: Int = 0
    private var lastFrameHeight: Int = 0

    // Swipe-back UI elements
    private var isSwipingBack = false
    private var swipeProgress = 0f
    private var dragStartX: Float? = null
    private lateinit var edgeSwipeArea: View
    private lateinit var edgeChevron: ImageView
    
    data class Detection(
        val x1: Float, val y1: Float, val x2: Float, val y2: Float,
        val confidence: Float, val classId: Int, val label: String
    )
    
    data class InferenceResult(
        val detections: List<Detection>,
        val annotatedBitmap: Bitmap? = null
    )

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_object_detection)
        hideSystemUI()
        get_permission()

        paint.style = Paint.Style.STROKE
        paint.strokeWidth = 4f
        paint.isAntiAlias = false
        
        val handlerThread = HandlerThread("videoThread")
        handlerThread.start()
        handler = Handler(handlerThread.looper)

        overlayView = findViewById(R.id.overlayView)
        overlayView.setZOrderOnTop(true)
        overlayView.holder.setFormat(PixelFormat.TRANSPARENT)

        settingsButton = findViewById(R.id.settingsButton)
        settingsButton.setOnClickListener { showModelSelectionDialog() }

        fpsTextView = findViewById(R.id.fpsTextView)

        // Back button clickfindViewById<ImageButton>(R.id.backButton)?.setOnClickListener { finish() }

        // Edge swipe and chevron
        edgeSwipeArea = findViewById(R.id.edgeSwipeArea)
        edgeChevron = findViewById(R.id.edgeChevron)
        setupEdgeSwipe()
        
        bottomDashboard = findViewById(R.id.bottomDashboard)
        dashboardEmoji1 = findViewById(R.id.dashboardEmoji1)
        dashboardCount1 = findViewById(R.id.dashboardCount1)
        dashboardEmoji2 = findViewById(R.id.dashboardEmoji2)
        dashboardCount2 = findViewById(R.id.dashboardCount2)
        bottomDashboard.setOnClickListener { showExpandedDashboard() }
        
        textureView = findViewById(R.id.textureView)
        textureView.surfaceTextureListener = object:TextureView.SurfaceTextureListener{
            override fun onSurfaceTextureAvailable(p0: SurfaceTexture, p1: Int, p2: Int) {
                Log.d("Camera", "Surface texture available")
                // Open camera when surface is ready and we have permission
                if (!isCameraOpen && ContextCompat.checkSelfPermission(this@MainActivity, android.Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
                    open_camera()
                }
            }
            override fun onSurfaceTextureSizeChanged(p0: SurfaceTexture, p1: Int, p2: Int) {
                Log.d("Camera", "Surface size changed: $p1 x $p2")
            }
            override fun onSurfaceTextureDestroyed(p0: SurfaceTexture): Boolean {
                Log.d("Camera", "Surface texture destroyed")
                close_camera()
                return true
            }
            override fun onSurfaceTextureUpdated(p0: SurfaceTexture) {
                // Only process if model is loaded and interpreter is ready
                if (currentModelIndex == null || currentInterpreter == null) return
                
                val currentTime = System.currentTimeMillis()
                if (currentTime - lastProcessTime < MIN_PROCESS_INTERVAL) return
                if (isProcessing) return
                lastProcessTime = currentTime
                isProcessing = true
                try {
                    val frameBitmap = textureView.bitmap ?: run {
                        isProcessing = false
                        return
                    }
                    processFrameSynchronous(frameBitmap)
                } catch (e: Exception) {
                    Log.e("FrameCapture", "Error: ${e.message}")
                    isProcessing = false
                }
            }
        }

        cameraManager = getSystemService(Context.CAMERA_SERVICE) as CameraManager

        // Initialize camera and upload buttons
        cameraButton = findViewById(R.id.cameraButton)
        uploadButton = findViewById(R.id.uploadButton)
        staticImageView = findViewById(R.id.staticImageView)
        
        cameraButton.setOnClickListener {
            switchToCameraMode()
        }
        
        uploadButton.setOnClickListener {
            showUploadOptions()
        }

        // Fetch model list from API (but don't download)
        showLoadingDialog("Fetching available models...")
        GlobalScope.launch(Dispatchers.IO) {
            try {
                fetchModelList()
                runOnUiThread {
                    dismissLoadingDialog("Models list loaded!")
                }
            } catch (e: Exception) {
                Log.e("ModelAPI", "Error: ${e.message}")
                runOnUiThread {
                    dismissLoadingDialog("Error: ${e.message}")
                    showErrorDialog("Failed to fetch model list. Check your connection.")
                }
            }
        }
    }

    private fun dp(value: Float): Float = value * resources.displayMetrics.density

    private fun setupEdgeSwipe() {
        edgeSwipeArea.setOnTouchListener { _, event ->
            when (event.actionMasked) {
                android.view.MotionEvent.ACTION_DOWN -> {
                    dragStartX = event.rawX
                    isSwipingBack = (dragStartX ?: 0f) < dp(32f)
                    if (isSwipingBack) {
                        swipeProgress = 0f
                        edgeChevron.visibility = View.VISIBLE
                        edgeChevron.alpha = 0f
                        edgeChevron.translationX = dp(16f)
                    }
                    isSwipingBack
                }
                android.view.MotionEvent.ACTION_MOVE -> {
                    if (!isSwipingBack || dragStartX == null) return@setOnTouchListener false
                    val dx = event.rawX - (dragStartX ?: 0f)
                    val progress = (dx / dp(140f)).coerceIn(0f, 1f)
                    swipeProgress = progress
                    edgeChevron.alpha = progress
                    edgeChevron.translationX = dp(16f) + progress * dp(56f)
                    true
                }
                android.view.MotionEvent.ACTION_UP, android.view.MotionEvent.ACTION_CANCEL -> {
                    if (isSwipingBack) {
                        val dx = (event.rawX - (dragStartX ?: 0f))
                        val shouldFinish = dx > dp(120f) || swipeProgress > 0.5f
                        if (shouldFinish) finish()
                    }
                    isSwipingBack = false
                    swipeProgress = 0f
                    dragStartX = null
                    edgeChevron.visibility = View.GONE
                    true
                }
                else -> false
            }
        }
    }

    private fun hideSystemUI() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
            window.setDecorFitsSystemWindows(false)
            val controller = window.insetsController
            controller?.hide(WindowInsets.Type.statusBars() or WindowInsets.Type.navigationBars())
            controller?.systemBarsBehavior = WindowInsetsController.BEHAVIOR_SHOW_TRANSIENT_BARS_BY_SWIPE
        } else {
            @Suppress("DEPRECATION")
            window.decorView.systemUiVisibility = (
                View.SYSTEM_UI_FLAG_IMMERSIVE_STICKY or
                View.SYSTEM_UI_FLAG_FULLSCREEN or
                View.SYSTEM_UI_FLAG_HIDE_NAVIGATION or
                View.SYSTEM_UI_FLAG_LAYOUT_FULLSCREEN or
                View.SYSTEM_UI_FLAG_LAYOUT_HIDE_NAVIGATION or
                View.SYSTEM_UI_FLAG_LAYOUT_STABLE
            )
        }
    }

    override fun onWindowFocusChanged(hasFocus: Boolean) {
        super.onWindowFocusChanged(hasFocus)
        if (hasFocus) hideSystemUI()
    }

    private suspend fun fetchModelList() {
    withContext(Dispatchers.IO) {
        availableModels.clear()
        
        try {
            Log.d("ModelAPI", "Fetching models from: $API_URL")
            
            // Fetch JSON response from API
            val jsonResponse = URL(API_URL).readText()
            Log.d("ModelAPI", "API Response: $jsonResponse")
            
            // Parse JSON response
            val jsonObject = JSONObject(jsonResponse)
            val dataArray = jsonObject.getJSONArray("Data")
            
            Log.d("ModelAPI", "Found ${dataArray.length()} models in API response")
            
            // Process each model from the API
            for (i in 0 until dataArray.length()) {
                val modelObj = dataArray.getJSONObject(i)
                
                val modelName = modelObj.getString("model_name")
                val modelUrl = modelObj.getString("model_url")
                val labelsUrl = modelObj.getString("model_labels")
                
                // Extract filename from URL
                val modelFileName = modelUrl.substringAfterLast("/")
                val labelsFileName = labelsUrl.substringAfterLast("/")
                
                // Check if model is already downloaded
                val modelFile = File(filesDir, modelFileName)
                val labelsFile = File(filesDir, labelsFileName)
                val isDownloaded = modelFile.exists() && labelsFile.exists()
                
                // Determine model type from model name
                // Only 3 special types need specific helpers, rest use faceHelper
                val modelType = when {
                    modelName.contains("pose", ignoreCase = true) -> "yolo 11 pose"
                    modelName.contains("seg", ignoreCase = true) -> "yolo 11 segmentation"
                    modelName.contains("yolox", ignoreCase = true) -> "yolox"
                    // ALL other models (including new ones added to API) will use faceHelper
                    else -> "standard"
                }
                
                // Create display name (capitalize and format)
                val displayName = modelName.split("_", "-")
                    .joinToString(" ") { word -> 
                        word.replaceFirstChar { 
                            if (it.isLowerCase()) it.titlecase() else it.toString() 
                        }
                    }
                
                val modelInfo = ModelInfo(
                    name = displayName,
                    modelUrl = modelUrl,
                    labelsUrl = labelsUrl,
                    type = modelType,
                    fileName = modelFileName,
                    labelsFileName = labelsFileName,
                    isDownloaded = isDownloaded
                )
                availableModels.add(modelInfo)
                
                Log.d("ModelAPI", "✅ Added model: $displayName")
                Log.d("ModelAPI", "   Type: $modelType")
                Log.d("ModelAPI", "   Model URL: $modelUrl")
                Log.d("ModelAPI", "   Labels URL: $labelsUrl")
                Log.d("ModelAPI", "   Downloaded: $isDownloaded")
            }
            
            Log.d("ModelAPI", "Successfully loaded ${availableModels.size} models from API")
            
        } catch (e: Exception) {
            Log.e("ModelAPI", "Error fetching model list: ${e.message}")
            e.printStackTrace()
            
            // Fallback to empty list - user will see error dialog
            availableModels.clear()
            throw e
        }
    }
}


    private suspend fun downloadModelIfNeeded(modelInfo: ModelInfo): Boolean {
        return try {
            if (!modelInfo.isDownloaded) {
                downloadModel(modelInfo)
                downloadLabels(modelInfo)
                modelInfo.isDownloaded = true
            }
            true
        } catch (e: Exception) {
            Log.e("ModelDownload", "Failed to download ${modelInfo.name}: ${e.message}")
            false
        }
    }

    private suspend fun downloadModel(modelInfo: ModelInfo) {
        val modelFile = File(filesDir, modelInfo.fileName)
        
        withContext(Dispatchers.IO) {
            try {
                Log.d("ModelDownload", "Downloading ${modelInfo.name}")
                val connection = URL(modelInfo.modelUrl).openConnection()
                connection.connect()
                val fileLength = connection.contentLength
                
                val input = BufferedInputStream(connection.getInputStream())
                val output = FileOutputStream(modelFile)
                val buffer = ByteArray(8192)
                var bytesRead: Int
                var totalBytes = 0L
                
                while (input.read(buffer).also { bytesRead = it } != -1) {
                    output.write(buffer, 0, bytesRead)
                    totalBytes += bytesRead
                    
                    if (fileLength > 0) {
                        val progress = (totalBytes * 100 / fileLength).toInt()
                        val mb = totalBytes / (1024.0 * 1024.0)
                        val totalMb = fileLength / (1024.0 * 1024.0)
                        
                        // Update UI every 1% or 100KB to avoid UI thread congestion
                        if (totalBytes % (50 * 1024) == 0L || progress % 1 == 0) {
                            updateDownloadProgress(progress, "Downloading ${String.format("%.1f", mb)}/${String.format("%.1f", totalMb)} MB")
                        }
                    } else {
                        // Unknown size
                        val mb = totalBytes / (1024.0 * 1024.0)
                        if (totalBytes % (100 * 1024) == 0L) {
                             updateDownloadProgress(0, "Downloading ${String.format("%.1f", mb)} MB...")
                        }
                    }
                }
                
                output.flush()
                output.close()
                input.close()
                Log.d("ModelDownload", "${modelInfo.name} downloaded successfully")
            } catch (e: Exception) {
                Log.e("ModelDownload", "Error downloading ${modelInfo.name}: ${e.message}")
                throw e
            }
        }
    }

    private suspend fun downloadLabels(modelInfo: ModelInfo) {
        val labelsFile = File(filesDir, modelInfo.labelsFileName)
        
        withContext(Dispatchers.IO) {
            try {
                Log.d("ModelDownload", "Downloading labels")
                val labelsText = URL(modelInfo.labelsUrl).readText()
                labelsFile.writeText(labelsText)
                Log.d("ModelDownload", "Labels downloaded successfully")
            } catch (e: Exception) {
                Log.e("ModelDownload", "Error downloading labels: ${e.message}")
                throw e
            }
        }
    }

    private fun loadModelFile(fileName: String): MappedByteBuffer {
        val modelFile = File(filesDir, fileName)
        val inputStream = FileInputStream(modelFile)
        val fileChannel = inputStream.channel
        val declaredLength = fileChannel.size()
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, 0L, declaredLength)
    }

    private fun loadAndInitializeModel(modelInfo: ModelInfo): Boolean {
    try {
        // CRITICAL: Stop processing during model switch
        val wasProcessing = isProcessing
        isProcessing = true
        
        // Check if already loaded
        if (modelInterpreters.containsKey(modelInfo.fileName)) {
            currentInterpreter = modelInterpreters[modelInfo.fileName]
            loadLabelsForModel(modelInfo)
            detectAndSetModelShapes(currentInterpreter!!, modelInfo.name)
            initializeHelperForModel(modelInfo)
            isProcessing = wasProcessing
            return true
        }
        
        val compatList = CompatibilityList()
        val interpreterOptions = Interpreter.Options()
        
        // IMPORTANT: Create a NEW GPU delegate for each model
        // Don't reuse the old one as it causes memory corruption
        var modelGpuDelegate: GpuDelegate? = null
        
        if(compatList.isDelegateSupportedOnThisDevice) {
            Log.d("ModelDebug", "✅ GPU acceleration enabled")
            try {
                modelGpuDelegate = GpuDelegate()
                interpreterOptions.addDelegate(modelGpuDelegate)
            } catch (e: Throwable) {
                Log.w("ModelDebug", "GPU delegate fallback to CPU: ${e.message}")
                modelGpuDelegate?.close()
                modelGpuDelegate = null
            }
        } else {
            Log.d("ModelDebug", "⚠️ Using CPU")
        }
        
        val interpreter = Interpreter(loadModelFile(modelInfo.fileName), interpreterOptions)
        
        // Store both interpreter and its dedicated GPU delegate
        modelInterpreters[modelInfo.fileName] = interpreter
        
        // Close old global GPU delegate if switching models
        if (modelGpuDelegate != null) {
            gpuDelegate?.close()
            gpuDelegate = modelGpuDelegate
        }
        
        currentInterpreter = interpreter
        
        Log.d("ModelDebug", "✅ Loaded ${modelInfo.name}")
        
        // Dynamically detect and set model shapes
        detectAndSetModelShapes(interpreter, modelInfo.name)
        
        // Load labels
        loadLabelsForModel(modelInfo)
        
        // Initialize specialized helpers if needed
        initializeHelperForModel(modelInfo)
        
        // Resume processing
        isProcessing = wasProcessing
        
        return true
    } catch (e: Exception) {
        Log.e("ModelDebug", "Error loading model: ${e.message}")
        e.printStackTrace()
        isProcessing = false
        return false
    }
}
    
private fun initializeHelperForModel(modelInfo: ModelInfo) {
    try {
        // CRITICAL: Clean up helpers in specific order to prevent crashes
        Log.d("ModelDebug", "Cleaning up old helpers...")
        
        // Close pose helper
        try {
            poseHelper?.close()
        } catch (e: Exception) {
            Log.e("ModelDebug", "Error closing pose helper: ${e.message}")
        }
        poseHelper = null
        
        // Close segmentation helper
        try {
            segmentationHelper?.close()
        } catch (e: Exception) {
            Log.e("ModelDebug", "Error closing segmentation helper: ${e.message}")
        }
        segmentationHelper = null
        
        // Close face helper
        try {
            faceHelper?.close()
        } catch (e: Exception) {
            Log.e("ModelDebug", "Error closing face helper: ${e.message}")
        }
        faceHelper = null
        
        // Clear annotated bitmap
        annotatedBitmap?.recycle()
        annotatedBitmap = null
        
        // Small delay to ensure cleanup is complete
        Thread.sleep(50)
        
        val modelFile = File(filesDir, modelInfo.fileName)
        val labelsFile = File(filesDir, modelInfo.labelsFileName)
        
        when (modelInfo.type.lowercase()) {
            "yolo 11 pose" -> {
                Log.d("ModelDebug", "Initializing PoseEstimationHelper")
                poseHelper = PoseEstimationHelper(this)
                poseHelper?.initialize(modelFile, labelsFile)
            }
            "yolo 11 segmentation" -> {
                Log.d("ModelDebug", "Initializing SegmentationHelper")
                segmentationHelper = SegmentationHelper(this)
                segmentationHelper?.initialize(modelFile, labelsFile)
            }
            "yolox", "yolo", "yolov5", "yolov8" -> {
                Log.d("ModelDebug", "Skipping helper initialization for YOLO variants (handled natively)")
            }
            else -> {
                Log.d("ModelDebug", "Initializing FaceDetectionHelper for model type: ${modelInfo.type}")
                faceHelper = FaceDetectionHelper(this)
                faceHelper?.initialize(modelFile, labelsFile)
            }
        }
    } catch (e: Exception) {
        Log.e("ModelDebug", "Error initializing helper: ${e.message}")
        e.printStackTrace()
    }
}



    private fun detectAndSetModelShapes(interpreter: Interpreter, modelName: String) {
        try {
            // Get input tensor shape
            val inputShape = interpreter.getInputTensor(0).shape()
            val inputDataType = interpreter.getInputTensor(0).dataType()
            
            Log.d("ModelShape", "📊 $modelName Input Shape: ${inputShape.contentToString()}")
            Log.d("ModelShape", "📊 $modelName Input DataType: $inputDataType")
            
            // Dynamically set input dimensions
            // Typical formats: [batch, height, width, channels] or [batch, channels, height, width]
            if (inputShape.size == 4) {
                // Determine format by checking channel position
                if (inputShape[3] == 3 || inputShape[3] == 1) {
                    // NHWC format: [1, height, width, channels]
                    INPUT_HEIGHT = inputShape[1]
                    INPUT_WIDTH = inputShape[2]
                    Log.d("ModelShape", "Detected NHWC format")
                } else if (inputShape[1] == 3 || inputShape[1] == 1) {
                    // NCHW format: [1, channels, height, width]
                    INPUT_HEIGHT = inputShape[2]
                    INPUT_WIDTH = inputShape[3]
                    Log.d("ModelShape", "Detected NCHW format")
                } else {
                    // Fallback: assume NHWC
                    Log.w("ModelShape", "⚠️ Unclear format, assuming NHWC")
                    INPUT_HEIGHT = inputShape[1]
                    INPUT_WIDTH = inputShape[2]
                }
                INPUT_SIZE = INPUT_HEIGHT
                
                Log.d("ModelShape", "✅ Dynamically detected input size: ${INPUT_WIDTH}x${INPUT_HEIGHT}")
            }
            
            // Get output tensor shape
            val outputShape = interpreter.getOutputTensor(0).shape()
            val outputDataType = interpreter.getOutputTensor(0).dataType()
            OUTPUT_SHAPE = outputShape
            
            Log.d("ModelShape", "📊 $modelName Output Shape: ${outputShape.contentToString()}")
            Log.d("ModelShape", "📊 $modelName Output DataType: $outputDataType")
            
            // Log all output tensors if multiple
            val numOutputs = interpreter.outputTensorCount
            Log.d("ModelShape", "📊 $modelName has $numOutputs output tensor(s)")
            for (i in 0 until numOutputs) {
                val shape = interpreter.getOutputTensor(i).shape()
                val dtype = interpreter.getOutputTensor(i).dataType()
                Log.d("ModelShape", "📊 Output[$i] Shape: ${shape.contentToString()}, Type: $dtype")
            }
            
            // Update image processor with dynamically detected dimensions
            imageProcessor = ImageProcessor.Builder()
                .add(ResizeOp(INPUT_WIDTH, INPUT_HEIGHT, ResizeOp.ResizeMethod.BILINEAR))
                .add(org.tensorflow.lite.support.image.ops.Rot90Op(0))
                .build()
            
            Log.d("ModelShape", "✅ Image processor updated for ${INPUT_WIDTH}x${INPUT_HEIGHT}")
            
        } catch (e: Exception) {
            Log.e("ModelShape", "❌ Error detecting model shapes: ${e.message}")
            e.printStackTrace()
            
            // Fallback to default values
            INPUT_SIZE = 300
            INPUT_WIDTH = 300
            INPUT_HEIGHT = 300
            OUTPUT_SHAPE = null
            
            imageProcessor = ImageProcessor.Builder()
                .add(ResizeOp(INPUT_WIDTH, INPUT_HEIGHT, ResizeOp.ResizeMethod.BILINEAR))
                .add(org.tensorflow.lite.support.image.ops.Rot90Op(0))
                .build()
        }
    }

    private fun loadLabelsForModel(modelInfo: ModelInfo) {
        val labelsFile = File(filesDir, modelInfo.labelsFileName)
        labels = labelsFile.readLines().filter { it.isNotBlank() }
        Log.d("ModelDebug", "Loaded ${labels.size} labels for ${modelInfo.name}")
    }

 private fun processFrameSynchronous(currentBitmap: Bitmap) {
    try {
        if (currentModelIndex == null || currentInterpreter == null) {
            isProcessing = false
            currentBitmap.recycle()
            return
        }
        
        // Check what data type the model expects
        val inputDataType = currentInterpreter!!.getInputTensor(0).dataType()
        
        // Create TensorImage with the correct data type
        var image = TensorImage(inputDataType)
        image.load(currentBitmap)
        image = imageProcessor.process(image)

        val result = runInference(image)
        
        // For pose/segmentation, composite the camera frame with the annotated overlay
        if (result.annotatedBitmap != null) {
            drawDetectionsWithComposite(result.detections, result.annotatedBitmap, currentBitmap)
        } else {
            drawDetections(result.detections, null)
        }
        
        isProcessing = false
        currentBitmap.recycle()
        
        updateFps()
        updateBottomDashboard()
        updateDashboardIfVisible()
        updateFpsText()
    } catch (e: Exception) {
        Log.e("ProcessingError", "Error: ${e.message}")
        e.printStackTrace()
        isProcessing = false
        currentBitmap.recycle()
    }
}


   private fun runInference(image: TensorImage): InferenceResult {
    val modelIndex = currentModelIndex ?: return InferenceResult(emptyList())
    val currentModel = availableModels[modelIndex]
    val interpreter = currentInterpreter ?: return InferenceResult(emptyList())
    
    return try {
        when (currentModel.type.lowercase()) {
            "yolo 11 segmentation" -> {
                Log.d("Inference", "Using SegmentationHelper for ${currentModel.name}")
                if (segmentationHelper != null) {
                    val bitmap = image.bitmap
                    val (segmentationResults, maskBitmap) = segmentationHelper!!.runInference(bitmap)
                    annotatedBitmap = maskBitmap
                    val detections = segmentationResults.map { result ->
                        Detection(
                            x1 = result.x1,
                            y1 = result.y1,
                            x2 = result.x2,
                            y2 = result.y2,
                            confidence = result.confidence,
                            classId = result.classId,
                            label = result.label
                        )
                    }
                    InferenceResult(detections, maskBitmap)
                } else {
                    Log.e("Inference", "SegmentationHelper not initialized")
                    InferenceResult(emptyList())
                }
            }
            "yolo 11 pose" -> {
                Log.d("Inference", "Using PoseEstimationHelper for ${currentModel.name}")
                if (poseHelper != null) {
                    val bitmap = image.bitmap
                    val (poseResults, poseBitmap) = poseHelper!!.runInference(bitmap)
                    annotatedBitmap = poseBitmap
                    val detections = poseResults.map { result ->
                        Detection(
                            x1 = result.x1,
                            y1 = result.y1,
                            x2 = result.x2,
                            y2 = result.y2,
                            confidence = result.confidence,
                            classId = 0,
                            label = "person"
                        )
                    }
                    InferenceResult(detections, poseBitmap)
                } else {
                    Log.e("Inference", "PoseEstimationHelper not initialized")
                    InferenceResult(emptyList())
                }
            }
            "yolox", "yolo", "yolov5", "yolov8" -> {
                Log.d("Inference", "Using native YOLO inference for ${currentModel.name}")
                InferenceResult(runYoloxInference(interpreter, image))
            }
            else -> {
                // ALL other models use FaceDetectionHelper
                // This includes: fire_smoke, dent, blink_drowse, helmet, license_plate, face_detection, etc.
                Log.d("Inference", "Using FaceDetectionHelper for ${currentModel.name} (type: ${currentModel.type})")
                if (faceHelper != null) {
                    val bitmap = image.bitmap
                    val detections = faceHelper!!.runInference(bitmap).map { det ->
                        Detection(det.x1, det.y1, det.x2, det.y2, det.confidence, det.classId, det.label)
                    }
                    InferenceResult(detections)
                } else {
                    Log.e("Inference", "FaceDetectionHelper not initialized for ${currentModel.name}")
                    InferenceResult(emptyList())
                }
            }
        }
    } catch (e: Exception) {
        Log.e("Inference", "Inference failed for ${currentModel.name}: ${e.message}")
        e.printStackTrace()
        InferenceResult(emptyList())
    }
}

    private fun runYoloxInference(interpreter: Interpreter, image: TensorImage): List<Detection> {
        val inputBuffer = image.tensorBuffer.buffer
        val outputShape = interpreter.getOutputTensor(0).shape()
        val outputSize = outputShape.fold(1) { acc, i -> acc * i }
        val outputBuffer = ByteBuffer.allocateDirect(outputSize * 4).order(ByteOrder.nativeOrder())
        
        interpreter.run(inputBuffer, outputBuffer)
        
        outputBuffer.rewind()
        val outputArray = FloatArray(outputSize)
        outputBuffer.asFloatBuffer().get(outputArray)
        
        return parseYoloxOutput(outputArray)
    }

    private fun runSsdInference(interpreter: Interpreter, image: TensorImage): List<Detection> {
    val inputBuffer = image.tensorBuffer.buffer
    
    // Log input details for debugging
    Log.d("SsdDebug", "Input buffer size: ${inputBuffer.capacity()} bytes")
    Log.d("SsdDebug", "Expected input: ${interpreter.getInputTensor(0).shape().contentToString()}")
    Log.d("SsdDebug", "Input data type: ${interpreter.getInputTensor(0).dataType()}")
    
    // Check if this model actually has 4 outputs (SSD format)
    val numOutputs = interpreter.outputTensorCount
    Log.d("SsdDebug", "Model has $numOutputs output tensor(s)")
    
    if (numOutputs < 4) {
        Log.e("SsdDebug", "⚠️ Model doesn't have 4 outputs (has $numOutputs). This might not be an SSD model. Falling back to YOLO inference.")
        // Fall back to single-output inference (might be YOLO format)
        return runYoloxInference(interpreter, image)
    }
    
    // SSD MobileNet typically outputs in this order:
    // Output 0: locations [1, 10, 4] - bounding boxes
    // Output 1: classes [1, 10] - class indices
    // Output 2: scores [1, 10] - confidence scores
    // Output 3: numberOfDetections [1] - number of valid detections
    
    val locationsShape = interpreter.getOutputTensor(0).shape()
    val classesShape = interpreter.getOutputTensor(1).shape()
    val scoresShape = interpreter.getOutputTensor(2).shape()
    val numDetShape = interpreter.getOutputTensor(3).shape()
    
    Log.d("SsdDebug", "Locations shape: ${locationsShape.contentToString()}")
    Log.d("SsdDebug", "Classes shape: ${classesShape.contentToString()}")
    Log.d("SsdDebug", "Scores shape: ${scoresShape.contentToString()}")
    Log.d("SsdDebug", "NumDet shape: ${numDetShape.contentToString()}")
    
    // Calculate actual sizes based on FLOAT32 data type (4 bytes per float)
    val locationsSize = locationsShape.fold(1) { acc, i -> acc * i }
    val classesSize = classesShape.fold(1) { acc, i -> acc * i }
    val scoresSize = scoresShape.fold(1) { acc, i -> acc * i }
    val numDetSize = numDetShape.fold(1) { acc, i -> acc * i }
    
    // Allocate properly sized ByteBuffers
    val locations = ByteBuffer.allocateDirect(locationsSize * 4).order(ByteOrder.nativeOrder())
    val classes = ByteBuffer.allocateDirect(classesSize * 4).order(ByteOrder.nativeOrder())
    val scores = ByteBuffer.allocateDirect(scoresSize * 4).order(ByteOrder.nativeOrder())
    val numDet = ByteBuffer.allocateDirect(numDetSize * 4).order(ByteOrder.nativeOrder())
    
    // Create output map - IMPORTANT: indices must match tensor output order
    val outputs = mapOf(
        0 to locations,
        1 to classes,
        2 to scores,
        3 to numDet
    )
    
    // Run inference
    try {
        interpreter.runForMultipleInputsOutputs(arrayOf(inputBuffer), outputs)
        Log.d("SsdDebug", "Inference completed successfully")
    } catch (e: Exception) {
        Log.e("SsdDebug", "Inference error: ${e.message}")
        e.printStackTrace()
        return emptyList()
    }
    
    return parseSsdOutput(locations, classes, scores, numDet)
}

private fun parseSsdOutput(
    locationsBuffer: ByteBuffer, 
    classesBuffer: ByteBuffer, 
    scoresBuffer: ByteBuffer, 
    numDetBuffer: ByteBuffer
): List<Detection> {
    val detections = mutableListOf<Detection>()
    
    // Rewind all buffers to start
    locationsBuffer.rewind()
    classesBuffer.rewind()
    scoresBuffer.rewind()
    numDetBuffer.rewind()
    
    // Convert ByteBuffers to FloatArrays
    val locationsArray = FloatArray(locationsBuffer.remaining() / 4)
    val classesArray = FloatArray(classesBuffer.remaining() / 4)
    val scoresArray = FloatArray(scoresBuffer.remaining() / 4)
    
    locationsBuffer.asFloatBuffer().get(locationsArray)
    classesBuffer.asFloatBuffer().get(classesArray)
    scoresBuffer.asFloatBuffer().get(scoresArray)
    
    val numberOfDetections = numDetBuffer.float.toInt()
    
    Log.d("SsdDebug", "Number of detections: $numberOfDetections")
    Log.d("SsdDebug", "Scores array size: ${scoresArray.size}")
    if (scoresArray.isNotEmpty()) {
        Log.d("SsdDebug", "First 5 scores: ${scoresArray.take(5).joinToString { "%.3f".format(it) }}")
    }
    
    // Process detections
    for (i in 0 until minOf(numberOfDetections, scoresArray.size)) {
        val score = scoresArray[i]
        
        if (score >= CONFIDENCE_THRESHOLD) {
            // SSD outputs locations as [ymin, xmin, ymax, xmax] normalized to [0, 1]
            val boxIndex = i * 4
            val ymin = locationsArray[boxIndex] * INPUT_SIZE
            val xmin = locationsArray[boxIndex + 1] * INPUT_SIZE
            val ymax = locationsArray[boxIndex + 2] * INPUT_SIZE
            val xmax = locationsArray[boxIndex + 3] * INPUT_SIZE
            
            val classId = classesArray[i].toInt()
            val label = if (classId < labels.size) labels[classId] else "Unknown"
            
            Log.d("SsdDebug", "✓ Detection: $label ($classId) conf=%.2f%% at [%.0f,%.0f,%.0f,%.0f]".format(
                score * 100, xmin, ymin, xmax, ymax))
            
            detections.add(Detection(xmin, ymin, xmax, ymax, score, classId, label))
        }
    }
    
    Log.d("SsdDebug", "Total valid detections: ${detections.size}")
    return detections
}

    private fun parseYoloxOutput(output: FloatArray): List<Detection> {
        val allDetections = mutableListOf<Detection>()
        
        val numValues = OUTPUT_SHAPE?.getOrNull(2) ?: 85
        val numPredictions = if (OUTPUT_SHAPE != null && OUTPUT_SHAPE!!.size >= 2) {
            OUTPUT_SHAPE!![1]
        } else {
            output.size / numValues
        }
        
        for (i in 0 until numPredictions) {
            val offset = i * numValues
            
            val xCenter = output[offset]
            val yCenter = output[offset + 1]
            val width = output[offset + 2]
            val height = output[offset + 3]
                val objectness = output[offset + 4]
                
                if (objectness < CONFIDENCE_THRESHOLD) continue
                
                val numClasses = numValues - 5
                var maxClassScore = 0f
                var maxClassId = 0
                for (c in 0 until numClasses) {
                    val classScore = output[offset + 5 + c]
                    if (classScore > maxClassScore) {
                        maxClassScore = classScore
                        maxClassId = c
                    }
                }
                
                val confidence = objectness * maxClassScore
                if (confidence < CONFIDENCE_THRESHOLD) continue
                
                val x1 = xCenter - width / 2f
                val y1 = yCenter - height / 2f
                val x2 = xCenter + width / 2f
                val y2 = yCenter + height / 2f
                
                val label = if (maxClassId < labels.size) labels[maxClassId] else "Unknown"
                
                allDetections.add(Detection(x1, y1, x2, y2, confidence, maxClassId, label))
        }
        
        return applyNMS(allDetections, NMS_THRESHOLD)
    }

    private fun applyNMS(detections: List<Detection>, iouThreshold: Float): List<Detection> {
        if (detections.isEmpty()) return emptyList()
        
        val sortedDetections = detections.sortedByDescending { it.confidence }
        val keep = mutableListOf<Detection>()
        val suppressed = BooleanArray(sortedDetections.size) { false }
        
        for (i in sortedDetections.indices) {
            if (suppressed[i]) continue
            keep.add(sortedDetections[i])
            
            for (j in (i + 1) until sortedDetections.size) {
                if (suppressed[j]) continue
                if (sortedDetections[i].classId == sortedDetections[j].classId) {
                    val iou = calculateIoU(sortedDetections[i], sortedDetections[j])
                    if (iou > iouThreshold) suppressed[j] = true
                }
            }
        }
        
        return keep
    }

    private fun calculateIoU(det1: Detection, det2: Detection): Float {
        val x1 = maxOf(det1.x1, det2.x1)
        val y1 = maxOf(det1.y1, det2.y1)
        val x2 = minOf(det1.x2, det2.x2)
        val y2 = minOf(det1.y2, det2.y2)
        
        val intersectionArea = maxOf(0f, x2 - x1) * maxOf(0f, y2 - y1)
        val area1 = (det1.x2 - det1.x1) * (det1.y2 - det1.y1)
        val area2 = (det2.x2 - det2.x1) * (det2.y2 - det2.y1)
        val unionArea = area1 + area2 - intersectionArea
        
        return if (unionArea > 0) intersectionArea / unionArea else 0f
    }

    private fun drawDetectionsWithComposite(detections: List<Detection>, annotatedBitmap: Bitmap, cameraFrame: Bitmap) {
        val canvas = overlayView.holder.lockCanvas()
        if (canvas != null) {
            try {
                val h = canvas.height.toFloat()
                val w = canvas.width.toFloat()
                
                canvas.drawColor(Color.TRANSPARENT, PorterDuff.Mode.CLEAR)

                // Draw ONLY the annotated bitmap (masks/poses) on the transparent overlay
                // The camera feed is already showing in the TextureView behind this overlay
                val annotRect = Rect(0, 0, annotatedBitmap.width, annotatedBitmap.height)
                val canvasRect = RectF(0f, 0f, w, h)
                canvas.drawBitmap(annotatedBitmap, annotRect, canvasRect, null)

                paint.textSize = h / 50f
                paint.strokeWidth = h / 250f
                paint.alpha = 255
                
                counts.clear()
                
                // IMPORTANT: When annotatedBitmap is present (pose/segmentation), 
                // it already contains properly drawn bounding boxes and other visualizations.
                // We should NOT draw the detection boxes again to avoid duplicate boxes.
                // However, we still count the detections for the dashboard.
                detections.forEach { detection ->
                    counts[detection.label] = (counts[detection.label] ?: 0) + 1
                }
            } finally {
                overlayView.holder.unlockCanvasAndPost(canvas)
            }
        }
    }
    
    private fun drawDetections(detections: List<Detection>, annotatedBitmap: Bitmap? = null) {
        val canvas = overlayView.holder.lockCanvas()
        if (canvas != null) {
            try {
                val h = canvas.height.toFloat()
                val w = canvas.width.toFloat()
                
                canvas.drawColor(Color.TRANSPARENT, PorterDuff.Mode.CLEAR)

                paint.textSize = h / 50f
                paint.strokeWidth = h / 250f
                paint.alpha = 255
                
                counts.clear()
                
                // Draw bounding boxes and labels for all detections
                detections.forEach { detection ->
                    val color = colors[detection.classId % colors.size]
                    paint.color = color
                    paint.style = Paint.Style.STROKE
                    
                    // Prefer scaling based on the actual last frame dimensions
                    // to match helpers that return coordinates in original frame space.
                    val baseWidth = if (lastFrameWidth > 0) lastFrameWidth.toFloat() else INPUT_WIDTH.toFloat()
                    val baseHeight = if (lastFrameHeight > 0) lastFrameHeight.toFloat() else INPUT_HEIGHT.toFloat()
                    val scaleX = w / baseWidth
                    val scaleY = h / baseHeight
                    
                    val left = detection.x1 * scaleX
                    val top = detection.y1 * scaleY
                    val right = detection.x2 * scaleX
                    val bottom = detection.y2 * scaleY
                    
                    // Draw bounding box
                    canvas.drawRect(left, top, right, bottom, paint)
                    
                    counts[detection.label] = (counts[detection.label] ?: 0) + 1

                    // Draw label background and text
                    paint.style = Paint.Style.FILL
                    val labelText = "${detection.label} ${String.format("%.0f%%", detection.confidence * 100)}"
                    
                    val textX = left
                    val textY = top - 5f
                    
                    paint.alpha = 200
                    val textBounds = Rect()
                    paint.getTextBounds(labelText, 0, labelText.length, textBounds)
                    canvas.drawRect(
                        textX - 4f,
                        textY - textBounds.height() - 4f,
                        textX + textBounds.width() + 4f,
                        textY + 4f,
                        paint
                    )
                    
                    paint.color = Color.WHITE
                    paint.alpha = 255
                    canvas.drawText(labelText, textX, textY, paint)
                }
            } finally {
                overlayView.holder.unlockCanvasAndPost(canvas)
            }
        }
    }

    override fun onDestroy() {
    super.onDestroy()
    
    // Stop processing first
    isProcessing = true
    
    // Close camera
    close_camera()
    
    // Close all helpers
    try {
        poseHelper?.close()
        segmentationHelper?.close()
        faceHelper?.close()
    } catch (e: Exception) {
        Log.e("Cleanup", "Error closing helpers: ${e.message}")
    }
    
    // Close all interpreters
    modelInterpreters.values.forEach { 
        try {
            it.close()
        } catch (e: Exception) {
            Log.e("Cleanup", "Error closing interpreter: ${e.message}")
        }
    }
    modelInterpreters.clear()
    
    // Close GPU delegate
    try {
        gpuDelegate?.close()
    } catch (e: Exception) {
        Log.e("Cleanup", "Error closing GPU delegate: ${e.message}")
    }
    
    // Clean up bitmaps
    annotatedBitmap?.recycle()
    annotatedBitmap = null
}
    
    override fun onPause() {
        super.onPause()
        Log.d("Lifecycle", "onPause - closing camera")
        close_camera()
    }
    
    override fun onResume() {
        super.onResume()
        Log.d("Lifecycle", "onResume - checking camera state")
        // Reopen camera if we have permission and surface is ready
        if (textureView.isAvailable && !isCameraOpen && 
            ContextCompat.checkSelfPermission(this, android.Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
            Log.d("Lifecycle", "Reopening camera")
            open_camera()
        }
    }

    @SuppressLint("MissingPermission")
    fun open_camera(){
        if (isCameraOpen) {
            Log.d("Camera", "Camera already open, skipping")
            return
        }
        
        try {
            Log.d("Camera", "Opening camera...")
            cameraManager.openCamera(cameraManager.cameraIdList[0], object:CameraDevice.StateCallback(){
                override fun onOpened(p0: CameraDevice) {
                    Log.d("Camera", "Camera opened successfully")
                    cameraDevice = p0
                    isCameraOpen = true
                    
                    try {
                        var surfaceTexture = textureView.surfaceTexture
                        if (surfaceTexture == null) {
                            Log.e("Camera", "Surface texture is null!")
                            isCameraOpen = false
                            p0.close()
                            return
                        }
                        
                        surfaceTexture.setDefaultBufferSize(1280, 720)
                        var surface = Surface(surfaceTexture)
                        var captureRequest = cameraDevice.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW)
                        captureRequest.addTarget(surface)

                        cameraDevice.createCaptureSession(listOf(surface), object: CameraCaptureSession.StateCallback(){
                            override fun onConfigured(p0: CameraCaptureSession) {
                                captureSession = p0
                                try {
                                    p0.setRepeatingRequest(captureRequest.build(), null, null)
                                    Log.d("Camera", "Preview started successfully")
                                } catch (e: Exception) {
                                    Log.e("Camera", "Failed to start preview: ${e.message}")
                                    e.printStackTrace()
                                }
                            }
                            override fun onConfigureFailed(p0: CameraCaptureSession) {
                                Log.e("Camera", "Session configuration failed")
                                isCameraOpen = false
                            }
                        }, handler)
                    } catch (e: Exception) {
                        Log.e("Camera", "Error creating capture session: ${e.message}")
                        e.printStackTrace()
                        isCameraOpen = false
                        p0.close()
                    }
                }
                override fun onDisconnected(p0: CameraDevice) {
                    Log.w("Camera", "Camera disconnected")
                    isCameraOpen = false
                    captureSession = null
                    p0.close()
                }
                override fun onError(p0: CameraDevice, p1: Int) {
                    Log.e("Camera", "Camera error: $p1")
                    isCameraOpen = false
                    captureSession = null
                    p0.close()
                }
            }, handler)
        } catch (e: Exception) {
            Log.e("Camera", "Exception opening camera: ${e.message}")
            e.printStackTrace()
            isCameraOpen = false
        }
    }
    
    fun close_camera() {
        if (!isCameraOpen) {
            Log.d("Camera", "Camera already closed")
            return
        }
        
        try {
            Log.d("Camera", "Closing camera...")
            captureSession?.close()
            captureSession = null
            
            if (::cameraDevice.isInitialized) {
                cameraDevice.close()
            }
            
            isCameraOpen = false
            Log.d("Camera", "Camera closed successfully")
        } catch (e: Exception) {
            Log.e("Camera", "Error closing camera: ${e.message}")
            e.printStackTrace()
            isCameraOpen = false
        }
    }

    fun get_permission(){
        if(ContextCompat.checkSelfPermission(this, android.Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED){
            requestPermissions(arrayOf(android.Manifest.permission.CAMERA), 101)
        }
    }
    
    override fun onRequestPermissionsResult(requestCode: Int, permissions: Array<out String>, grantResults: IntArray) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if(grantResults.isNotEmpty() && grantResults[0] != PackageManager.PERMISSION_GRANTED) {
            get_permission()
        } else if (grantResults.isNotEmpty() && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
            Log.d("Permission", "Camera permission granted")
            // Permission granted, open camera if surface is ready
            if (textureView.isAvailable && !isCameraOpen) {
                open_camera()
            }
        }
    }
    
    private fun showLoadingDialog(message: String) {
        runOnUiThread {
            val builder = AlertDialog.Builder(this)
            builder.setTitle("Loading")
            builder.setMessage(message)
            builder.setCancelable(false)
            loadingDialog = builder.create()
            loadingDialog?.show()
        }
    }
    
    private fun updateLoadingDialog(message: String) {
        runOnUiThread {
            loadingDialog?.setMessage(message)
        }
    }
    
    private fun dismissLoadingDialog(message: String) {
        runOnUiThread {
            loadingDialog?.setMessage(message)
            Handler(mainLooper).postDelayed({
                loadingDialog?.dismiss()
                loadingDialog = null
            }, 800)
        }
    }

    private fun showDownloadProgressDialog(title: String) {
        runOnUiThread {
            val builder = AlertDialog.Builder(this)
            builder.setTitle(title)
            builder.setCancelable(false)
            
            // Create layout programmatically
            val layout = LinearLayout(this)
            layout.orientation = LinearLayout.VERTICAL
            layout.setPadding(50, 40, 50, 10)
            
            downloadProgressBar = android.widget.ProgressBar(this, null, android.R.attr.progressBarStyleHorizontal)
            downloadProgressBar?.isIndeterminate = false
            downloadProgressBar?.max = 100
            layout.addView(downloadProgressBar)
            
            downloadProgressText = TextView(this)
            downloadProgressText?.text = "Starting download..."
            downloadProgressText?.setPadding(0, 20, 0, 0)
            downloadProgressText?.gravity = Gravity.CENTER
            layout.addView(downloadProgressText)
            
            builder.setView(layout)
            loadingDialog = builder.create()
            loadingDialog?.show()
        }
    }
    
    private fun updateDownloadProgress(progress: Int, message: String) {
        runOnUiThread {
            downloadProgressBar?.progress = progress
            downloadProgressText?.text = message
        }
    }
    
    private fun showErrorDialog(message: String) {
        runOnUiThread {
            AlertDialog.Builder(this)
                .setTitle("Error")
                .setMessage(message)
                .setPositiveButton("OK") { dialog, _ -> dialog.dismiss() }
                .show()
        }
    }

    private fun updateFps() {
        frameCounter++
        val now = System.currentTimeMillis()
        val elapsed = now - lastFpsTimestamp
        if (elapsed >= 1000) {
            currentFps = (frameCounter * 1000.0) / elapsed
            frameCounter = 0
            lastFpsTimestamp = now
        }
    }
    
    private fun updateFpsText() {
        runOnUiThread {
            fpsTextView.text = String.format("FPS: %.2f", currentFps)
        }
    }

    private fun updateBottomDashboard() {
        val top2 = counts.entries.sortedByDescending { it.value }.take(2)
        
        runOnUiThread {
            if (top2.isNotEmpty()) {
                val (label1, cnt1) = top2[0]
                val emoji1 = labelToEmoji[label1] ?: label1
                dashboardEmoji1.text = emoji1
                dashboardCount1.text = cnt1.toString()
            } else {
                dashboardEmoji1.text = "🔍"
                dashboardCount1.text = "0"
            }
            
            if (top2.size >= 2) {
                val (label2, cnt2) = top2[1]
                val emoji2 = labelToEmoji[label2] ?: label2
                dashboardEmoji2.text = emoji2
                dashboardCount2.text = cnt2.toString()
            } else {
                dashboardEmoji2.text = "🔍"
                dashboardCount2.text = "0"
            }
        }
    }

    private var dashboardDialog: AlertDialog? = null
    private var dashboardContainer: LinearLayout? = null
    
    private fun showExpandedDashboard() {
        if (dashboardDialog != null) {
            dashboardDialog?.show()
            return
        }
        val container = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(32,32,32,32)
        }
        dashboardContainer = container
        val builder = AlertDialog.Builder(this, android.R.style.Theme_DeviceDefault_Dialog)
            .setTitle("📊 Current Frames Detected Objects Dashboard")
            .setView(container)
            .setPositiveButton("Close") { d, _ -> d.dismiss() }
        dashboardDialog = builder.create()
        dashboardDialog?.show()
        updateDashboardIfVisible()
    }
    
    private fun updateDashboardIfVisible() {
        val dialog = dashboardDialog ?: return
        if (!dialog.isShowing) return
        val container = dashboardContainer ?: return
        runOnUiThread {
            container.removeAllViews()
            if (counts.isEmpty()) {
                val tv = TextView(this).apply {
                    text = "🔍 No objects detected"
                    textSize = 16f
                    gravity = Gravity.CENTER
                    setPadding(16, 32, 16, 32)
                    setTextColor(Color.GRAY)
                }
                container.addView(tv)
            } else {
                val header = TextView(this).apply {
                    text = "Live Detection (${counts.values.sum()} total)"
                    textSize = 14f
                    setTextColor(Color.GRAY)
                    setPadding(0, 0, 0, 16)
                }
                container.addView(header)
                
                counts.entries.sortedByDescending { it.value }.forEach { (label, cnt) ->
                    val row = LinearLayout(this).apply {
                        orientation = LinearLayout.HORIZONTAL
                        gravity = Gravity.CENTER_VERTICAL
                        setPadding(8, 12, 8, 12)
                    }

                    val isEmoji = labelToEmoji.containsKey(label)
                    val emoji = if (isEmoji) labelToEmoji[label] else label
                    
                    val emojiView = TextView(this@MainActivity).apply {
                        text = emoji
                        textSize = if (isEmoji) 32f else 18f
                        setPadding(0, 0, 16, 0)
                    }
                    
                    val countBadge = TextView(this@MainActivity).apply {
                        text = "× $cnt"
                        textSize = 20f
                        setTextColor(Color.WHITE)
                        setPadding(16, 8, 16, 8)
                        setBackgroundResource(android.R.drawable.dialog_holo_dark_frame)
                        gravity = Gravity.CENTER
                        layoutParams = LinearLayout.LayoutParams(
                            LinearLayout.LayoutParams.WRAP_CONTENT,
                            LinearLayout.LayoutParams.WRAP_CONTENT
                        ).apply {
                            setMargins(0, 0, 0, 0)
                        }
                    }

                    row.addView(emojiView)
                    row.addView(countBadge)
                    container.addView(row)
                }
            }
        }
    }
    
    private fun showModelSelectionDialog() {
        if (availableModels.isEmpty()) {
            AlertDialog.Builder(this)
                .setTitle("No Models Available")
                .setMessage("Failed to fetch models list. Please check your internet connection and restart the app.")
                .setPositiveButton("Retry") { _, _ ->
                    // Retry fetching model list
                    showLoadingDialog("Fetching available models...")
                    GlobalScope.launch(Dispatchers.IO) {
                        try {
                            fetchModelList()
                            runOnUiThread {
                                dismissLoadingDialog("Models list loaded!")
                                showModelSelectionDialog()
                            }
                        } catch (e: Exception) {
                            Log.e("ModelAPI", "Error: ${e.message}")
                            runOnUiThread {
                                dismissLoadingDialog("Error: ${e.message}")
                                showErrorDialog("Failed to fetch model list. Check your connection.")
                            }
                        }
                    }
                }
                .setNegativeButton("Cancel", null)
                .show()
            return
        }
        
        // Create custom list items showing model name and download status
        val modelItems = availableModels.map { model ->
            val typeTag = model.type.uppercase()
            if (model.isDownloaded) {
                "${model.name} ✓ [$typeTag]"
            } else {
                "${model.name} (Download) [$typeTag]"
            }
        }.toTypedArray()
        
        val currentSelection = currentModelIndex ?: -1
        
        AlertDialog.Builder(this)
            .setTitle("Select Detection Model")
            .setSingleChoiceItems(modelItems, currentSelection) { dialog, which ->
                dialog.dismiss()
                val selectedModel = availableModels[which]
                
                if (selectedModel.isDownloaded) {
                    // Model already downloaded, just load it
                    loadAndSwitchToModel(which)
                } else {
                    // Need to download the model first
                    downloadAndLoadModel(which)
                }
            }
            .setNeutralButton("Refresh List") { _, _ ->
                // Refresh model list from API
                showLoadingDialog("Refreshing model list...")
                GlobalScope.launch(Dispatchers.IO) {
                    try {
                        fetchModelList()
                        runOnUiThread {
                            dismissLoadingDialog("Models list updated!")
                            showModelSelectionDialog()
                        }
                    } catch (e: Exception) {
                        Log.e("ModelAPI", "Error: ${e.message}")
                        runOnUiThread {
                            dismissLoadingDialog("Error: ${e.message}")
                            showErrorDialog("Failed to refresh model list.")
                        }
                    }
                }
            }
            .setNegativeButton("Cancel", null)
            .show()
    }
    
    private fun loadAndSwitchToModel(modelIndex: Int) {
    val modelInfo = availableModels[modelIndex]
    
    // CRITICAL: Stop all frame processing before switching
    isProcessing = true
    
    showLoadingDialog("Loading ${modelInfo.name}...")
    
    GlobalScope.launch(Dispatchers.Main) {
        // Add a small delay to let any in-flight processing complete
        delay(100)
        
        val success = withContext(Dispatchers.IO) {
            loadAndInitializeModel(modelInfo)
        }
        
        if (success) {
            currentModelIndex = modelIndex
            dismissLoadingDialog("${modelInfo.name} loaded!")
            
            // Open camera if not already open
            if (textureView.isAvailable && !this@MainActivity::cameraDevice.isInitialized) {
                open_camera()
            }
            
            Log.d("ModelDebug", "✅ Switched to: ${modelInfo.name}")
            Log.d("ModelDebug", "✅ Input size: ${INPUT_WIDTH}x${INPUT_HEIGHT}")
            Log.d("ModelDebug", "✅ Output shape: ${OUTPUT_SHAPE?.contentToString()}")
            
            // Resume frame processing after a small delay
            delay(100)
            isProcessing = false
        } else {
            dismissLoadingDialog("Failed to load ${modelInfo.name}")
            showErrorDialog("Failed to load ${modelInfo.name}. Please try again.")
            isProcessing = false
        }
    }
}

    
    private fun downloadAndLoadModel(modelIndex: Int) {
    val modelInfo = availableModels[modelIndex]
    
    // CRITICAL: Stop all frame processing before downloading/switching
    isProcessing = true
    
    showDownloadProgressDialog("Downloading ${modelInfo.name}")
    
    GlobalScope.launch(Dispatchers.Main) {
        val downloadSuccess = withContext(Dispatchers.IO) {
            downloadModelIfNeeded(modelInfo)
        }
        
        if (downloadSuccess) {
            updateDownloadProgress(100, "Loading model options...")
            
            // Add delay before loading
            delay(100)
            
            val loadSuccess = withContext(Dispatchers.IO) {
                loadAndInitializeModel(modelInfo)
            }
            
            if (loadSuccess) {
                currentModelIndex = modelIndex
                dismissLoadingDialog("${modelInfo.name} ready!")
                
                // Open camera if not already open
                if (textureView.isAvailable && !this@MainActivity::cameraDevice.isInitialized) {
                    open_camera()
                }
                
                Log.d("ModelDebug", "✅ Downloaded and loaded: ${modelInfo.name}")
                Log.d("ModelDebug", "✅ Input size: ${INPUT_WIDTH}x${INPUT_HEIGHT}")
                Log.d("ModelDebug", "✅ Output shape: ${OUTPUT_SHAPE?.contentToString()}")
                
                // Resume processing after delay
                delay(100)
                isProcessing = false
            } else {
                dismissLoadingDialog("Failed to load ${modelInfo.name}")
                showErrorDialog("Downloaded but failed to load ${modelInfo.name}. Please try again.")
                isProcessing = false
            }
        } else {
            dismissLoadingDialog("Download failed")
            showErrorDialog("Failed to download ${modelInfo.name}. Please check your internet connection.")
            isProcessing = false
        }
    }
}

    
    private fun showUploadOptions() {
        val options = arrayOf("Image", "Video")
        val builder = AlertDialog.Builder(this)
        builder.setTitle("Select Media Type")
        builder.setItems(options) { _, which ->
            when (which) {
                0 -> pickMediaLauncher.launch("image/*")
                1 -> pickMediaLauncher.launch("video/*")
            }
        }
        builder.show()
    }
    
    private fun handleUploadedMedia(uri: Uri) {
        try {
            switchToStaticMode()
            
            val bitmap = MediaStore.Images.Media.getBitmap(contentResolver, uri)
            staticImageView.setImageBitmap(bitmap)
            
            // Show model selection dialog if not already selected
            if (currentInterpreter == null) {
                showModelSelectionDialog()
            } else {
                processStaticImage(bitmap)
            }
        } catch (e: Exception) {
            showErrorDialog("Failed to load image: ${e.message}")
            switchToCameraMode()
        }
    }
    
    private fun switchToCameraMode() {
        isInCameraMode = true
        textureView.visibility = View.VISIBLE
        staticImageView.visibility = View.GONE
        
        if (textureView.isAvailable && !this::cameraDevice.isInitialized) {
            open_camera()
        }
    }
    
    private fun switchToStaticMode() {
        isInCameraMode = false
        textureView.visibility = View.GONE
        staticImageView.visibility = View.VISIBLE
        
        if (this::cameraDevice.isInitialized) {
            cameraDevice.close()
        }
    }
    
    private fun processStaticImage(bitmap: Bitmap) {
        if (currentInterpreter == null) {
            showErrorDialog("Please select a model first")
            return
        }
        
        try {
            // Create a mutable bitmap for drawing results
            val mutableBitmap = bitmap.copy(Bitmap.Config.ARGB_8888, true)

            // Match the camera pipeline: use the model's expected input data type
            val inputDataType = currentInterpreter!!.getInputTensor(0).dataType()
            var tensorImage = TensorImage(inputDataType)
            tensorImage.load(bitmap)
            val processedImage = imageProcessor.process(tensorImage)
            
            // Run inference
            val result = runInference(processedImage)
            
            // If we have an annotated bitmap (pose/segmentation), composite it with the original image
            if (result.annotatedBitmap != null) {
                // IMPORTANT: annotatedBitmap is at INPUT_WIDTH x INPUT_HEIGHT dimensions
                // We need to scale it to match the original bitmap dimensions
                val scaledAnnotatedBitmap = Bitmap.createScaledBitmap(
                    result.annotatedBitmap,
                    mutableBitmap.width,
                    mutableBitmap.height,
                    true
                )
                
                // Create a composite image: original + masks/poses
                val compositeBitmap = mutableBitmap.copy(Bitmap.Config.ARGB_8888, true)
                val canvas = Canvas(compositeBitmap)
                
                // Draw the scaled annotated overlay on top of the original image
                // NOTE: annotatedBitmap already contains properly drawn bounding boxes,
                // skeleton overlays, and all necessary visualizations.
                // DO NOT redraw detection boxes to avoid duplicate boxes!
                canvas.drawBitmap(scaledAnnotatedBitmap, 0f, 0f, null)
                
                // For pose/segmentation models, the annotatedBitmap is the complete visualization.
                // Detections list is only used for counting/statistics, not for drawing.
                
                staticImageView.setImageBitmap(compositeBitmap)
                
                // Clean up scaled bitmap
                scaledAnnotatedBitmap.recycle()
                return
            }
            
            // Otherwise, draw detections on bitmap
            val canvas = Canvas(mutableBitmap)
            val scaleX = mutableBitmap.width.toFloat() / INPUT_WIDTH
            val scaleY = mutableBitmap.height.toFloat() / INPUT_HEIGHT
            
            result.detections.forEach { detection ->
                paint.color = colors[detection.classId % colors.size]
                val rect = RectF(
                    detection.x1 * scaleX,
                    detection.y1 * scaleY,
                    detection.x2 * scaleX,
                    detection.y2 * scaleY
                )
                canvas.drawRect(rect, paint)
                
                // Draw label
                paint.style = Paint.Style.FILL
                paint.textSize = 40f
                canvas.drawText("${detection.label} ${(detection.confidence * 100).toInt()}%",
                    rect.left, rect.top - 10, paint)
                paint.style = Paint.Style.STROKE
            }
            
            // Update the image view
            staticImageView.setImageBitmap(mutableBitmap)
            
        } catch (e: Exception) {
            showErrorDialog("Detection failed: ${e.message}")
            Log.e("StaticImage", "Error processing image", e)
        }
    }
}