package com.example.verseye_demo.helpers

import android.content.Context
import android.graphics.*
import android.util.Log
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.CompatibilityList
import org.tensorflow.lite.gpu.GpuDelegate
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.DataType
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.common.ops.CastOp
import java.io.*
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.max
import kotlin.math.min

class PoseEstimationHelper(
    private val context: Context
) {
    
    companion object {
        private const val TAG = "PoseEstimationHelper"
        private const val OUTPUT_FEATURES = 56  // xywh(4) + conf(1) + keypoints(17*3=51)
        private const val KEYPOINTS_COUNT = 17
        private const val KEYPOINTS_FEATURES = KEYPOINTS_COUNT * 3 // x, y, conf per keypoint
    }
    
    private var interpreter: Interpreter? = null
    private var labels = listOf<String>()
    private var gpuDelegate: GpuDelegate? = null
    
    // Model input/output dimensions
    private var inputWidth = 640
    private var inputHeight = 640
    
    private var batchSize = 0
    private var numAnchors = 0
    
    private lateinit var imageProcessor: ImageProcessor
    
    private var confidenceThreshold = 0.5f
    private var iouThreshold = 0.45f
    
    // Reuse buffers
    private lateinit var inputBuffer: ByteBuffer
    private lateinit var outputArray: Array<Array<FloatArray>>
    
    // Skeleton connections for drawing
    private val skeletonConnections = listOf(
        // Head
        0 to 1, 0 to 2,  // nose to eyes
        1 to 3, 2 to 4,  // eyes to ears
        
        // Torso
        5 to 6,  // shoulders
        5 to 11, 6 to 12,  // shoulders to hips
        11 to 12,  // hips
        
        // Arms
        5 to 7, 7 to 9,  // left arm
        6 to 8, 8 to 10,  // right arm
        
        // Legs
        11 to 13, 13 to 15,  // left leg
        12 to 14, 14 to 16   // right leg
    )
    
    private val keypointColors = intArrayOf(
        Color.RED, Color.RED, Color.RED, Color.RED, Color.RED,  // Head (0-4)
        Color.BLUE, Color.GREEN,  // Shoulders (5-6)
        Color.BLUE, Color.GREEN,  // Elbows (7-8)
        Color.BLUE, Color.GREEN,  // Wrists (9-10)
        Color.CYAN, Color.MAGENTA,  // Hips (11-12)
        Color.CYAN, Color.MAGENTA,  // Knees (13-14)
        Color.CYAN, Color.MAGENTA   // Ankles (15-16)
    )
    
    data class PoseResult(
        val x1: Float,
        val y1: Float,
        val x2: Float,
        val y2: Float,
        val confidence: Float,
        val keypoints: List<Keypoint>
    )
    
    data class Keypoint(
        val x: Float,
        val y: Float,
        val confidence: Float
    )
    
    private data class PoseDetection(
        val box: RectF,
        val confidence: Float,
        val keypoints: List<Keypoint>
    )
    
    fun initialize(modelFile: File, labelsFile: File): Boolean {
        return try {
            Log.d(TAG, "Initializing pose estimation model...")
            
            val modelInputStream = FileInputStream(modelFile)
            val modelFileChannel = modelInputStream.channel
            val modelByteBuffer = modelFileChannel.map(
                java.nio.channels.FileChannel.MapMode.READ_ONLY,
                0,
                modelFile.length()
            )
            modelInputStream.close()
            
            val labelsInputStream = FileInputStream(labelsFile)
            val labelsReader = BufferedReader(InputStreamReader(labelsInputStream))
            labels = labelsReader.readLines().filter { it.isNotBlank() }
            labelsReader.close()
            
            Log.d(TAG, "Loaded ${labels.size} labels")
            
            val compatList = CompatibilityList()
            val options = Interpreter.Options().apply {
                if (compatList.isDelegateSupportedOnThisDevice) {
                    try {
                        gpuDelegate = GpuDelegate()
                        addDelegate(gpuDelegate)
                        Log.d(TAG, "GPU delegate enabled")
                    } catch (e: Throwable) {
                        setNumThreads(Runtime.getRuntime().availableProcessors())
                        Log.w(TAG, "GPU delegate fallback to CPU: ${e.message}")
                    }
                } else {
                    setNumThreads(Runtime.getRuntime().availableProcessors())
                    Log.d(TAG, "GPU not supported, using CPU with ${Runtime.getRuntime().availableProcessors()} threads")
                }
            }
            
            interpreter = Interpreter(modelByteBuffer, options)
            interpreter!!.allocateTensors()
            
            val inputShape = interpreter!!.getInputTensor(0).shape()
            inputHeight = inputShape[1]
            inputWidth = inputShape[2]
            
            Log.d(TAG, "Input shape: [${inputShape.joinToString()}]")
            
            val outputShape = interpreter!!.getOutputTensor(0).shape()
            Log.d(TAG, "Output shape: [${outputShape.joinToString()}]")
            
            batchSize = outputShape[0]
            val outFeatures = outputShape[1]
            numAnchors = outputShape[2]
            
            require(outFeatures == OUTPUT_FEATURES) {
                "Unexpected output feature size. Expected=$OUTPUT_FEATURES, Actual=$outFeatures"
            }
            
            outputArray = Array(batchSize) {
                Array(outFeatures) { FloatArray(numAnchors) }
            }
            
            val inputBytes = 1 * inputHeight * inputWidth * 3 * 4
            inputBuffer = ByteBuffer.allocateDirect(inputBytes).apply {
                order(ByteOrder.nativeOrder())
            }
            
            imageProcessor = ImageProcessor.Builder()
                .add(ResizeOp(inputHeight, inputWidth, ResizeOp.ResizeMethod.BILINEAR))
                .add(NormalizeOp(0f, 255f))
                .add(CastOp(DataType.FLOAT32))
                .build()
            
            Log.d(TAG, "Pose estimation model initialized successfully")
            true
        } catch (e: Exception) {
            Log.e(TAG, "Error initializing pose estimation model", e)
            false
        }
    }
    
    fun runInference(bitmap: Bitmap): Pair<List<PoseResult>, Bitmap?> {
        val currentInterpreter = interpreter ?: run {
            Log.e(TAG, "Interpreter not initialized")
            return Pair(emptyList(), null)
        }
        
        return try {
            val tensorImage = TensorImage(DataType.FLOAT32)
            tensorImage.load(bitmap)
            val processedImage = imageProcessor.process(tensorImage)
            
            inputBuffer.clear()
            inputBuffer.put(processedImage.buffer)
            inputBuffer.rewind()
            
            currentInterpreter.run(inputBuffer, outputArray)
            
            val detections = postProcessPose(
                features = outputArray[0],
                numAnchors = numAnchors,
                confidenceThreshold = confidenceThreshold,
                iouThreshold = iouThreshold,
                origWidth = bitmap.width,
                origHeight = bitmap.height
            )
            
            val results = detections.map { det ->
                PoseResult(
                    x1 = det.box.left,
                    y1 = det.box.top,
                    x2 = det.box.right,
                    y2 = det.box.bottom,
                    confidence = det.confidence,
                    keypoints = det.keypoints
                )
            }
            
            val visualizationBitmap = if (detections.isNotEmpty()) {
                drawPoseOnBitmap(bitmap, detections)
            } else {
                null
            }
            
            Pair(results, visualizationBitmap)
        } catch (e: Exception) {
            Log.e(TAG, "Error during inference", e)
            Pair(emptyList(), null)
        }
    }
    
    private fun postProcessPose(
        features: Array<FloatArray>,
        numAnchors: Int,
        confidenceThreshold: Float,
        iouThreshold: Float,
        origWidth: Int,
        origHeight: Int
    ): List<PoseDetection> {
        val detections = mutableListOf<PoseDetection>()
        
        val scaleX = origWidth.toFloat() / inputWidth
        val scaleY = origHeight.toFloat() / inputHeight
        
        for (j in 0 until numAnchors) {
            val rawX = features[0][j]
            val rawY = features[1][j]
            val rawW = features[2][j]
            val rawH = features[3][j]
            val conf = features[4][j]
            
            if (conf < confidenceThreshold) continue
            
            val xScaled = rawX * inputWidth
            val yScaled = rawY * inputHeight
            val wScaled = rawW * inputWidth
            val hScaled = rawH * inputHeight
            
            val left   = (xScaled - wScaled / 2f) * scaleX
            val top    = (yScaled - hScaled / 2f) * scaleY
            val right  = (xScaled + wScaled / 2f) * scaleX
            val bottom = (yScaled + hScaled / 2f) * scaleY
            
            val rectF = RectF(left, top, right, bottom)
            
            val keypoints = mutableListOf<Keypoint>()
            for (k in 0 until KEYPOINTS_COUNT) {
                val rawKx = features[5 + k * 3][j]
                val rawKy = features[5 + k * 3 + 1][j]
                val kpConf = features[5 + k * 3 + 2][j]
                
                // Check if normalized (0-1 range) or absolute
                val isNormalized = rawKx <= 1.0f && rawKy <= 1.0f
                
                val finalKx: Float
                val finalKy: Float
                
                if (isNormalized) {
                    finalKx = rawKx * inputWidth * scaleX
                    finalKy = rawKy * inputHeight * scaleY
                } else {
                    finalKx = rawKx * scaleX
                    finalKy = rawKy * scaleY
                }
                
                keypoints.add(Keypoint(finalKx, finalKy, kpConf))
            }
            
            detections.add(PoseDetection(rectF, conf, keypoints))
        }
        
        Log.d(TAG, "Before NMS: Found ${detections.size} pose detections with conf >= $confidenceThreshold")
        val filtered = applyNMS(detections, iouThreshold)
        Log.d(TAG, "After NMS: Kept ${filtered.size} pose detections")
        
        return filtered
    }
    
    private fun applyNMS(
        detections: List<PoseDetection>,
        iouThreshold: Float
    ): List<PoseDetection> {
        if (detections.isEmpty()) return emptyList()
        
        val sorted = detections.sortedByDescending { it.confidence }
        val picked = mutableListOf<PoseDetection>()
        val used = BooleanArray(sorted.size)
        
        // Very aggressive NMS to prevent duplicate detections
        // Use 30% of original threshold for much stricter filtering
        val strictIouThreshold = iouThreshold * 0.3f
        
        for (i in sorted.indices) {
            if (used[i]) continue
            
            picked.add(sorted[i])
            
            for (j in i + 1 until sorted.size) {
                if (used[j]) continue
                val overlapRatio = iou(sorted[i].box, sorted[j].box)
                
                // Suppress any detection that has significant overlap
                if (overlapRatio > strictIouThreshold) {
                    used[j] = true
                    Log.d(TAG, "Suppressed duplicate: IoU=$overlapRatio with best detection")
                }
            }
        }
        
        Log.d(TAG, "NMS: ${detections.size} detections -> ${picked.size} after filtering")
        return picked
    }
    
    private fun iou(a: RectF, b: RectF): Float {
        val interLeft = max(a.left, b.left)
        val interTop = max(a.top, b.top)
        val interRight = min(a.right, b.right)
        val interBottom = min(a.bottom, b.bottom)
        val interW = max(0f, interRight - interLeft)
        val interH = max(0f, interBottom - interTop)
        val interArea = interW * interH
        val unionArea = a.width() * a.height() + b.width() * b.height() - interArea
        return if (unionArea <= 0f) 0f else (interArea / unionArea)
    }
    
    private fun drawPoseOnBitmap(
        bitmap: Bitmap,
        detections: List<PoseDetection>
    ): Bitmap {
        val output = Bitmap.createBitmap(bitmap.width, bitmap.height, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(output)
        
        // Draw semi-transparent overlay
        val overlayPaint = Paint().apply {
            color = Color.argb(0, 0, 0, 0) // Fully transparent
        }
        canvas.drawRect(0f, 0f, bitmap.width.toFloat(), bitmap.height.toFloat(), overlayPaint)
        
        // Draw each detected pose
        for (detection in detections) {
            // Draw bounding box
            val boxPaint = Paint().apply {
                style = Paint.Style.STROKE
                color = Color.GREEN
                strokeWidth = 4f
            }
            canvas.drawRect(detection.box, boxPaint)
            
            // Draw confidence
            val textPaint = Paint().apply {
                color = Color.WHITE
                textSize = 40f
                style = Paint.Style.FILL
            }
            val backgroundPaint = Paint().apply {
                color = Color.argb(200, 0, 128, 0)
                style = Paint.Style.FILL
            }
            val confText = "Person ${String.format("%.0f%%", detection.confidence * 100)}"
            val textBounds = Rect()
            textPaint.getTextBounds(confText, 0, confText.length, textBounds)
            canvas.drawRect(
                detection.box.left,
                detection.box.top - textBounds.height() - 10f,
                detection.box.left + textBounds.width() + 10f,
                detection.box.top,
                backgroundPaint
            )
            canvas.drawText(confText, detection.box.left + 5f, detection.box.top - 5f, textPaint)
            
            // Draw skeleton connections
            val linePaint = Paint().apply {
                style = Paint.Style.STROKE
                strokeWidth = 3f
                color = Color.YELLOW
                isAntiAlias = true
            }
            
            for ((start, end) in skeletonConnections) {
                val kp1 = detection.keypoints[start]
                val kp2 = detection.keypoints[end]
                
                if (kp1.confidence > 0.5f && kp2.confidence > 0.5f) {
                    canvas.drawLine(kp1.x, kp1.y, kp2.x, kp2.y, linePaint)
                }
            }
            
            // Draw keypoints
            for ((index, keypoint) in detection.keypoints.withIndex()) {
                if (keypoint.confidence > 0.5f) {
                    val keypointPaint = Paint().apply {
                        style = Paint.Style.FILL
                        color = keypointColors[index % keypointColors.size]
                        isAntiAlias = true
                    }
                    canvas.drawCircle(keypoint.x, keypoint.y, 8f, keypointPaint)
                    
                    // Draw white border around keypoint
                    val borderPaint = Paint().apply {
                        style = Paint.Style.STROKE
                        color = Color.WHITE
                        strokeWidth = 2f
                        isAntiAlias = true
                    }
                    canvas.drawCircle(keypoint.x, keypoint.y, 8f, borderPaint)
                }
            }
        }
        
        return output
    }
    
    fun setConfidenceThreshold(threshold: Float) {
        confidenceThreshold = threshold
    }
    
    fun setIouThreshold(threshold: Float) {
        iouThreshold = threshold
    }
    
    fun close() {
        try {
            interpreter?.close()
            gpuDelegate?.close()
            interpreter = null
            gpuDelegate = null
            Log.d(TAG, "Pose estimation helper closed")
        } catch (e: Exception) {
            Log.e(TAG, "Error closing pose estimation helper", e)
        }
    }
}