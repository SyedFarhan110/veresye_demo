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

/**
 * Fire and Smoke Detection Helper
 * 
 * Model Specifications:
 * - Input Shape: (1, 640, 640, 3)
 * - Output Shape: (1, 7, 8400) for 2-class model OR (1, 5, 8400) for single-class
 * 
 * Output Format (Multi-class):
 * - 7 features per detection: [x_center, y_center, width, height, obj_conf, fire_conf, smoke_conf]
 * - 8400 anchor points across the image
 * 
 * Classes:
 * - Fire: Active fire/flame detection
 * - Smoke: Smoke detection
 * 
 * This helper handles:
 * 1. Image preprocessing and resizing to 640x640
 * 2. Model inference for fire and smoke detection
 * 3. Post-processing of detected fire/smoke
 * 4. Non-Maximum Suppression (NMS)
 * 5. Drawing bounding boxes with alert colors
 */
class FireSmokeDetectionHelper(
    private val context: Context
) {
    
    companion object {
        private const val TAG = "FireSmokeHelper"
        
        // Model constants
        private const val INPUT_WIDTH = 640
        private const val INPUT_HEIGHT = 640
        private const val INPUT_CHANNELS = 3
        
        // Output dimensions (will be dynamically set)
        private var NUM_FEATURES = 7  // Default for multi-class
        private const val NUM_ANCHORS = 8400
        
        // Detection thresholds
        private const val CONFIDENCE_THRESHOLD = 0.45f
        private const val IOU_THRESHOLD = 0.45f
        
        // Colors for visualization
        private val FIRE_COLOR = Color.parseColor("#FF4500")     // Orange Red
        private val SMOKE_COLOR = Color.parseColor("#808080")    // Gray
        private const val BOX_STROKE_WIDTH = 6f
        private const val TEXT_SIZE = 45f
    }
    
    // TensorFlow Lite components
    private var interpreter: Interpreter? = null
    private var gpuDelegate: GpuDelegate? = null
    
    // Image processing
    private lateinit var imageProcessor: ImageProcessor
    
    // Labels
    private var labels = listOf<String>()
    
    // Reusable buffers
    private lateinit var inputBuffer: ByteBuffer
    private lateinit var outputBuffer: Array<Array<FloatArray>>
    
    // Paint for drawing
    private val boxPaint = Paint().apply {
        style = Paint.Style.STROKE
        strokeWidth = BOX_STROKE_WIDTH
        isAntiAlias = true
    }
    
    private val textPaint = Paint().apply {
        color = Color.WHITE
        textSize = TEXT_SIZE
        style = Paint.Style.FILL
        isAntiAlias = true
        typeface = Typeface.DEFAULT_BOLD
        setShadowLayer(3f, 0f, 0f, Color.BLACK)
    }
    
    private val textBgPaint = Paint().apply {
        style = Paint.Style.FILL
        alpha = 230
    }
    
    /**
     * Detection result data class
     */
    data class FireSmokeDetection(
        val x1: Float,
        val y1: Float,
        val x2: Float,
        val y2: Float,
        val confidence: Float,
        val classId: Int,
        val label: String,
        val isFire: Boolean  // Helper flag to identify fire vs smoke
    )
    
    /**
     * Initialize the model
     */
    fun initialize(modelFile: File, labelsFile: File): Boolean {
        return try {
            Log.d(TAG, "Initializing Fire/Smoke Detection model...")
            
            // Load model
            val modelInputStream = FileInputStream(modelFile)
            val modelFileChannel = modelInputStream.channel
            val modelByteBuffer = modelFileChannel.map(
                java.nio.channels.FileChannel.MapMode.READ_ONLY,
                0,
                modelFile.length()
            )
            modelInputStream.close()
            
            // Load labels
            val labelsInputStream = FileInputStream(labelsFile)
            val labelsReader = BufferedReader(InputStreamReader(labelsInputStream))
            labels = labelsReader.readLines().filter { it.isNotBlank() }
            labelsReader.close()
            
            Log.d(TAG, "Loaded ${labels.size} label(s): ${labels.joinToString(", ")}")
            
            // Configure GPU
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
                    Log.d(TAG, "GPU not supported, using CPU")
                }
            }
            
            // Create interpreter
            interpreter = Interpreter(modelByteBuffer, options)
            interpreter!!.allocateTensors()
            
            // Verify shapes
            val inputShape = interpreter!!.getInputTensor(0).shape()
            val outputShape = interpreter!!.getOutputTensor(0).shape()
            
            Log.d(TAG, "Input shape: [${inputShape.joinToString()}]")
            Log.d(TAG, "Output shape: [${outputShape.joinToString()}]")
            
            // Dynamically determine NUM_FEATURES from output shape
            NUM_FEATURES = outputShape[1]
            
            // Calculate actual number of classes from model output
            val actualModelClasses = NUM_FEATURES - 5  // Subtract 4 bbox coords + 1 objectness
            Log.d(TAG, "Detected NUM_FEATURES: $NUM_FEATURES")
            Log.d(TAG, "Label file has ${labels.size} classes, model outputs $actualModelClasses class scores")
            
            if (actualModelClasses != labels.size) {
                Log.w(TAG, "⚠️ WARNING: Model outputs $actualModelClasses classes but labels file has ${labels.size} classes!")
                Log.w(TAG, "⚠️ Will use first $actualModelClasses labels from file")
            }
            
            // Allocate buffers with dynamic size
            val inputBytes = 1 * INPUT_HEIGHT * INPUT_WIDTH * INPUT_CHANNELS * 4
            inputBuffer = ByteBuffer.allocateDirect(inputBytes).apply {
                order(ByteOrder.nativeOrder())
            }
            
            outputBuffer = Array(1) { Array(NUM_FEATURES) { FloatArray(NUM_ANCHORS) } }
            
            // Create image processor
            imageProcessor = ImageProcessor.Builder()
                .add(ResizeOp(INPUT_HEIGHT, INPUT_WIDTH, ResizeOp.ResizeMethod.BILINEAR))
                .add(NormalizeOp(0f, 255f))
                .add(CastOp(DataType.FLOAT32))
                .build()
            
            Log.d(TAG, "✅ Fire/Smoke Detection model initialized successfully")
            true
            
        } catch (e: Exception) {
            Log.e(TAG, "❌ Error initializing model", e)
            false
        }
    }
    
    /**
     * Run inference on a bitmap image
     */
    fun runInference(bitmap: Bitmap): List<FireSmokeDetection> {
        val currentInterpreter = interpreter ?: run {
            Log.e(TAG, "Interpreter not initialized")
            return emptyList()
        }
        
        try {
            // Preprocess image
            val tensorImage = TensorImage(DataType.FLOAT32)
            tensorImage.load(bitmap)
            val processedImage = imageProcessor.process(tensorImage)
            
            // Prepare input
            inputBuffer.clear()
            inputBuffer.put(processedImage.buffer)
            inputBuffer.rewind()
            
            // Run inference
            val outputMap = mapOf(0 to outputBuffer)
            currentInterpreter.runForMultipleInputsOutputs(arrayOf(inputBuffer), outputMap)
            
            // Post-process results
            val detections = postProcessOutput(
                output = outputBuffer[0],
                originalWidth = bitmap.width,
                originalHeight = bitmap.height,
                confidenceThreshold = CONFIDENCE_THRESHOLD
            )
            
            // Apply NMS
            val nmsDetections = applyNMS(detections, IOU_THRESHOLD)
            
            Log.d(TAG, "✅ Detected ${nmsDetections.size} fire/smoke instance(s)")
            nmsDetections.forEach { det ->
                Log.d(TAG, "  ${det.label}: conf=${det.confidence}")
            }
            
            return nmsDetections
            
        } catch (e: Exception) {
            Log.e(TAG, "❌ Error during inference", e)
            return emptyList()
        }
    }
    
    /**
     * Post-process the model output
     * Supports both single-class (5 features) and multi-class (7+ features) outputs
     */
    private fun postProcessOutput(
        output: Array<FloatArray>,
        originalWidth: Int,
        originalHeight: Int,
        confidenceThreshold: Float
    ): List<FireSmokeDetection> {
        val detections = mutableListOf<FireSmokeDetection>()
        val numClasses = labels.size
        
        // Determine if this is single-class or multi-class output
        val isMultiClass = NUM_FEATURES > 5
        
        Log.d(TAG, "Post-processing: NUM_FEATURES=$NUM_FEATURES, isMultiClass=$isMultiClass, numClasses=$numClasses")
        
        var maxObjectnessFound = 0f
        var sampleIndex = -1
        
        for (i in 0 until NUM_ANCHORS) {
            // Extract bounding box coordinates
            var xCenter = output[0][i]
            var yCenter = output[1][i]
            var width = output[2][i]
            var height = output[3][i]
            
            // Determine class and confidence based on NUM_FEATURES
            val classId: Int
            val confidence: Float
            
            if (NUM_FEATURES == 6 && numClasses == 2) {
                // Special case: 6 features with 2 classes means indices 4 and 5 are class scores directly
                val class0Conf = output[4][i]
                val class1Conf = output[5][i]
                
                if (class0Conf > class1Conf) {
                    classId = 0
                    confidence = class0Conf
                } else {
                    classId = 1
                    confidence = class1Conf
                }
                
                // Track for debugging
                if (confidence > maxObjectnessFound) {
                    maxObjectnessFound = confidence
                    sampleIndex = i
                }
                
                if (confidence < confidenceThreshold) continue
                
            } else if (isMultiClass && numClasses > 1) {
                // Multi-class: find the class with highest confidence
                var maxClassConf = 0f
                var maxClassId = 0
                
                // Use actual model output classes, not label file count
                val actualModelClasses = NUM_FEATURES - 5
                
                for (c in 0 until actualModelClasses) {
                    val classConfIndex = 5 + c  // Class confidences start at index 5
                    if (classConfIndex < NUM_FEATURES) {
                        val classConf = output[classConfIndex][i]
                        if (classConf > maxClassConf) {
                            maxClassConf = classConf
                            maxClassId = c
                        }
                    }
                }
                
                classId = maxClassId
                // Use class confidence directly (not multiplied by objectness)
                // since these models output low objectness but high class scores
                confidence = maxClassConf
                
                // Filter by class confidence
                if (confidence < confidenceThreshold) continue
                
            } else {
                // Single-class: feature 4 is the final confidence
                classId = 0
                confidence = output[4][i]
                
                // Track for debugging
                if (confidence > maxObjectnessFound) {
                    maxObjectnessFound = confidence
                    sampleIndex = i
                }
                
                if (confidence < confidenceThreshold) continue
            }
            
            // Check if normalized
            val isNormalized = (xCenter <= 1.5f && yCenter <= 1.5f)
            
            if (isNormalized) {
                xCenter *= originalWidth.toFloat()
                yCenter *= originalHeight.toFloat()
                width *= originalWidth.toFloat()
                height *= originalHeight.toFloat()
            } else {
                val scaleX = originalWidth.toFloat() / INPUT_WIDTH.toFloat()
                val scaleY = originalHeight.toFloat() / INPUT_HEIGHT.toFloat()
                xCenter *= scaleX
                yCenter *= scaleY
                width *= scaleX
                height *= scaleY
            }
            
            // Convert to corners
            val x1 = (xCenter - width / 2f).coerceIn(0f, originalWidth.toFloat())
            val y1 = (yCenter - height / 2f).coerceIn(0f, originalHeight.toFloat())
            val x2 = (xCenter + width / 2f).coerceIn(0f, originalWidth.toFloat())
            val y2 = (yCenter + height / 2f).coerceIn(0f, originalHeight.toFloat())
            
            if (x2 <= x1 || y2 <= y1) continue
            
            val label = labels.getOrElse(classId) { "unknown" }
            val isFire = label.contains("fire", ignoreCase = true) || 
                        label.contains("flame", ignoreCase = true)
            
            detections.add(
                FireSmokeDetection(
                    x1 = x1, y1 = y1, x2 = x2, y2 = y2,
                    confidence = confidence,
                    classId = classId,
                    label = label,
                    isFire = isFire
                )
            )
        }
        
        // Log sample detection for debugging
        if (sampleIndex >= 0) {
            val sampleValues = StringBuilder()
            for (f in 0 until NUM_FEATURES) {
                sampleValues.append("[$f]=${output[f][sampleIndex]} ")
            }
            Log.d(TAG, "Sample anchor $sampleIndex (max objectness=$maxObjectnessFound): $sampleValues")
        }
        
        Log.d(TAG, "Post-processing: Found ${detections.size} detections before NMS")
        return detections
    }
    
    /**
     * Apply Non-Maximum Suppression
     */
    private fun applyNMS(detections: List<FireSmokeDetection>, iouThreshold: Float): List<FireSmokeDetection> {
        if (detections.isEmpty()) return emptyList()
        
        val sortedDetections = detections.sortedByDescending { it.confidence }
        val selectedDetections = mutableListOf<FireSmokeDetection>()
        val suppressed = BooleanArray(sortedDetections.size) { false }
        
        for (i in sortedDetections.indices) {
            if (suppressed[i]) continue
            selectedDetections.add(sortedDetections[i])
            
            for (j in i + 1 until sortedDetections.size) {
                if (suppressed[j]) continue
                val iou = calculateIoU(sortedDetections[i], sortedDetections[j])
                if (iou > iouThreshold) {
                    suppressed[j] = true
                }
            }
        }
        
        return selectedDetections
    }
    
    /**
     * Calculate IoU
     */
    private fun calculateIoU(det1: FireSmokeDetection, det2: FireSmokeDetection): Float {
        val x1 = max(det1.x1, det2.x1)
        val y1 = max(det1.y1, det2.y1)
        val x2 = min(det1.x2, det2.x2)
        val y2 = min(det1.y2, det2.y2)
        
        val intersectionArea = max(0f, x2 - x1) * max(0f, y2 - y1)
        val area1 = (det1.x2 - det1.x1) * (det1.y2 - det1.y1)
        val area2 = (det2.x2 - det2.x1) * (det2.y2 - det2.y1)
        val unionArea = area1 + area2 - intersectionArea
        
        return if (unionArea > 0f) intersectionArea / unionArea else 0f
    }
    
    /**
     * Draw detections on canvas with alert styling
     */
    fun drawDetections(
        canvas: Canvas,
        detections: List<FireSmokeDetection>,
        scaleX: Float = 1f,
        scaleY: Float = 1f
    ) {
        for (detection in detections) {
            val x1 = detection.x1 * scaleX
            val y1 = detection.y1 * scaleY
            val x2 = detection.x2 * scaleX
            val y2 = detection.y2 * scaleY
            
            // Choose color based on type
            val color = if (detection.isFire) FIRE_COLOR else SMOKE_COLOR
            boxPaint.color = color
            textBgPaint.color = color
            
            // Draw bounding box
            canvas.drawRect(x1, y1, x2, y2, boxPaint)
            
            // Prepare label with WARNING prefix for fire
            val prefix = if (detection.isFire) "⚠️ " else ""
            val label = "$prefix${detection.label.uppercase()} ${String.format("%.2f", detection.confidence)}"
            
            // Measure text
            val textBounds = Rect()
            textPaint.getTextBounds(label, 0, label.length, textBounds)
            val textWidth = textBounds.width()
            val textHeight = textBounds.height()
            
            // Draw text background
            val textX = x1
            val textY = max(y1 - 10f, textHeight.toFloat())
            canvas.drawRect(
                textX, textY - textHeight - 10f,
                textX + textWidth + 10f, textY,
                textBgPaint
            )
            
            // Draw text
            canvas.drawText(label, textX + 5f, textY - 5f, textPaint)
        }
    }
    
    /**
     * Release resources
     */
    fun close() {
        try {
            interpreter?.close()
            interpreter = null
            gpuDelegate?.close()
            gpuDelegate = null
            Log.d(TAG, "Resources released")
        } catch (e: Exception) {
            Log.e(TAG, "Error releasing resources", e)
        }
    }
}
