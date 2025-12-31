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
 * Enhanced Object Detection Helper
 * 
 * Key Improvements:
 * 1. Class-wise NMS to detect multiple classes simultaneously
 * 2. Configurable thresholds per class
 * 3. Better color coding for different classes
 * 4. Optimized performance with reusable objects
 * 5. Better logging and debugging
 */
class FaceDetectionHelper(
    private val context: Context
) {
    
    companion object {
        private const val TAG = "FaceDetectionHelper"
        
        // Model constants
        private const val INPUT_WIDTH = 640
        private const val INPUT_HEIGHT = 640
        private const val INPUT_CHANNELS = 3
        
        // Output dimensions (will be dynamically set)
        private var NUM_FEATURES = 5
        private const val NUM_ANCHORS = 8400
        
        // Detection thresholds (can be adjusted per class)
        private const val DEFAULT_CONFIDENCE_THRESHOLD = 0.45f  // Lowered for better recall
        private const val DEFAULT_IOU_THRESHOLD = 0.45f
        
        // Visualization
        private const val BOX_STROKE_WIDTH = 6f
        private const val TEXT_SIZE = 40f
        
        // Predefined colors for different classes
        private val CLASS_COLORS = listOf(
            Color.parseColor("#FF6B6B"), // Red
            Color.parseColor("#4ECDC4"), // Turquoise
            Color.parseColor("#FFD93D"), // Yellow
            Color.parseColor("#95E1D3"), // Mint
            Color.parseColor("#F38181"), // Pink
            Color.parseColor("#AA96DA"), // Purple
            Color.parseColor("#FCBAD3"), // Light Pink
            Color.parseColor("#A8D8EA"), // Light Blue
            Color.parseColor("#FFA07A"), // Light Salmon
            Color.parseColor("#98D8C8")  // Seafoam
        )
    }
    
    // TensorFlow Lite components
    private var interpreter: Interpreter? = null
    private var gpuDelegate: GpuDelegate? = null
    
    // Image processing
    private lateinit var imageProcessor: ImageProcessor
    
    // Labels and thresholds
    private var labels = listOf<String>()
    private val classConfidenceThresholds = mutableMapOf<Int, Float>()
    private val classColors = mutableMapOf<Int, Int>()
    
    // Reusable buffers
    private lateinit var inputBuffer: ByteBuffer
    private lateinit var outputBuffer: Array<Array<FloatArray>>
    
    // Reusable Paint objects
    private val boxPaints = mutableMapOf<Int, Paint>()
    private val textPaint = Paint().apply {
        color = Color.WHITE
        textSize = TEXT_SIZE
        style = Paint.Style.FILL
        isAntiAlias = true
        typeface = Typeface.DEFAULT_BOLD
        setShadowLayer(4f, 2f, 2f, Color.BLACK)
    }
    private val textBgPaints = mutableMapOf<Int, Paint>()
    
    /**
     * Detection result data class
     */
    data class Detection(
        val x1: Float,
        val y1: Float,
        val x2: Float,
        val y2: Float,
        val confidence: Float,
        val classId: Int,
        val label: String
    ) {
        fun area(): Float = (x2 - x1) * (y2 - y1)
    }
    
    /**
     * Initialize the model
     */
    fun initialize(modelFile: File, labelsFile: File): Boolean {
        return try {
            Log.d(TAG, "Initializing Object Detection model...")
            
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
            
            // Initialize class-specific settings
            labels.forEachIndexed { index, label ->
                classConfidenceThresholds[index] = getConfidenceThresholdForClass(label)
                classColors[index] = CLASS_COLORS[index % CLASS_COLORS.size]
                
                // Create Paint objects for each class
                boxPaints[index] = Paint().apply {
                    color = classColors[index]!!
                    style = Paint.Style.STROKE
                    strokeWidth = BOX_STROKE_WIDTH
                    isAntiAlias = true
                }
                
                textBgPaints[index] = Paint().apply {
                    color = classColors[index]!!
                    style = Paint.Style.FILL
                    alpha = 230
                }
            }
            
            // Configure GPU
            val compatList = CompatibilityList()
            val options = Interpreter.Options().apply {
                if (compatList.isDelegateSupportedOnThisDevice) {
                    gpuDelegate = GpuDelegate(compatList.bestOptionsForThisDevice)
                    addDelegate(gpuDelegate)
                    Log.d(TAG, "✅ GPU delegate enabled")
                } else {
                    setNumThreads(Runtime.getRuntime().availableProcessors())
                    Log.d(TAG, "⚠️ GPU not supported, using ${Runtime.getRuntime().availableProcessors()} CPU threads")
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
            
            // Dynamically determine NUM_FEATURES
            NUM_FEATURES = outputShape[1]
            val actualModelClasses = NUM_FEATURES - 5
            
            Log.d(TAG, "NUM_FEATURES: $NUM_FEATURES")
            Log.d(TAG, "Model classes: $actualModelClasses, Label classes: ${labels.size}")
            
            if (actualModelClasses != labels.size && labels.size > 1) {
                Log.w(TAG, "⚠️ Model/Label mismatch: using first $actualModelClasses labels")
            }
            
            // Allocate buffers
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
            
            Log.d(TAG, "✅ Model initialized successfully")
            true
            
        } catch (e: Exception) {
            Log.e(TAG, "❌ Error initializing model", e)
            false
        }
    }
    
    /**
     * Get confidence threshold for specific class
     * Can be customized per class for better results
     */
    private fun getConfidenceThresholdForClass(label: String): Float {
        return when (label.lowercase()) {
            "fire" -> 0.40f      // Lower threshold for critical detection
            "smoke" -> 0.35f     // Even lower for smoke (harder to detect)
            "person", "face" -> 0.50f
            else -> DEFAULT_CONFIDENCE_THRESHOLD
        }
    }
    
    /**
     * Set custom confidence threshold for a specific class
     */
    fun setClassConfidenceThreshold(classId: Int, threshold: Float) {
        classConfidenceThresholds[classId] = threshold.coerceIn(0f, 1f)
        Log.d(TAG, "Set confidence threshold for class $classId: $threshold")
    }
    
    /**
     * Run inference on a bitmap image
     */
    fun runInference(bitmap: Bitmap): List<Detection> {
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
            val startTime = System.currentTimeMillis()
            val outputMap = mapOf(0 to outputBuffer)
            currentInterpreter.runForMultipleInputsOutputs(arrayOf(inputBuffer), outputMap)
            val inferenceTime = System.currentTimeMillis() - startTime
            
            // Post-process results
            val detections = postProcessOutput(
                output = outputBuffer[0],
                originalWidth = bitmap.width,
                originalHeight = bitmap.height
            )
            
            // Apply class-wise NMS
            val nmsDetections = applyClassWiseNMS(detections, DEFAULT_IOU_THRESHOLD)
            
            Log.d(TAG, "✅ Inference: ${inferenceTime}ms | Detections: ${nmsDetections.size}")
            
            // Log detections by class
            val detectionsByClass = nmsDetections.groupBy { it.label }
            detectionsByClass.forEach { (label, dets) ->
                Log.d(TAG, "  $label: ${dets.size} detection(s)")
            }
            
            return nmsDetections
            
        } catch (e: Exception) {
            Log.e(TAG, "❌ Error during inference", e)
            return emptyList()
        }
    }
    
    /**
     * Enhanced post-processing with class-specific thresholds
     */
    private fun postProcessOutput(
        output: Array<FloatArray>,
        originalWidth: Int,
        originalHeight: Int
    ): List<Detection> {
        val detections = mutableListOf<Detection>()
        val numClasses = labels.size
        val isMultiClass = NUM_FEATURES > 5
        
        for (i in 0 until NUM_ANCHORS) {
            // Extract bounding box
            var xCenter = output[0][i]
            var yCenter = output[1][i]
            var width = output[2][i]
            var height = output[3][i]
            
            // Determine class and confidence
            val classId: Int
            val confidence: Float
            
            when {
                NUM_FEATURES == 6 && numClasses == 2 -> {
                    // Binary classification with explicit class scores
                    val class0Conf = output[4][i]
                    val class1Conf = output[5][i]
                    
                    if (class0Conf > class1Conf) {
                        classId = 0
                        confidence = class0Conf
                    } else {
                        classId = 1
                        confidence = class1Conf
                    }
                }
                
                isMultiClass && numClasses > 1 -> {
                    // Multi-class: find max confidence class
                    var maxClassConf = 0f
                    var maxClassId = 0
                    val actualModelClasses = NUM_FEATURES - 5
                    
                    for (c in 0 until actualModelClasses) {
                        val classConfIndex = 5 + c
                        if (classConfIndex < NUM_FEATURES) {
                            val classConf = output[classConfIndex][i]
                            if (classConf > maxClassConf) {
                                maxClassConf = classConf
                                maxClassId = c
                            }
                        }
                    }
                    
                    classId = maxClassId
                    confidence = maxClassConf
                }
                
                else -> {
                    // Single-class
                    classId = 0
                    confidence = output[4][i]
                }
            }
            
            // Use class-specific threshold
            val threshold = classConfidenceThresholds[classId] ?: DEFAULT_CONFIDENCE_THRESHOLD
            if (confidence < threshold) continue
            
            // Scale coordinates
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
            
            // Validate box
            if (x2 <= x1 || y2 <= y1) continue
            
            val label = labels.getOrElse(classId) { 
                if (classId == 0) "object" else "class_$classId" 
            }
            
            detections.add(
                Detection(
                    x1 = x1, y1 = y1, x2 = x2, y2 = y2,
                    confidence = confidence,
                    classId = classId,
                    label = label
                )
            )
        }
        
        return detections
    }
    
    /**
     * Apply NMS separately for each class
     * This allows detecting multiple different classes in the same location
     */
    private fun applyClassWiseNMS(detections: List<Detection>, iouThreshold: Float): List<Detection> {
        if (detections.isEmpty()) return emptyList()
        
        val selectedDetections = mutableListOf<Detection>()
        
        // Group by class and apply NMS to each group
        val detectionsByClass = detections.groupBy { it.classId }
        
        for ((classId, classDetections) in detectionsByClass) {
            // Sort by confidence (highest first)
            val sortedDetections = classDetections.sortedByDescending { it.confidence }
            val suppressed = BooleanArray(sortedDetections.size) { false }
            
            for (i in sortedDetections.indices) {
                if (suppressed[i]) continue
                
                val currentDet = sortedDetections[i]
                selectedDetections.add(currentDet)
                
                // Suppress overlapping detections of the SAME class
                for (j in i + 1 until sortedDetections.size) {
                    if (suppressed[j]) continue
                    
                    val iou = calculateIoU(currentDet, sortedDetections[j])
                    if (iou > iouThreshold) {
                        suppressed[j] = true
                    }
                }
            }
        }
        
        return selectedDetections
    }
    
    /**
     * Calculate Intersection over Union
     */
    private fun calculateIoU(det1: Detection, det2: Detection): Float {
        val x1 = max(det1.x1, det2.x1)
        val y1 = max(det1.y1, det2.y1)
        val x2 = min(det1.x2, det2.x2)
        val y2 = min(det1.y2, det2.y2)
        
        val intersectionArea = max(0f, x2 - x1) * max(0f, y2 - y1)
        val area1 = det1.area()
        val area2 = det2.area()
        val unionArea = area1 + area2 - intersectionArea
        
        return if (unionArea > 0f) intersectionArea / unionArea else 0f
    }
    
    /**
     * Draw detections with class-specific colors
     */
    fun drawDetections(
        canvas: Canvas,
        detections: List<Detection>,
        scaleX: Float = 1f,
        scaleY: Float = 1f
    ) {
        for (detection in detections) {
            val x1 = detection.x1 * scaleX
            val y1 = detection.y1 * scaleY
            val x2 = detection.x2 * scaleX
            val y2 = detection.y2 * scaleY
            
            // Get class-specific paints
            val boxPaint = boxPaints[detection.classId] ?: boxPaints[0]!!
            val bgPaint = textBgPaints[detection.classId] ?: textBgPaints[0]!!
            
            // Draw bounding box
            canvas.drawRect(x1, y1, x2, y2, boxPaint)
            
            // Prepare label
            val label = "${detection.label} ${String.format("%.0f%%", detection.confidence * 100)}"
            
            // Measure text
            val textBounds = Rect()
            textPaint.getTextBounds(label, 0, label.length, textBounds)
            val textWidth = textBounds.width()
            val textHeight = textBounds.height()
            
            // Position label (above box if space, otherwise inside)
            val textX = x1
            val textY = if (y1 > textHeight + 20f) {
                y1 - 10f
            } else {
                y1 + textHeight + 20f
            }
            
            // Draw background
            canvas.drawRoundRect(
                textX, textY - textHeight - 10f,
                textX + textWidth + 20f, textY + 5f,
                8f, 8f, bgPaint
            )
            
            // Draw text
            canvas.drawText(label, textX + 10f, textY - 5f, textPaint)
        }
    }
    
    /**
     * Get detection statistics
     */
    fun getDetectionStats(detections: List<Detection>): Map<String, Int> {
        return detections.groupBy { it.label }
            .mapValues { it.value.size }
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
            boxPaints.clear()
            textBgPaints.clear()
            Log.d(TAG, "Resources released")
        } catch (e: Exception) {
            Log.e(TAG, "Error releasing resources", e)
        }
    }
}