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
 * Generic YOLO Detection Helper
 * 
 * A unified, reusable detection helper for all YOLOv8-based fine-tuned models.
 * Eliminates code duplication by using configuration-based approach.
 * 
 * Supports:
 * - Single-class models (output: 1, 5, 8400)
 * - Multi-class models (output: 1, N, 8400 where N = 5 + num_classes)
 * - Custom colors per class
 * - Custom confidence/IoU thresholds
 * - GPU acceleration
 * 
 * Usage:
 * ```kotlin
 * val config = YOLOModelConfig.Grocery()
 * val helper = GenericYOLODetectionHelper(context, config)
 * helper.initialize(modelFile, labelsFile)
 * val detections = helper.runInference(bitmap)
 * ```
 */
class GenericYOLODetectionHelper(
    private val context: Context,
    private val config: YOLOModelConfig
) {
    
    companion object {
        private const val TAG = "GenericYOLOHelper"
        
        // Model constants (same for all YOLOv8 models)
        private const val INPUT_WIDTH = 640
        private const val INPUT_HEIGHT = 640
        private const val INPUT_CHANNELS = 3
        private const val NUM_ANCHORS = 8400
    }
    
    // TensorFlow Lite components
    private var interpreter: Interpreter? = null
    private var gpuDelegate: GpuDelegate? = null
    
    // Image processing
    private lateinit var imageProcessor: ImageProcessor
    
    // Labels
    private var labels = listOf<String>()
    
    // Dynamic output dimensions
    private var numFeatures = 5  // Will be set based on model output shape
    
    // Reusable buffers
    private lateinit var inputBuffer: ByteBuffer
    private lateinit var outputBuffer: Array<Array<FloatArray>>
    
    // Paint for drawing
    private val boxPaint = Paint().apply {
        style = Paint.Style.STROKE
        strokeWidth = config.boxStrokeWidth
        isAntiAlias = true
    }
    
    private val textPaint = Paint().apply {
        color = Color.WHITE
        textSize = config.textSize
        style = Paint.Style.FILL
        isAntiAlias = true
        typeface = Typeface.DEFAULT_BOLD
        if (config.textShadow) {
            setShadowLayer(3f, 0f, 0f, Color.BLACK)
        }
    }
    
    private val textBgPaint = Paint().apply {
        style = Paint.Style.FILL
        alpha = 220
    }
    
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
    )
    
    /**
     * Initialize the model
     */
    fun initialize(modelFile: File, labelsFile: File): Boolean {
        return try {
            Log.d(TAG, "Initializing ${config.modelName} model...")
            
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
            
            Log.d(TAG, "Loaded ${labels.size} label(s)")
            if (labels.size <= 10) {
                Log.d(TAG, "Labels: ${labels.joinToString(", ")}")
            } else {
                Log.d(TAG, "First 10 labels: ${labels.take(10).joinToString(", ")}")
            }
            
            // Configure GPU
            val compatList = CompatibilityList()
            val options = Interpreter.Options().apply {
                if (compatList.isDelegateSupportedOnThisDevice) {
                    try {
                        gpuDelegate = GpuDelegate()
                        addDelegate(gpuDelegate)
                        Log.d(TAG, "✅ GPU delegate enabled")
                    } catch (e: Throwable) {
                        setNumThreads(Runtime.getRuntime().availableProcessors())
                        Log.w(TAG, "GPU delegate fallback to CPU: ${e.message}")
                    }
                } else {
                    setNumThreads(Runtime.getRuntime().availableProcessors())
                    Log.d(TAG, "⚠️ GPU not supported, using CPU")
                }
            }
            
            // Create interpreter
            interpreter = Interpreter(modelByteBuffer, options)
            interpreter!!.allocateTensors()
            
            // Verify shapes and set dynamic dimensions
            val inputShape = interpreter!!.getInputTensor(0).shape()
            val outputShape = interpreter!!.getOutputTensor(0).shape()
            
            Log.d(TAG, "Input shape: [${inputShape.joinToString()}]")
            Log.d(TAG, "Output shape: [${outputShape.joinToString()}]")
            
            // Set number of features from output shape
            numFeatures = outputShape[1]
            Log.d(TAG, "Detected $numFeatures features per anchor")
            
            // Allocate buffers
            val inputBytes = 1 * INPUT_HEIGHT * INPUT_WIDTH * INPUT_CHANNELS * 4
            inputBuffer = ByteBuffer.allocateDirect(inputBytes).apply {
                order(ByteOrder.nativeOrder())
            }
            
            outputBuffer = Array(1) { Array(numFeatures) { FloatArray(NUM_ANCHORS) } }
            
            // Create image processor
            imageProcessor = ImageProcessor.Builder()
                .add(ResizeOp(INPUT_HEIGHT, INPUT_WIDTH, ResizeOp.ResizeMethod.BILINEAR))
                .add(NormalizeOp(0f, 255f))
                .add(CastOp(DataType.FLOAT32))
                .build()
            
            Log.d(TAG, "✅ ${config.modelName} model initialized successfully")
            true
            
        } catch (e: Exception) {
            Log.e(TAG, "❌ Error initializing model", e)
            false
        }
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
            val outputMap = mapOf(0 to outputBuffer)
            currentInterpreter.runForMultipleInputsOutputs(arrayOf(inputBuffer), outputMap)
            
            // Post-process results
            val detections = postProcessOutput(
                output = outputBuffer[0],
                originalWidth = bitmap.width,
                originalHeight = bitmap.height,
                confidenceThreshold = config.confidenceThreshold
            )
            
            // Apply NMS
            val nmsDetections = applyNMS(detections, config.iouThreshold)
            
            Log.d(TAG, "✅ Detected ${nmsDetections.size} object(s)")
            nmsDetections.forEach { det ->
                Log.d(TAG, "  ${det.label}: conf=${String.format("%.3f", det.confidence)}")
            }
            
            return nmsDetections
            
        } catch (e: Exception) {
            Log.e(TAG, "❌ Error during inference", e)
            return emptyList()
        }
    }
    
    /**
     * Post-process the model output
     */
    private fun postProcessOutput(
        output: Array<FloatArray>,
        originalWidth: Int,
        originalHeight: Int,
        confidenceThreshold: Float
    ): List<Detection> {
        val detections = mutableListOf<Detection>()
        
        for (i in 0 until NUM_ANCHORS) {
            // Extract bounding box coordinates
            var xCenter = output[0][i]
            var yCenter = output[1][i]
            var width = output[2][i]
            var height = output[3][i]
            
            // Determine confidence and class
            val (confidence, classId) = when {
                // Single-class model: output[4] is confidence
                numFeatures == 5 -> Pair(output[4][i], 0)
                
                // Multi-class model: output[4] is objectness, rest are class confidences
                else -> {
                    val objectness = output[4][i]
                    var maxClassConf = 0f
                    var maxClassId = 0
                    
                    for (c in 5 until numFeatures) {
                        val classConf = output[c][i]
                        if (classConf > maxClassConf) {
                            maxClassConf = classConf
                            maxClassId = c - 5
                        }
                    }
                    
                    // Final confidence = objectness * class_confidence
                    Pair(objectness * maxClassConf, maxClassId)
                }
            }
            
            if (confidence < confidenceThreshold) continue
            
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
            
            val label = labels.getOrElse(classId) { "class_$classId" }
            
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
     * Apply Non-Maximum Suppression
     */
    private fun applyNMS(detections: List<Detection>, iouThreshold: Float): List<Detection> {
        if (detections.isEmpty()) return emptyList()
        
        val sortedDetections = detections.sortedByDescending { it.confidence }
        val selectedDetections = mutableListOf<Detection>()
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
     * Calculate IoU (Intersection over Union)
     */
    private fun calculateIoU(det1: Detection, det2: Detection): Float {
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
     * Draw detections on canvas with color coding
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
            
            // Get color for this class
            val color = config.getColorForClass(detection.classId, detection.label)
            boxPaint.color = color
            textBgPaint.color = color
            
            // Draw bounding box
            canvas.drawRect(x1, y1, x2, y2, boxPaint)
            
            // Prepare label
            val labelText = if (config.showConfidence) {
                "${detection.label} ${String.format("%.2f", detection.confidence)}"
            } else {
                detection.label
            }
            
            // Measure text
            val textBounds = Rect()
            textPaint.getTextBounds(labelText, 0, labelText.length, textBounds)
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
            canvas.drawText(labelText, textX + 5f, textY - 5f, textPaint)
        }
    }
    
    /**
     * Get detection summary (count per class)
     */
    fun getDetectionSummary(detections: List<Detection>): Map<String, Int> {
        return detections.groupBy { it.label }.mapValues { it.value.size }
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
