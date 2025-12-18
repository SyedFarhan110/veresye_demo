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
 * Helmet Detection Helper
 * 
 * Model Specifications:
 * - Input Shape: (1, 640, 640, 3)
 * - Output Shape: (1, 8, 8400) for 3-class model OR (1, 5, 8400) for single-class
 * 
 * Output Format (Multi-class):
 * - 8 features per detection: [x_center, y_center, width, height, obj_conf, class1_conf, class2_conf, class3_conf]
 * - 8400 anchor points across the image
 * 
 * Classes:
 * - head: Person's head without helmet (unsafe)
 * - helmet: Person wearing a helmet (safe)
 * - person: General person detection
 * 
 * Use Cases:
 * - Construction site safety monitoring
 * - Motorcycle rider safety compliance
 * - Industrial workplace safety
 * 
 * This helper handles:
 * 1. Image preprocessing and resizing to 640x640
 * 2. Model inference for helmet detection
 * 3. Post-processing of detected persons and helmets
 * 4. Non-Maximum Suppression (NMS)
 * 5. Drawing bounding boxes with safety status indicators
 */
class HelmetDetectionHelper(
    private val context: Context
) {
    
    companion object {
        private const val TAG = "HelmetDetectionHelper"
        
        // Model constants
        private const val INPUT_WIDTH = 640
        private const val INPUT_HEIGHT = 640
        private const val INPUT_CHANNELS = 3
        
        // Output dimensions (will be dynamically set)
        private var NUM_FEATURES = 8  // Default for multi-class
        private const val NUM_ANCHORS = 8400
        
        // Detection thresholds
        private const val CONFIDENCE_THRESHOLD = 0.25f
        private const val IOU_THRESHOLD = 0.45f
        
        // Colors for visualization
        private val SAFE_COLOR = Color.parseColor("#00FF00")      // Green - wearing helmet
        private val UNSAFE_COLOR = Color.parseColor("#FF0000")    // Red - no helmet
        private val PERSON_COLOR = Color.parseColor("#FFA500")    // Orange - person
        private const val BOX_STROKE_WIDTH = 5f
        private const val TEXT_SIZE = 42f
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
    }
    
    private val textBgPaint = Paint().apply {
        style = Paint.Style.FILL
        alpha = 220
    }
    
    /**
     * Detection result data class
     */
    data class HelmetDetection(
        val x1: Float,
        val y1: Float,
        val x2: Float,
        val y2: Float,
        val confidence: Float,
        val classId: Int,
        val label: String,
        val isSafe: Boolean  // True if wearing helmet, false if not
    )
    
    /**
     * Safety statistics data class
     */
    data class SafetyStats(
        val totalPersons: Int,
        val withHelmet: Int,
        val withoutHelmet: Int,
        val complianceRate: Float
    )
    
    /**
     * Initialize the model
     */
    fun initialize(modelFile: File, labelsFile: File): Boolean {
        return try {
            Log.d(TAG, "Initializing Helmet Detection model...")
            
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
            
            // Verify shapes and dynamically adjust
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
            
            Log.d(TAG, "✅ Helmet Detection model initialized successfully")
            true
            
        } catch (e: Exception) {
            Log.e(TAG, "❌ Error initializing model", e)
            false
        }
    }
    
    /**
     * Run inference on a bitmap image
     */
    fun runInference(bitmap: Bitmap): List<HelmetDetection> {
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
            
            // Calculate safety statistics
            val stats = calculateSafetyStats(nmsDetections)
            
            Log.d(TAG, "✅ Detected ${nmsDetections.size} person(s)")
            Log.d(TAG, "   Safety Stats - With Helmet: ${stats.withHelmet}, Without: ${stats.withoutHelmet}, Compliance: ${String.format("%.1f%%", stats.complianceRate)}")
            
            nmsDetections.forEach { det ->
                Log.d(TAG, "  ${det.label}: conf=${det.confidence} safe=${det.isSafe}")
            }
            
            return nmsDetections
            
        } catch (e: Exception) {
            Log.e(TAG, "❌ Error during inference", e)
            return emptyList()
        }
    }
    
    /**
     * Post-process the model output - FIXED VERSION
     * Supports both single-class (5 features) and multi-class (8+ features) outputs
     */
    private fun postProcessOutput(
        output: Array<FloatArray>,
        originalWidth: Int,
        originalHeight: Int,
        confidenceThreshold: Float
    ): List<HelmetDetection> {
        val detections = mutableListOf<HelmetDetection>()
        val numClasses = labels.size
        
        // Determine if this is single-class or multi-class output
        val isMultiClass = NUM_FEATURES > 5
        
        Log.d(TAG, "Post-processing: NUM_FEATURES=$NUM_FEATURES, isMultiClass=$isMultiClass, numClasses=$numClasses")
        
        // Create class index map for known classes
        val helmetClassIds = mutableListOf<Int>()
        val headClassIds = mutableListOf<Int>()
        val personClassIds = mutableListOf<Int>()
        
        labels.forEachIndexed { index, label ->
            when {
                label.equals("helmet", ignoreCase = true) -> helmetClassIds.add(index)
                label.equals("head", ignoreCase = true) -> headClassIds.add(index)
                label.equals("person", ignoreCase = true) -> personClassIds.add(index)
            }
        }
        
        Log.d(TAG, "Class mapping - helmet: $helmetClassIds, head: $headClassIds, person: $personClassIds")
        
        var detectionCount = 0
        
        for (i in 0 until NUM_ANCHORS) {
            // Extract bounding box coordinates
            var xCenter = output[0][i]
            var yCenter = output[1][i]
            var width = output[2][i]
            var height = output[3][i]
            val objectness = output[4][i]
            
            // Determine class and confidence
            val classId: Int
            var confidence: Float
            
            if (isMultiClass && numClasses > 1) {
                // Multi-class: find the class with highest confidence
                var maxClassConf = 0f
                var maxClassId = 0
                
                val actualModelClasses = NUM_FEATURES - 5
                val classScores = FloatArray(actualModelClasses)
                
                for (c in 0 until actualModelClasses) {
                    val classConfIndex = 5 + c
                    if (classConfIndex < NUM_FEATURES) {
                        classScores[c] = output[classConfIndex][i]
                        if (classScores[c] > maxClassConf) {
                            maxClassConf = classScores[c]
                            maxClassId = c
                        }
                    }
                }
                
                classId = maxClassId
                
                // FIX: Try multiplying by objectness if class scores seem too uniform
                // Some YOLO models need this, others don't
                confidence = maxClassConf * objectness
                
                // Alternative: Use class confidence directly if objectness is very low
                // Lower the objectness threshold to be more permissive
                if (objectness < 0.3f && maxClassConf > 0.3f) {
                    // Model outputs low objectness but high class scores - use class score directly
                    confidence = maxClassConf
                }
                
                // ALWAYS log first 10 detections to debug class distribution
                if (detectionCount < 10) {
                    val scoresStr = classScores.mapIndexed { idx, score -> 
                        "${labels.getOrElse(idx){"c$idx"}}=${String.format("%.4f", score)}"
                    }.joinToString(", ")
                    Log.d(TAG, "Detection #$detectionCount: obj=${String.format("%.4f", objectness)}, scores=[$scoresStr], picked=${labels.getOrElse(maxClassId){"c$maxClassId"}}, final_conf=${String.format("%.4f", confidence)}")
                    detectionCount++
                }
                
                // Filter by confidence
                if (confidence < confidenceThreshold) continue
                
            } else {
                // Single-class: objectness is the final confidence
                classId = 0
                confidence = objectness
                if (confidence < confidenceThreshold) continue
            }
            
            // Scale coordinates to original image size
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
            
            val label = labels.getOrElse(classId) { "object" }
            
            // FIX: Determine safety based on explicit class ID mapping
            val isSafe = when {
                classId in helmetClassIds -> true  // Wearing helmet = SAFE
                classId in headClassIds -> false   // Bare head = UNSAFE
                classId in personClassIds -> false // General person = treat as UNSAFE (unknown helmet status)
                else -> false // Unknown class = treat as UNSAFE
            }
            
            detections.add(
                HelmetDetection(
                    x1 = x1, y1 = y1, x2 = x2, y2 = y2,
                    confidence = confidence,
                    classId = classId,
                    label = label,
                    isSafe = isSafe
                )
            )
        }
        
        Log.d(TAG, "Post-processing complete: ${detections.size} detections before NMS")
        return detections
    }
    
    /**
     * Alternative approach: Filter to ensure balanced detection of all classes
     * Use this if one class is dominating due to model bias
     */
    private fun postProcessOutputWithBalancing(
        output: Array<FloatArray>,
        originalWidth: Int,
        originalHeight: Int,
        confidenceThreshold: Float
    ): List<HelmetDetection> {
        val allDetections = mutableListOf<HelmetDetection>()
        val numClasses = labels.size
        val isMultiClass = NUM_FEATURES > 5
        
        // Lower threshold for underrepresented classes
        val helmetClassIds = labels.indices.filter { labels[it].equals("helmet", ignoreCase = true) }
        val headClassIds = labels.indices.filter { labels[it].equals("head", ignoreCase = true) }
        
        for (i in 0 until NUM_ANCHORS) {
            var xCenter = output[0][i]
            var yCenter = output[1][i]
            var width = output[2][i]
            var height = output[3][i]
            val objectness = output[4][i]
            
            if (isMultiClass && numClasses > 1) {
                val actualModelClasses = NUM_FEATURES - 5
                
                // Check EACH class separately with potentially different thresholds
                for (c in 0 until actualModelClasses) {
                    val classConfIndex = 5 + c
                    if (classConfIndex >= NUM_FEATURES) continue
                    
                    val classConf = output[classConfIndex][i]
                    
                    // Adaptive threshold: lower for helmet class if it's being under-detected
                    val adaptiveThreshold = when (c) {
                        in helmetClassIds -> confidenceThreshold * 0.7f  // 30% lower threshold
                        in headClassIds -> confidenceThreshold * 1.2f     // 20% higher threshold
                        else -> confidenceThreshold
                    }
                    
                    val finalConf = classConf * objectness
                    if (finalConf < adaptiveThreshold) continue
                    
                    // Scale coordinates
                    val scaledXCenter = if (xCenter <= 1.5f) xCenter * originalWidth else xCenter * originalWidth / INPUT_WIDTH
                    val scaledYCenter = if (yCenter <= 1.5f) yCenter * originalHeight else yCenter * originalHeight / INPUT_HEIGHT
                    val scaledWidth = if (width <= 1.5f) width * originalWidth else width * originalWidth / INPUT_WIDTH
                    val scaledHeight = if (height <= 1.5f) height * originalHeight else height * originalHeight / INPUT_HEIGHT
                    
                    val x1 = (scaledXCenter - scaledWidth / 2f).coerceIn(0f, originalWidth.toFloat())
                    val y1 = (scaledYCenter - scaledHeight / 2f).coerceIn(0f, originalHeight.toFloat())
                    val x2 = (scaledXCenter + scaledWidth / 2f).coerceIn(0f, originalWidth.toFloat())
                    val y2 = (scaledYCenter + scaledHeight / 2f).coerceIn(0f, originalHeight.toFloat())
                    
                    if (x2 <= x1 || y2 <= y1) continue
                    
                    val label = labels.getOrElse(c) { "object" }
                    val isSafe = c in helmetClassIds
                    
                    allDetections.add(
                        HelmetDetection(
                            x1 = x1, y1 = y1, x2 = x2, y2 = y2,
                            confidence = finalConf,
                            classId = c,
                            label = label,
                            isSafe = isSafe
                        )
                    )
                }
            }
        }
        
        Log.d(TAG, "Balanced post-processing: ${allDetections.size} detections before NMS")
        return allDetections
    }
    
    /**
     * Apply Non-Maximum Suppression
     */
    private fun applyNMS(detections: List<HelmetDetection>, iouThreshold: Float): List<HelmetDetection> {
        if (detections.isEmpty()) return emptyList()
        
        val sortedDetections = detections.sortedByDescending { it.confidence }
        val selectedDetections = mutableListOf<HelmetDetection>()
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
    private fun calculateIoU(det1: HelmetDetection, det2: HelmetDetection): Float {
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
     * Calculate safety statistics
     */
    fun calculateSafetyStats(detections: List<HelmetDetection>): SafetyStats {
        val withHelmet = detections.count { it.isSafe }
        val withoutHelmet = detections.count { !it.isSafe }
        val total = detections.size
        val complianceRate = if (total > 0) (withHelmet.toFloat() / total.toFloat()) * 100f else 0f
        
        return SafetyStats(
            totalPersons = total,
            withHelmet = withHelmet,
            withoutHelmet = withoutHelmet,
            complianceRate = complianceRate
        )
    }
    
    /**
     * Draw detections on canvas with safety color coding
     */
    fun drawDetections(
        canvas: Canvas,
        detections: List<HelmetDetection>,
        scaleX: Float = 1f,
        scaleY: Float = 1f
    ) {
        for (detection in detections) {
            val x1 = detection.x1 * scaleX
            val y1 = detection.y1 * scaleY
            val x2 = detection.x2 * scaleX
            val y2 = detection.y2 * scaleY
            
            // Choose color based on safety status
            val color = if (detection.isSafe) SAFE_COLOR else UNSAFE_COLOR
            boxPaint.color = color
            textBgPaint.color = color
            
            // Draw bounding box
            canvas.drawRect(x1, y1, x2, y2, boxPaint)
            
            // Prepare label with safety indicator
            val safetyIcon = if (detection.isSafe) "✓" else "⚠"
            val label = "$safetyIcon ${detection.label} ${String.format("%.2f", detection.confidence)}"
            
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
     * Draw safety statistics overlay
     */
    fun drawSafetyStats(canvas: Canvas, stats: SafetyStats) {
        val statText = "Total: ${stats.totalPersons} | Safe: ${stats.withHelmet} | Unsafe: ${stats.withoutHelmet} | Compliance: ${String.format("%.1f%%", stats.complianceRate)}"
        
        val statPaint = Paint().apply {
            color = Color.WHITE
            textSize = 35f
            style = Paint.Style.FILL
            isAntiAlias = true
            typeface = Typeface.DEFAULT_BOLD
        }
        
        val bgPaint = Paint().apply {
            color = Color.BLACK
            alpha = 180
            style = Paint.Style.FILL
        }
        
        val textBounds = Rect()
        statPaint.getTextBounds(statText, 0, statText.length, textBounds)
        
        val padding = 20f
        val x = padding
        val y = canvas.height - padding - textBounds.height()
        
        canvas.drawRect(
            0f, y - padding,
            textBounds.width() + padding * 2, canvas.height.toFloat(),
            bgPaint
        )
        
        canvas.drawText(statText, x, y, statPaint)
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
