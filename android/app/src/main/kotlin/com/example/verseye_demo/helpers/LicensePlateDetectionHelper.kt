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
import kotlin.math.exp

/**
 * License Plate Detection Helper
 * 
 * Model Specifications:
 * - Input Shape: (1, 640, 640, 3)
 * - Output Shape: (1, 5, 8400)
 * 
 * Output Format:
 * - 5 features per detection: [x_center, y_center, width, height, confidence]
 * - 8400 anchor points across the image
 * 
 * This helper handles the complete inference pipeline:
 * 1. Image preprocessing and resizing to 640x640
 * 2. Model inference
 * 3. Post-processing of detections
 * 4. Non-Maximum Suppression (NMS)
 * 5. Drawing bounding boxes on canvas
 */
class LicensePlateDetectionHelper(
    private val context: Context
) {
    
    companion object {
        private const val TAG = "LPDHelper"
        
        // Model constants
        private const val INPUT_WIDTH = 640
        private const val INPUT_HEIGHT = 640
        private const val INPUT_CHANNELS = 3
        
        // Output dimensions
        private const val NUM_FEATURES = 5  // [x_center, y_center, width, height, confidence]
        private const val NUM_ANCHORS = 8400
        
        // Detection thresholds
        private const val CONFIDENCE_THRESHOLD = 0.5f
        private const val IOU_THRESHOLD = 0.45f
        
        // Colors for visualization
        private val DETECTION_COLOR = Color.parseColor("#00FF00") // Green
        private const val BOX_STROKE_WIDTH = 4f
        private const val TEXT_SIZE = 40f
    }
    
    // TensorFlow Lite components
    private var interpreter: Interpreter? = null
    private var gpuDelegate: GpuDelegate? = null
    
    // Image processing
    private lateinit var imageProcessor: ImageProcessor
    
    // Labels (typically just "license_plate")
    private var labels = listOf<String>()
    
    // Reusable buffers for efficiency
    private lateinit var inputBuffer: ByteBuffer
    private lateinit var outputBuffer: Array<Array<FloatArray>>
    
    // Paint for drawing
    private val boxPaint = Paint().apply {
        color = DETECTION_COLOR
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
        color = DETECTION_COLOR
        style = Paint.Style.FILL
        alpha = 200
    }
    
    /**
     * Detection result data class
     */
    data class LPDetection(
        val x1: Float,      // Top-left x coordinate
        val y1: Float,      // Top-left y coordinate
        val x2: Float,      // Bottom-right x coordinate
        val y2: Float,      // Bottom-right y coordinate
        val confidence: Float,
        val classId: Int,
        val label: String
    )
    
    /**
     * Initialize the model and allocate resources
     * 
     * @param modelFile The TFLite model file
     * @param labelsFile The labels text file
     * @return true if initialization successful, false otherwise
     */
    fun initialize(modelFile: File, labelsFile: File): Boolean {
        return try {
            Log.d(TAG, "Initializing License Plate Detection model...")
            
            // Load model file into memory
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
            
            // Configure GPU acceleration if available
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
            
            // Verify model shapes
            val inputShape = interpreter!!.getInputTensor(0).shape()
            val outputShape = interpreter!!.getOutputTensor(0).shape()
            
            Log.d(TAG, "Input shape: [${inputShape.joinToString()}]")
            Log.d(TAG, "Output shape: [${outputShape.joinToString()}]")
            
            // Validate shapes
            require(inputShape.contentEquals(intArrayOf(1, INPUT_HEIGHT, INPUT_WIDTH, INPUT_CHANNELS))) {
                "Expected input shape [1, $INPUT_HEIGHT, $INPUT_WIDTH, $INPUT_CHANNELS], got ${inputShape.contentToString()}"
            }
            
            require(outputShape.contentEquals(intArrayOf(1, NUM_FEATURES, NUM_ANCHORS))) {
                "Expected output shape [1, $NUM_FEATURES, $NUM_ANCHORS], got ${outputShape.contentToString()}"
            }
            
            // Allocate input buffer
            val inputBytes = 1 * INPUT_HEIGHT * INPUT_WIDTH * INPUT_CHANNELS * 4  // 4 bytes per float
            inputBuffer = ByteBuffer.allocateDirect(inputBytes).apply {
                order(ByteOrder.nativeOrder())
            }
            
            // Allocate output buffer: [1, 5, 8400]
            outputBuffer = Array(1) { Array(NUM_FEATURES) { FloatArray(NUM_ANCHORS) } }
            
            // Create image processor
            imageProcessor = ImageProcessor.Builder()
                .add(ResizeOp(INPUT_HEIGHT, INPUT_WIDTH, ResizeOp.ResizeMethod.BILINEAR))
                .add(NormalizeOp(0f, 255f))  // Normalize to [0, 1]
                .add(CastOp(DataType.FLOAT32))
                .build()
            
            Log.d(TAG, "✅ License Plate Detection model initialized successfully")
            true
            
        } catch (e: Exception) {
            Log.e(TAG, "❌ Error initializing model", e)
            false
        }
    }
    
    /**
     * Run inference on a bitmap image
     * 
     * @param bitmap Input image
     * @return List of detected license plates
     */
    fun runInference(bitmap: Bitmap): List<LPDetection> {
        val currentInterpreter = interpreter ?: run {
            Log.e(TAG, "Interpreter not initialized")
            return emptyList()
        }
        
        try {
            Log.d(TAG, "Running inference on image: ${bitmap.width}x${bitmap.height}")
            
            // Preprocess image
            val tensorImage = TensorImage(DataType.FLOAT32)
            tensorImage.load(bitmap)
            val processedImage = imageProcessor.process(tensorImage)
            
            // Prepare input buffer
            inputBuffer.clear()
            inputBuffer.put(processedImage.buffer)
            inputBuffer.rewind()
            
            Log.d(TAG, "Input buffer prepared, running model...")
            
            // Run inference
            val outputMap = mapOf(0 to outputBuffer)
            currentInterpreter.runForMultipleInputsOutputs(arrayOf(inputBuffer), outputMap)
            
            Log.d(TAG, "Model inference complete, processing outputs...")
            
            // Log sample output values for debugging
            Log.d(TAG, "Sample output values - First 5 anchors:")
            for (i in 0 until minOf(5, NUM_ANCHORS)) {
                Log.d(TAG, "  Anchor $i: xc=${outputBuffer[0][0][i]}, yc=${outputBuffer[0][1][i]}, " +
                        "w=${outputBuffer[0][2][i]}, h=${outputBuffer[0][3][i]}, conf=${outputBuffer[0][4][i]}")
            }
            
            // Post-process results
            val detections = postProcessOutput(
                output = outputBuffer[0],
                originalWidth = bitmap.width,
                originalHeight = bitmap.height,
                confidenceThreshold = CONFIDENCE_THRESHOLD
            )
            
            // Apply Non-Maximum Suppression
            val nmsDetections = applyNMS(detections, IOU_THRESHOLD)
            
            Log.d(TAG, "✅ Detected ${nmsDetections.size} license plate(s)")
            if (nmsDetections.isNotEmpty()) {
                nmsDetections.forEach { det ->
                    Log.d(TAG, "  Plate: [${det.x1.toInt()},${det.y1.toInt()}] to [${det.x2.toInt()},${det.y2.toInt()}] conf=${det.confidence}")
                }
            }
            
            return nmsDetections
            
        } catch (e: Exception) {
            Log.e(TAG, "❌ Error during inference", e)
            e.printStackTrace()
            return emptyList()
        }
    }
    
    /**
     * Post-process the model output
     * 
     * Output format: [1, 5, 8400]
     * - Feature 0: x_center (normalized 0-1 OR in pixels 0-640)
     * - Feature 1: y_center (normalized 0-1 OR in pixels 0-640)
     * - Feature 2: width (normalized 0-1 OR in pixels 0-640)
     * - Feature 3: height (normalized 0-1 OR in pixels 0-640)
     * - Feature 4: confidence score
     */
    private fun postProcessOutput(
        output: Array<FloatArray>,
        originalWidth: Int,
        originalHeight: Int,
        confidenceThreshold: Float
    ): List<LPDetection> {
        val detections = mutableListOf<LPDetection>()
        
        // Iterate through all anchor points
        for (i in 0 until NUM_ANCHORS) {
            val confidence = output[4][i]  // Confidence is in index 4
            
            // Filter by confidence threshold
            if (confidence < confidenceThreshold) continue
            
            // Extract box coordinates
            var xCenter = output[0][i]
            var yCenter = output[1][i]
            var width = output[2][i]
            var height = output[3][i]
            
            // Check if coordinates are normalized (0-1) or in pixel space (0-640)
            // If max coordinate is <= 1.5, assume normalized; otherwise pixel space
            val isNormalized = (xCenter <= 1.5f && yCenter <= 1.5f)
            
            if (isNormalized) {
                // Coordinates are normalized (0-1), convert to pixels relative to original image
                xCenter *= originalWidth.toFloat()
                yCenter *= originalHeight.toFloat()
                width *= originalWidth.toFloat()
                height *= originalHeight.toFloat()
            } else {
                // Coordinates are in pixel space relative to 640x640, scale to original image
                val scaleX = originalWidth.toFloat() / INPUT_WIDTH.toFloat()
                val scaleY = originalHeight.toFloat() / INPUT_HEIGHT.toFloat()
                xCenter *= scaleX
                yCenter *= scaleY
                width *= scaleX
                height *= scaleY
            }
            
            // Convert from center format to corner format
            val x1 = xCenter - width / 2f
            val y1 = yCenter - height / 2f
            val x2 = xCenter + width / 2f
            val y2 = yCenter + height / 2f
            
            // Clamp coordinates to image bounds
            val clampedX1 = x1.coerceIn(0f, originalWidth.toFloat())
            val clampedY1 = y1.coerceIn(0f, originalHeight.toFloat())
            val clampedX2 = x2.coerceIn(0f, originalWidth.toFloat())
            val clampedY2 = y2.coerceIn(0f, originalHeight.toFloat())
            
            // Skip invalid boxes
            if (clampedX2 <= clampedX1 || clampedY2 <= clampedY1) {
                Log.d(TAG, "  Skipping invalid box: [$x1,$y1] to [$x2,$y2]")
                continue
            }
            
            Log.d(TAG, "  Valid detection at anchor $i: [${clampedX1.toInt()},${clampedY1.toInt()}] to [${clampedX2.toInt()},${clampedY2.toInt()}] conf=$confidence")
            
            // Add detection
            detections.add(
                LPDetection(
                    x1 = clampedX1,
                    y1 = clampedY1,
                    x2 = clampedX2,
                    y2 = clampedY2,
                    confidence = confidence,
                    classId = 0,  // Single class: license_plate
                    label = labels.getOrElse(0) { "license_plate" }
                )
            )
        }
        
        Log.d(TAG, "Post-processing: ${detections.size} detections before NMS (confidence > $confidenceThreshold)")
        return detections
    }
    
    /**
     * Apply Non-Maximum Suppression to remove overlapping detections
     */
    private fun applyNMS(detections: List<LPDetection>, iouThreshold: Float): List<LPDetection> {
        if (detections.isEmpty()) return emptyList()
        
        // Sort by confidence (descending)
        val sortedDetections = detections.sortedByDescending { it.confidence }
        val selectedDetections = mutableListOf<LPDetection>()
        val suppressed = BooleanArray(sortedDetections.size) { false }
        
        for (i in sortedDetections.indices) {
            if (suppressed[i]) continue
            
            selectedDetections.add(sortedDetections[i])
            
            // Suppress overlapping boxes
            for (j in i + 1 until sortedDetections.size) {
                if (suppressed[j]) continue
                
                val iou = calculateIoU(sortedDetections[i], sortedDetections[j])
                if (iou > iouThreshold) {
                    suppressed[j] = true
                }
            }
        }
        
        Log.d(TAG, "NMS: ${selectedDetections.size} detections after suppression")
        return selectedDetections
    }
    
    /**
     * Calculate Intersection over Union (IoU) between two detections
     */
    private fun calculateIoU(det1: LPDetection, det2: LPDetection): Float {
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
     * Draw detections on a canvas
     * 
     * @param canvas Canvas to draw on
     * @param detections List of detections to visualize
     * @param scaleX Horizontal scale factor
     * @param scaleY Vertical scale factor
     */
    fun drawDetections(
        canvas: Canvas,
        detections: List<LPDetection>,
        scaleX: Float = 1f,
        scaleY: Float = 1f
    ) {
        for (detection in detections) {
            // Scale coordinates
            val x1 = detection.x1 * scaleX
            val y1 = detection.y1 * scaleY
            val x2 = detection.x2 * scaleX
            val y2 = detection.y2 * scaleY
            
            // Draw bounding box
            canvas.drawRect(x1, y1, x2, y2, boxPaint)
            
            // Prepare label text
            val label = "${detection.label} ${String.format("%.2f", detection.confidence)}"
            
            // Measure text size
            val textBounds = Rect()
            textPaint.getTextBounds(label, 0, label.length, textBounds)
            val textWidth = textBounds.width()
            val textHeight = textBounds.height()
            
            // Draw background for text
            val textX = x1
            val textY = max(y1 - 10f, textHeight.toFloat())
            canvas.drawRect(
                textX,
                textY - textHeight - 10f,
                textX + textWidth + 10f,
                textY,
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
