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
import java.util.concurrent.Executors
import kotlin.math.max
import kotlin.math.min
import kotlin.math.exp

class SegmentationHelper(
    private val context: Context
) {
    
    companion object {
        private const val TAG = "SegmentationHelper"
    }
    
    private var interpreter: Interpreter? = null
    private var labels = listOf<String>()
    private var gpuDelegate: GpuDelegate? = null
    
    // Model input/output dimensions
    private var inputWidth = 640
    private var inputHeight = 640
    private var maskWidth = 160
    private var maskHeight = 160
    private var maskChannels = 32
    
    private var out0NumFeatures = 0
    private var out0NumAnchors = 0
    
    private val boxFeatureLength = 4
    private val maskConfidenceLength = 32
    private var numClasses = 0
    
    private lateinit var imageProcessor: ImageProcessor
    
    private var confidenceThreshold = 0.25f
    private var iouThreshold = 0.45f
    
    // Reuse buffers
    private lateinit var inputBuffer: ByteBuffer
    private lateinit var output0: Array<Array<FloatArray>>
    private lateinit var output1: Array<Array<Array<FloatArray>>>
    
    // Pre-allocated buffers for mask generation
    private lateinit var maskPixelsBuffer: IntArray
    private lateinit var tempMaskBuffer: FloatArray
    private lateinit var sigmoidLookupTable: FloatArray
    
    // Thread pool for parallel processing
    private val executorService = Executors.newFixedThreadPool(4)

    private var frameCounter = 0
    private var skipFrames = 0  // CHANGED: Disabled frame skipping
    
    // Temporal smoothing: Store last N detections
    private val detectionHistory = mutableListOf<Pair<Long, List<SegmentationResult>>>()
    private val maxHistoryFrames = 3
    private var lastSmoothedMaskBitmap: Bitmap? = null
    private var inferenceCount = 0
    
    private val colors = intArrayOf(
        Color.parseColor("#FF3838"),
        Color.parseColor("#FF9D97"),
        Color.parseColor("#FF701F"),
        Color.parseColor("#FFB21D"),
        Color.parseColor("#CFD231"),
        Color.parseColor("#48F90A"),
        Color.parseColor("#92CC17"),
        Color.parseColor("#3DDB86"),
        Color.parseColor("#1A9334"),
        Color.parseColor("#00D4BB"),
        Color.parseColor("#2C99A8"),
        Color.parseColor("#00C2FF"),
        Color.parseColor("#344593"),
        Color.parseColor("#6473FF"),
        Color.parseColor("#0018EC"),
        Color.parseColor("#8438FF"),
        Color.parseColor("#520085"),
        Color.parseColor("#CB38FF"),
        Color.parseColor("#FF95C8"),
        Color.parseColor("#FF37C7")
    )
    
    data class SegmentationResult(
        val x1: Float,
        val y1: Float,
        val x2: Float,
        val y2: Float,
        val confidence: Float,
        val classId: Int,
        val label: String,
        val maskCoeffs: FloatArray
    )
    
    private data class Detection(
        val box: RectF,
        val cls: Int,
        val score: Float,
        val maskCoeffs: FloatArray
    )
    
    fun initialize(modelFile: File, labelsFile: File): Boolean {
        return try {
            Log.d(TAG, "Initializing segmentation model...")
            
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
            
            val output0Shape = interpreter!!.getOutputTensor(0).shape()
            val output1Shape = interpreter!!.getOutputTensor(1).shape()
            
            Log.d(TAG, "Output 0 shape: [${output0Shape.joinToString()}]")
            Log.d(TAG, "Output 1 shape: [${output1Shape.joinToString()}]")
            
            val batch0 = output0Shape[0]
            out0NumFeatures = output0Shape[1]
            out0NumAnchors = output0Shape[2]
            output0 = Array(batch0) { Array(out0NumFeatures) { FloatArray(out0NumAnchors) } }
            
            val batch1 = output1Shape[0]
            maskHeight = output1Shape[1]
            maskWidth = output1Shape[2]
            maskChannels = output1Shape[3]
            output1 = Array(batch1) { Array(maskHeight) { Array(maskWidth) { FloatArray(maskChannels) } } }
            
            numClasses = out0NumFeatures - boxFeatureLength - maskConfidenceLength
            Log.d(TAG, "Number of classes: $numClasses")
            
            val inputBytes = 1 * inputHeight * inputWidth * 3 * 4
            inputBuffer = ByteBuffer.allocateDirect(inputBytes).apply {
                order(ByteOrder.nativeOrder())
            }
            
            imageProcessor = ImageProcessor.Builder()
                .add(ResizeOp(inputHeight, inputWidth, ResizeOp.ResizeMethod.BILINEAR))
                .add(NormalizeOp(0f, 255f))
                .add(CastOp(DataType.FLOAT32))
                .build()
            
            maskPixelsBuffer = IntArray(maskWidth * maskHeight)
            tempMaskBuffer = FloatArray(maskWidth * maskHeight)
            
            initSigmoidLookupTable()
            
            Log.d(TAG, "Segmentation model initialized successfully")
            true
        } catch (e: Exception) {
            Log.e(TAG, "Error initializing segmentation model", e)
            false
        }
    }
    
    private fun initSigmoidLookupTable() {
        val tableSize = 2000
        sigmoidLookupTable = FloatArray(tableSize)
        val minVal = -10f
        val maxVal = 10f
        val range = maxVal - minVal
        
        for (i in 0 until tableSize) {
            val x = minVal + (i.toFloat() / tableSize) * range
            sigmoidLookupTable[i] = (1.0f / (1.0f + exp(-x))).toFloat()
        }
    }
    
    private fun fastSigmoid(x: Float): Float {
        val minVal = -10f
        val maxVal = 10f
        val clampedX = x.coerceIn(minVal, maxVal)
        val range = maxVal - minVal
        val index = ((clampedX - minVal) / range * (sigmoidLookupTable.size - 1)).toInt()
        return sigmoidLookupTable[index.coerceIn(0, sigmoidLookupTable.size - 1)]
    }
    
    fun runInference(bitmap: Bitmap): Pair<List<SegmentationResult>, Bitmap?> {
        val currentInterpreter = interpreter ?: run {
            Log.e(TAG, "Interpreter not initialized")
            return Pair(emptyList(), null)
        }
        
        try {
            val tensorImage = TensorImage(DataType.FLOAT32)
            tensorImage.load(bitmap)
            val processedImage = imageProcessor.process(tensorImage)
            
            inputBuffer.clear()
            inputBuffer.put(processedImage.buffer)
            inputBuffer.rewind()
            
            val outputMap = mapOf(0 to output0, 1 to output1)
            currentInterpreter.runForMultipleInputsOutputs(arrayOf(inputBuffer), outputMap)
            
            val rawDetections = postProcessSegmentOptimized(
                feature = output0[0],
                numAnchors = out0NumAnchors,
                confidenceThreshold = confidenceThreshold,
                iouThreshold = iouThreshold
            )
            
            val results = mutableListOf<SegmentationResult>()
            for (det in rawDetections) {
                val x1 = det.box.left * bitmap.width
                val y1 = det.box.top * bitmap.height
                val x2 = det.box.right * bitmap.width
                val y2 = det.box.bottom * bitmap.height
                
                val label = labels.getOrElse(det.cls) { "Class ${det.cls}" }
                
                results.add(
                    SegmentationResult(
                        x1 = x1,
                        y1 = y1,
                        x2 = x2,
                        y2 = y2,
                        confidence = det.score,
                        classId = det.cls,
                        label = label,
                        maskCoeffs = det.maskCoeffs
                    )
                )
            }
            
            // Add to history for temporal smoothing
            val currentTime = System.currentTimeMillis()
            detectionHistory.add(0, Pair(currentTime, results))
            if (detectionHistory.size > maxHistoryFrames) {
                detectionHistory.removeAt(detectionHistory.size - 1)
            }
            
            // Generate mask bitmap
            val maskBitmap = if (rawDetections.isNotEmpty()) {
                generateCombinedMaskImageOptimized(
                    detections = rawDetections,
                    protos = output1[0],
                    maskW = maskWidth,
                    maskH = maskHeight,
                    threshold = 0.5f,
                    origWidth = bitmap.width,
                    origHeight = bitmap.height
                )
            } else {
                null
            }
            
            lastSmoothedMaskBitmap = maskBitmap
            inferenceCount++
            
            return Pair(results, maskBitmap)
        } catch (e: Exception) {
            Log.e(TAG, "Error during inference", e)
            return Pair(emptyList(), null)
        }
    }
    
    // Get smoothed results by averaging detections over time
    fun getSmoothedResults(): Pair<List<SegmentationResult>, Bitmap?> {
        if (detectionHistory.isEmpty()) {
            return Pair(emptyList(), null)
        }
        
        // If we have recent detections, use them; otherwise return the last one
        val smoothedResults = if (detectionHistory.isNotEmpty()) {
            averageDetections(detectionHistory.map { it.second })
        } else {
            detectionHistory.first().second
        }
        
        return Pair(smoothedResults, lastSmoothedMaskBitmap)
    }
    
    // Average detections over multiple frames for smooth results
    private fun averageDetections(detectionsList: List<List<SegmentationResult>>): List<SegmentationResult> {
        if (detectionsList.isEmpty()) return emptyList()
        
        // Match detections across frames and average their positions
        val latestDetections = detectionsList.firstOrNull() ?: return emptyList()
        val smoothedResults = mutableListOf<SegmentationResult>()
            
        for (latestDet in latestDetections) {
            var sumX1 = latestDet.x1
            var sumY1 = latestDet.y1
            var sumX2 = latestDet.x2
            var sumY2 = latestDet.y2
            var sumConfidence = latestDet.confidence
            var matchCount = 1
            
            // Look for matching detections in history
            for (i in 1 until detectionsList.size) {
                val historyDets = detectionsList[i]
                
                // Find closest detection by class and proximity
                val closest = historyDets.filter { it.classId == latestDet.classId }
                    .minByOrNull { histDet ->
                        val dx = (latestDet.x1 + latestDet.x2) / 2 - (histDet.x1 + histDet.x2) / 2
                        val dy = (latestDet.y1 + latestDet.y2) / 2 - (histDet.y1 + histDet.y2) / 2
                        dx * dx + dy * dy
                    }
                
                if (closest != null) {
                    sumX1 += closest.x1
                    sumY1 += closest.y1
                    sumX2 += closest.x2
                    sumY2 += closest.y2
                    sumConfidence += closest.confidence
                    matchCount++
                }
            }
            
            smoothedResults.add(
                SegmentationResult(
                    x1 = sumX1 / matchCount,
                    y1 = sumY1 / matchCount,
                    x2 = sumX2 / matchCount,
                    y2 = sumY2 / matchCount,
                    confidence = sumConfidence / matchCount,
                    classId = latestDet.classId,
                    label = latestDet.label,
                    maskCoeffs = latestDet.maskCoeffs
                )
            )
        }
        
        return smoothedResults
    }
    
    private fun postProcessSegmentOptimized(
        feature: Array<FloatArray>,
        numAnchors: Int,
        confidenceThreshold: Float,
        iouThreshold: Float
    ): List<Detection> {
        val results = ArrayList<Detection>(numAnchors / 10)
        val earlyThreshold = confidenceThreshold * 0.7f
        
        val boxes = Array(numAnchors) { j ->
            val cx = feature[0][j]
            val cy = feature[1][j]
            val w = feature[2][j]
            val h = feature[3][j]
            floatArrayOf(cx - w / 2f, cy - h / 2f, cx + w / 2f, cy + h / 2f)
        }
        
        for (j in 0 until numAnchors) {
            var maxScore = 0f
            var maxClassIdx = 0
            
            for (c in 0 until numClasses) {
                val score = feature[4 + c][j]
                if (score > maxScore) {
                    maxScore = score
                    maxClassIdx = c
                }
            }
            
            if (maxScore < earlyThreshold) continue
            
            if (maxScore >= confidenceThreshold) {
                val maskCoeffs = FloatArray(maskConfidenceLength)
                val base = 4 + numClasses
                for (m in 0 until maskConfidenceLength) {
                    maskCoeffs[m] = feature[base + m][j]
                }
                
                val box = boxes[j]
                results.add(Detection(
                    RectF(box[0], box[1], box[2], box[3]),
                    maxClassIdx,
                    maxScore,
                    maskCoeffs
                ))
            }
        }
        
        return applyNMSOptimized(results, iouThreshold)
    }
    
    private fun applyNMSOptimized(
        detections: List<Detection>,
        iouThreshold: Float
    ): List<Detection> {
        if (detections.isEmpty()) return emptyList()
        
        val finalDetections = mutableListOf<Detection>()
        
        // More aggressive NMS - lower threshold to filter more overlapping detections
        val strictIouThreshold = iouThreshold * 0.7f // Use 70% of original threshold for stricter filtering
        
        for (classIndex in 0 until numClasses) {
            val sameClass = detections.filter { it.cls == classIndex }
            if (sameClass.isEmpty()) continue
            
            val sorted = sameClass.sortedByDescending { it.score }
            val keep = BooleanArray(sorted.size) { true }
            
            for (i in sorted.indices) {
                if (!keep[i]) continue
                finalDetections.add(sorted[i])
                
                val boxA = sorted[i].box
                for (j in i + 1 until sorted.size) {
                    if (!keep[j]) continue
                    if (iou(boxA, sorted[j].box) > strictIouThreshold) {
                        keep[j] = false
                    }
                }
            }
        }
        
        return finalDetections
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
        return if (unionArea <= 0f) 0f else interArea / unionArea
    }
    
    private fun generateCombinedMaskImageOptimized(
        detections: List<Detection>,
        protos: Array<Array<FloatArray>>,
        maskW: Int,
        maskH: Int,
        threshold: Float,
        origWidth: Int,
        origHeight: Int
    ): Bitmap? {
        if (detections.isEmpty()) return null
        
        try {
            maskPixelsBuffer.fill(Color.TRANSPARENT)
            
            detections.forEachIndexed { _, det ->
                val color = colors[det.cls % colors.size]
                
                var idx = 0
                for (y in 0 until maskH) {
                    for (x in 0 until maskW) {
                        var v = 0f
                        val protoRow = protos[y][x]
                        
                        var c = 0
                        while (c < maskConfidenceLength - 3) {
                            v += det.maskCoeffs[c] * protoRow[c]
                            v += det.maskCoeffs[c + 1] * protoRow[c + 1]
                            v += det.maskCoeffs[c + 2] * protoRow[c + 2]
                            v += det.maskCoeffs[c + 3] * protoRow[c + 3]
                            c += 4
                        }
                        while (c < maskConfidenceLength) {
                            v += det.maskCoeffs[c] * protoRow[c]
                            c++
                        }
                        
                        if (fastSigmoid(v) > threshold) {
                            maskPixelsBuffer[idx] = color
                        }
                        idx++
                    }
                }
            }
            
            val maskBitmap = Bitmap.createBitmap(maskW, maskH, Bitmap.Config.ARGB_8888)
            maskBitmap.setPixels(maskPixelsBuffer, 0, maskW, 0, 0, maskW, maskH)
            
            val scaledBitmap = Bitmap.createScaledBitmap(maskBitmap, origWidth, origHeight, false)
            maskBitmap.recycle()
            
            return scaledBitmap
        } catch (e: Exception) {
            Log.e(TAG, "Error generating mask image", e)
            return null
        }
    }
    
    fun setConfidenceThreshold(threshold: Float) {
        confidenceThreshold = threshold
    }
    
    fun setIouThreshold(threshold: Float) {
        iouThreshold = threshold
    }
    
    fun close() {
        try {
            executorService.shutdown()
            interpreter?.close()
            gpuDelegate?.close()
            interpreter = null
            gpuDelegate = null
            detectionHistory.clear()
            lastSmoothedMaskBitmap?.recycle()
            Log.d(TAG, "Segmentation helper closed")
        } catch (e: Exception) {
            Log.e(TAG, "Error closing segmentation helper", e)
        }
    }
}