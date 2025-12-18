package com.example.verseye_demo.helpers

import android.content.Context
import android.graphics.*
import java.io.*

/**
 * Generic Object Detection Helper Factory
 * 
 * Provides easy instantiation of detection helpers for all model types.
 * Eliminates the need for separate helper classes while maintaining flexibility.
 * 
 * Usage Examples:
 * 
 * 1. Using predefined models:
 * ```kotlin
 * val groceryHelper = YOLODetectionFactory.createGroceryDetector(context)
 * val helmetHelper = YOLODetectionFactory.createHelmetDetector(context)
 * ```
 * 
 * 2. Using custom configuration:
 * ```kotlin
 * val customConfig = YOLOModelConfig(
 *     modelName = "My Model",
 *     confidenceThreshold = 0.6f,
 *     classColors = mapOf("car" to Color.BLUE)
 * )
 * val customHelper = YOLODetectionFactory.create(context, customConfig)
 * ```
 * 
 * 3. Quick one-liner:
 * ```kotlin
 * val detector = YOLODetectionFactory.createGroceryDetector(context)
 *     .also { it.initialize(modelFile, labelsFile) }
 * val results = detector.runInference(bitmap)
 * ```
 */
object YOLODetectionFactory {
    
    /**
     * Create a generic detector with custom configuration
     */
    fun create(context: Context, config: YOLOModelConfig): GenericYOLODetectionHelper {
        return GenericYOLODetectionHelper(context, config)
    }
    
    /**
     * Create a grocery item detector
     */
    fun createGroceryDetector(context: Context): GenericYOLODetectionHelper {
        return GenericYOLODetectionHelper(context, YOLOModelConfig.Grocery())
    }
    
    /**
     * Create a helmet safety detector
     */
    fun createHelmetDetector(context: Context): GenericYOLODetectionHelper {
        return GenericYOLODetectionHelper(context, YOLOModelConfig.Helmet())
    }
    
    /**
     * Create a fire and smoke detector
     */
    fun createFireSmokeDetector(context: Context): GenericYOLODetectionHelper {
        return GenericYOLODetectionHelper(context, YOLOModelConfig.FireSmoke())
    }
    
    /**
     * Create a face detector
     */
    fun createFaceDetector(context: Context): GenericYOLODetectionHelper {
        return GenericYOLODetectionHelper(context, YOLOModelConfig.Face())
    }
    
    /**
     * Create a vehicle dent detector
     */
    fun createDentDetector(context: Context): GenericYOLODetectionHelper {
        return GenericYOLODetectionHelper(context, YOLOModelConfig.Dent())
    }
    
    /**
     * Create a license plate detector
     */
    fun createLicensePlateDetector(context: Context): GenericYOLODetectionHelper {
        return GenericYOLODetectionHelper(context, YOLOModelConfig.LicensePlate())
    }
    
    /**
     * Create a blink/drowsiness detector
     */
    fun createBlinkDrowseDetector(context: Context): GenericYOLODetectionHelper {
        return GenericYOLODetectionHelper(context, YOLOModelConfig.BlinkDrowse())
    }
    
    /**
     * Create a general object detector (COCO 80 classes)
     */
    fun createGeneralDetector(context: Context): GenericYOLODetectionHelper {
        return GenericYOLODetectionHelper(context, YOLOModelConfig.GeneralObject())
    }
    
    /**
     * Create an oil spill detector
     */
    fun createOilSpillDetector(context: Context): GenericYOLODetectionHelper {
        return GenericYOLODetectionHelper(context, YOLOModelConfig.OilSpill())
    }
    
    /**
     * Create a tea/scanner detector
     */
    fun createTeaScannerDetector(context: Context): GenericYOLODetectionHelper {
        return GenericYOLODetectionHelper(context, YOLOModelConfig.TeaScanner())
    }
}

/**
 * Extension functions for easier usage
 */
fun GenericYOLODetectionHelper.initializeAndRun(
    modelFile: File,
    labelsFile: File,
    bitmap: Bitmap
): List<GenericYOLODetectionHelper.Detection> {
    return if (initialize(modelFile, labelsFile)) {
        runInference(bitmap)
    } else {
        emptyList()
    }
}

/**
 * DSL-style builder for custom detection configurations
 */
class YOLODetectorBuilder(private val context: Context) {
    private var modelName: String = "Custom YOLO"
    private var confidenceThreshold: Float = 0.5f
    private var iouThreshold: Float = 0.45f
    private var boxStrokeWidth: Float = 4f
    private var textSize: Float = 40f
    private var textShadow: Boolean = false
    private var showConfidence: Boolean = true
    private var classColors: Map<String, Int> = emptyMap()
    
    fun modelName(name: String) = apply { this.modelName = name }
    fun confidenceThreshold(threshold: Float) = apply { this.confidenceThreshold = threshold }
    fun iouThreshold(threshold: Float) = apply { this.iouThreshold = threshold }
    fun boxStrokeWidth(width: Float) = apply { this.boxStrokeWidth = width }
    fun textSize(size: Float) = apply { this.textSize = size }
    fun textShadow(enabled: Boolean) = apply { this.textShadow = enabled }
    fun showConfidence(show: Boolean) = apply { this.showConfidence = show }
    fun classColors(colors: Map<String, Int>) = apply { this.classColors = colors }
    
    fun build(): GenericYOLODetectionHelper {
        val config = YOLOModelConfig(
            modelName = modelName,
            confidenceThreshold = confidenceThreshold,
            iouThreshold = iouThreshold,
            boxStrokeWidth = boxStrokeWidth,
            textSize = textSize,
            textShadow = textShadow,
            showConfidence = showConfidence,
            classColors = classColors
        )
        return GenericYOLODetectionHelper(context, config)
    }
}

/**
 * DSL entry point
 */
fun yoloDetector(context: Context, block: YOLODetectorBuilder.() -> Unit): GenericYOLODetectionHelper {
    return YOLODetectorBuilder(context).apply(block).build()
}
