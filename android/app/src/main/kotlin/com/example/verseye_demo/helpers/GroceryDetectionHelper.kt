package com.example.verseye_demo.helpers

import android.content.Context
import android.graphics.*
import java.io.*

/**
 * Grocery Item Detection Helper (Refactored)
 * 
 * This is now a lightweight wrapper around GenericYOLODetectionHelper.
 * The heavy lifting is done by the generic helper, eliminating code duplication.
 * 
 * Model Specifications:
 * - Input Shape: (1, 640, 640, 3)
 * - Output Shape: (1, 5, 8400)
 * 
 * Output Format:
 * - 5 features per detection: [x_center, y_center, width, height, confidence]
 * - 8400 anchor points across the image
 * 
 * Classes:
 * - Multiple grocery items (fruits, vegetables, packaged goods, etc.)
 * - Useful for retail, inventory, and shopping applications
 * 
 * Benefits of Refactoring:
 * - No code duplication with other detection helpers
 * - Easier maintenance and bug fixes
 * - Consistent behavior across all models
 * - Same API maintained for backward compatibility
 */
class GroceryDetectionHelper(
    private val context: Context
) {
    // Delegate to the generic helper with grocery-specific configuration
    private val genericHelper = GenericYOLODetectionHelper(
        context = context,
        config = YOLOModelConfig.Grocery()
    )
    
    /**
     * Detection result data class (maintained for backward compatibility)
     */
    data class GroceryDetection(
        val x1: Float,
        val y1: Float,
        val x2: Float,
        val y2: Float,
        val confidence: Float,
        val classId: Int,
        val label: String
    )
    
    /**
     * Initialize the model (delegates to generic helper)
     */
    fun initialize(modelFile: File, labelsFile: File): Boolean {
        return genericHelper.initialize(modelFile, labelsFile)
    }
    
    
    /**
     * Run inference on a bitmap image (delegates to generic helper and converts result)
     */
    fun runInference(bitmap: Bitmap): List<GroceryDetection> {
        return genericHelper.runInference(bitmap).map { detection ->
            GroceryDetection(
                x1 = detection.x1,
                y1 = detection.y1,
                x2 = detection.x2,
                y2 = detection.y2,
                confidence = detection.confidence,
                classId = detection.classId,
                label = detection.label
            )
        }
    }
    
    /**
     * Draw detections on canvas with color coding (delegates to generic helper)
     */
    fun drawDetections(
        canvas: Canvas,
        detections: List<GroceryDetection>,
        scaleX: Float = 1f,
        scaleY: Float = 1f
    ) {
        // Convert back to generic detections for drawing
        val genericDetections = detections.map { detection ->
            GenericYOLODetectionHelper.Detection(
                x1 = detection.x1,
                y1 = detection.y1,
                x2 = detection.x2,
                y2 = detection.y2,
                confidence = detection.confidence,
                classId = detection.classId,
                label = detection.label
            )
        }
        genericHelper.drawDetections(canvas, genericDetections, scaleX, scaleY)
    }
    
    /**
     * Get item count summary
     */
    fun getItemSummary(detections: List<GroceryDetection>): Map<String, Int> {
        return detections.groupBy { it.label }.mapValues { it.value.size }
    }
    
    /**
     * Release resources (delegates to generic helper)
     */
    fun close() {
        genericHelper.close()
    }
}
