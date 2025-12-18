package com.example.verseye_demo.helpers

import android.graphics.Color

/**
 * Configuration for YOLO detection models
 * 
 * Provides a centralized way to configure different fine-tuned YOLO models
 * without duplicating code.
 * 
 * Usage:
 * ```kotlin
 * // Use predefined configurations
 * val groceryHelper = GenericYOLODetectionHelper(context, YOLOModelConfig.Grocery())
 * val helmetHelper = GenericYOLODetectionHelper(context, YOLOModelConfig.Helmet())
 * 
 * // Or create custom configuration
 * val customConfig = YOLOModelConfig(
 *     modelName = "Custom Model",
 *     classColors = mapOf("car" to Color.BLUE, "truck" to Color.GREEN)
 * )
 * ```
 */
data class YOLOModelConfig(
    val modelName: String,
    val confidenceThreshold: Float = 0.5f,
    val iouThreshold: Float = 0.45f,
    val boxStrokeWidth: Float = 4f,
    val textSize: Float = 40f,
    val textShadow: Boolean = false,
    val showConfidence: Boolean = true,
    val classColors: Map<String, Int> = emptyMap(),
    val defaultColors: List<Int> = DEFAULT_COLORS
) {
    
    companion object {
        /**
         * Default color palette for cycling through classes
         */
        val DEFAULT_COLORS = listOf(
            Color.parseColor("#FF6B6B"),  // Red
            Color.parseColor("#4ECDC4"),  // Teal
            Color.parseColor("#45B7D1"),  // Blue
            Color.parseColor("#FFA07A"),  // Light Salmon
            Color.parseColor("#98D8C8"),  // Mint
            Color.parseColor("#F7DC6F"),  // Yellow
            Color.parseColor("#BB8FCE"),  // Purple
            Color.parseColor("#85C1E2"),  // Sky Blue
            Color.parseColor("#F8B88B"),  // Peach
            Color.parseColor("#AED581")   // Light Green
        )
        
        // ==================== Predefined Model Configurations ====================
        
        /**
         * Grocery Item Detection Configuration
         * For retail, inventory, and shopping applications
         */
        fun Grocery() = YOLOModelConfig(
            modelName = "Grocery Detection",
            confidenceThreshold = 0.5f,
            iouThreshold = 0.45f,
            boxStrokeWidth = 4f,
            textSize = 40f
        )
        
        /**
         * Helmet Safety Detection Configuration
         * For construction site and workplace safety monitoring
         */
        fun Helmet() = YOLOModelConfig(
            modelName = "Helmet Detection",
            confidenceThreshold = 0.25f,
            iouThreshold = 0.45f,
            boxStrokeWidth = 5f,
            textSize = 42f,
            classColors = mapOf(
                "head" to Color.parseColor("#FF0000"),      // Red - unsafe
                "helmet" to Color.parseColor("#00FF00"),    // Green - safe
                "person" to Color.parseColor("#FFA500")     // Orange - person
            )
        )
        
        /**
         * Fire and Smoke Detection Configuration
         * For fire safety and emergency response systems
         */
        fun FireSmoke() = YOLOModelConfig(
            modelName = "Fire & Smoke Detection",
            confidenceThreshold = 0.45f,
            iouThreshold = 0.45f,
            boxStrokeWidth = 6f,
            textSize = 45f,
            textShadow = true,
            classColors = mapOf(
                "fire" to Color.parseColor("#FF4500"),      // Orange Red
                "smoke" to Color.parseColor("#808080")      // Gray
            )
        )
        
        /**
         * Face Detection Configuration
         * For face recognition and analysis applications
         */
        fun Face() = YOLOModelConfig(
            modelName = "Face Detection",
            confidenceThreshold = 0.5f,
            iouThreshold = 0.45f,
            boxStrokeWidth = 3f,
            textSize = 38f,
            classColors = mapOf(
                "face" to Color.parseColor("#00CED1")      // Dark Turquoise
            )
        )
        
        /**
         * Vehicle Dent Detection Configuration
         * For automotive inspection and insurance claims
         */
        fun Dent() = YOLOModelConfig(
            modelName = "Dent Detection",
            confidenceThreshold = 0.4f,
            iouThreshold = 0.45f,
            boxStrokeWidth = 4f,
            textSize = 40f,
            classColors = mapOf(
                "dent" to Color.parseColor("#DC143C"),      // Crimson
                "scratch" to Color.parseColor("#FF8C00")    // Dark Orange
            )
        )
        
        /**
         * License Plate Detection Configuration
         * For vehicle identification and parking systems
         */
        fun LicensePlate() = YOLOModelConfig(
            modelName = "License Plate Detection",
            confidenceThreshold = 0.5f,
            iouThreshold = 0.45f,
            boxStrokeWidth = 4f,
            textSize = 40f,
            classColors = mapOf(
                "plate" to Color.parseColor("#FFD700")     // Gold
            )
        )
        
        /**
         * Blink and Drowsiness Detection Configuration
         * For driver safety and attention monitoring
         */
        fun BlinkDrowse() = YOLOModelConfig(
            modelName = "Blink & Drowsiness Detection",
            confidenceThreshold = 0.5f,
            iouThreshold = 0.45f,
            boxStrokeWidth = 3f,
            textSize = 38f,
            classColors = mapOf(
                "open_eyes" to Color.parseColor("#00FF00"),     // Green - alert
                "closed_eyes" to Color.parseColor("#FF0000"),   // Red - drowsy
                "yawn" to Color.parseColor("#FFA500")           // Orange - tired
            )
        )
        
        /**
         * General Object Detection Configuration (COCO dataset)
         * For general-purpose object detection (80 classes)
         */
        fun GeneralObject() = YOLOModelConfig(
            modelName = "General Object Detection",
            confidenceThreshold = 0.45f,
            iouThreshold = 0.45f,
            boxStrokeWidth = 4f,
            textSize = 40f
        )
        
        /**
         * Oil Spill Detection Configuration
         * For environmental monitoring
         */
        fun OilSpill() = YOLOModelConfig(
            modelName = "Oil Spill Detection",
            confidenceThreshold = 0.5f,
            iouThreshold = 0.45f,
            boxStrokeWidth = 5f,
            textSize = 42f,
            classColors = mapOf(
                "oil" to Color.parseColor("#8B4513")       // Saddle Brown
            )
        )
        
        /**
         * Tea and Scanner Detection Configuration
         * For specific retail/product detection
         */
        fun TeaScanner() = YOLOModelConfig(
            modelName = "Tea & Scanner Detection",
            confidenceThreshold = 0.5f,
            iouThreshold = 0.45f,
            boxStrokeWidth = 4f,
            textSize = 40f
        )
    }
    
    /**
     * Get color for a specific class
     * Uses classColors map if defined, otherwise cycles through defaultColors
     */
    fun getColorForClass(classId: Int, className: String): Int {
        return classColors[className.lowercase()] 
            ?: classColors[className]
            ?: defaultColors[classId % defaultColors.size]
    }
}
