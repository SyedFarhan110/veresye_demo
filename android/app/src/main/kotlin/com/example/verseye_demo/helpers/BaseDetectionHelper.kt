package com.example.verseye_demo.helpers

import android.graphics.Bitmap
import android.graphics.Canvas
import java.io.File

/**
 * Base interface for all detection helpers
 * Provides common methods for model initialization, inference, and cleanup
 */
interface BaseDetectionHelper {
    
    /**
     * Initialize the detection model with GPU acceleration if available
     * @param modelFile The TensorFlow Lite model file
     * @param labelsFile The labels file (optional)
     * @param useGpu Whether to use GPU acceleration
     */
    fun initialize(modelFile: File, labelsFile: File? = null, useGpu: Boolean = true)
    
    /**
     * Detect objects in the provided bitmap
     * @param bitmap Input image
     * @return Detection results (implementation specific)
     */
    fun detect(bitmap: Bitmap): Any
    
    /**
     * Draw detection results on the canvas
     * @param canvas Canvas to draw on
     * @param results Detection results from detect()
     * @param originalWidth Original image width
     * @param originalHeight Original image height
     */
    fun draw(canvas: Canvas, results: Any, originalWidth: Int, originalHeight: Int)
    
    /**
     * Clean up resources
     */
    fun close()
    
    /**
     * Check if the helper is initialized
     */
    fun isInitialized(): Boolean
    
    /**
     * Get model input size
     */
    fun getInputSize(): Pair<Int, Int>
}
