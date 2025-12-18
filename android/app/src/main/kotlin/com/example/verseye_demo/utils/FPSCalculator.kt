package com.example.verseye_demo.utils

import android.widget.TextView

/**
 * FPSCalculator handles FPS calculation and display
 */
class FPSCalculator {
    
    private var frameCounter = 0
    private var lastFpsTimestamp = System.currentTimeMillis()
    private var currentFps = 0.0
    
    /**
     * Update FPS counter and return current FPS
     */
    fun updateFPS(): Double {
        frameCounter++
        val currentTime = System.currentTimeMillis()
        val elapsedTime = currentTime - lastFpsTimestamp
        
        // Update FPS every second
        if (elapsedTime >= 1000) {
            currentFps = (frameCounter * 1000.0) / elapsedTime
            frameCounter = 0
            lastFpsTimestamp = currentTime
        }
        
        return currentFps
    }
    
    /**
     * Update FPS display on TextView
     */
    fun updateFPSDisplay(textView: TextView) {
        val fps = updateFPS()
        textView.post {
            textView.text = String.format("%.1f FPS", fps)
        }
    }
    
    /**
     * Get current FPS value
     */
    fun getCurrentFPS(): Double = currentFps
    
    /**
     * Reset FPS counter
     */
    fun reset() {
        frameCounter = 0
        lastFpsTimestamp = System.currentTimeMillis()
        currentFps = 0.0
    }
}
