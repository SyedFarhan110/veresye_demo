package com.equationl.paddleocr4android.app

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.util.AttributeSet
import android.view.View
import com.equationl.paddleocr4android.bean.OcrResult

class OverlayView(context: Context, attrs: AttributeSet?) : View(context, attrs) {

    private var ocrResult: OcrResult? = null
    private val paintBox = Paint().apply {
        color = Color.GREEN
        style = Paint.Style.STROKE
        strokeWidth = 8f
        isAntiAlias = true
    }
    private val paintText = Paint().apply {
        color = Color.WHITE
        textSize = 36f
        style = Paint.Style.FILL
        isAntiAlias = true
        setShadowLayer(4f, 2f, 2f, Color.BLACK) // Add shadow for better visibility
    }
    private val paintBackground = Paint().apply {
        color = Color.argb(180, 0, 0, 0) // Semi-transparent black background
        style = Paint.Style.FILL
    }
    
    private var scaleX = 1f
    private var scaleY = 1f
    private var offsetX = 0f
    private var offsetY = 0f
    private var lastUpdateTime = 0L
    private val FADE_DURATION = 2000L // Results fade after 2 seconds

    fun setResult(
        result: OcrResult, 
        imageWidth: Int, 
        imageHeight: Int,
        rotation: Int,
        previewWidth: Int,
        previewHeight: Int
    ) {
        ocrResult = result
        lastUpdateTime = System.currentTimeMillis()
        
        // Calculate scaling to fit the preview while maintaining aspect ratio
        val imageAspectRatio = imageWidth.toFloat() / imageHeight.toFloat()
        val previewAspectRatio = previewWidth.toFloat() / previewHeight.toFloat()
        
        if (imageAspectRatio > previewAspectRatio) {
            // Image is wider, fit to width
            scaleX = previewWidth.toFloat() / imageWidth.toFloat()
            scaleY = scaleX
            offsetX = 0f
            offsetY = (previewHeight - (imageHeight * scaleY)) / 2f
        } else {
            // Image is taller, fit to height
            scaleY = previewHeight.toFloat() / imageHeight.toFloat()
            scaleX = scaleY
            offsetY = 0f
            offsetX = (previewWidth - (imageWidth * scaleX)) / 2f
        }
        
        invalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        
        // Calculate fade alpha based on time since last update
        val timeSinceUpdate = System.currentTimeMillis() - lastUpdateTime
        if (timeSinceUpdate > FADE_DURATION) {
            ocrResult = null // Clear old results
            return
        }
        
        val alpha = (255 * (1 - (timeSinceUpdate.toFloat() / FADE_DURATION))).toInt().coerceIn(0, 255)
        
        ocrResult?.let { result ->
            result.outputRawResult.forEach { item ->
                val points = item.points
                if (points.isNotEmpty()) {
                    // Apply alpha for fade effect
                    paintBox.alpha = alpha
                    paintText.alpha = alpha
                    paintBackground.alpha = (alpha * 0.7f).toInt()
                    
                    // Draw bounding box
                    val path = android.graphics.Path()
                    path.moveTo(
                        points[0].x * scaleX + offsetX, 
                        points[0].y * scaleY + offsetY
                    )
                    for (i in 1 until points.size) {
                        path.lineTo(
                            points[i].x * scaleX + offsetX, 
                            points[i].y * scaleY + offsetY
                        )
                    }
                    path.close()
                    canvas.drawPath(path, paintBox)
                    
                    // Draw text with background for better visibility
                    val text = item.label
                    val textX = points[0].x * scaleX + offsetX
                    val textY = points[0].y * scaleY + offsetY - 15
                    
                    // Measure text to draw background
                    val textBounds = android.graphics.Rect()
                    paintText.getTextBounds(text, 0, text.length, textBounds)
                    
                    // Draw background rectangle
                    canvas.drawRect(
                        textX - 5,
                        textY + textBounds.top - 5,
                        textX + textBounds.width() + 5,
                        textY + textBounds.bottom + 5,
                        paintBackground
                    )
                    
                    // Draw text
                    canvas.drawText(text, textX, textY, paintText)
                }
            }
            
            // Keep invalidating for fade animation
            if (timeSinceUpdate < FADE_DURATION) {
                postInvalidateDelayed(16) // ~60fps
            }
        }
    }
}