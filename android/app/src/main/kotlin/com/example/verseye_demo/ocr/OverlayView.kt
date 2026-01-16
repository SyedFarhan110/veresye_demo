package com.example.verseye_demo.ocr

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
        setShadowLayer(4f, 2f, 2f, Color.BLACK)
    }
    private val paintBackground = Paint().apply {
        color = Color.argb(180, 0, 0, 0)
        style = Paint.Style.FILL
    }
    
    private var scaleX = 1f
    private var scaleY = 1f
    private var offsetX = 0f
    private var offsetY = 0f
    private var lastUpdateTime = 0L
    private val FADE_DURATION = 2000L

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
        
        val imageAspectRatio = imageWidth.toFloat() / imageHeight.toFloat()
        val previewAspectRatio = previewWidth.toFloat() / previewHeight.toFloat()
        
        if (imageAspectRatio > previewAspectRatio) {
            scaleX = previewWidth.toFloat() / imageWidth.toFloat()
            scaleY = scaleX
            offsetX = 0f
            offsetY = (previewHeight - (imageHeight * scaleY)) / 2f
        } else {
            scaleY = previewHeight.toFloat() / imageHeight.toFloat()
            scaleX = scaleY
            offsetY = 0f
            offsetX = (previewWidth - (imageWidth * scaleX)) / 2f
        }
        
        invalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        
        val timeSinceUpdate = System.currentTimeMillis() - lastUpdateTime
        if (timeSinceUpdate > FADE_DURATION) {
            ocrResult = null
            return
        }
        
        val alpha = (255 * (1 - (timeSinceUpdate.toFloat() / FADE_DURATION))).toInt().coerceIn(0, 255)
        
        ocrResult?.let { result ->
            result.outputRawResult.forEach { item ->
                val points = item.points
                if (points.isNotEmpty()) {
                    paintBox.alpha = alpha
                    paintText.alpha = alpha
                    paintBackground.alpha = (alpha * 0.7f).toInt()
                    
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
                    
                    val text = item.label
                    val textX = points[0].x * scaleX + offsetX
                    val textY = points[0].y * scaleY + offsetY - 15
                    
                    val textBounds = android.graphics.Rect()
                    paintText.getTextBounds(text, 0, text.length, textBounds)
                    
                    canvas.drawRect(
                        textX - 5,
                        textY + textBounds.top - 5,
                        textX + textBounds.width() + 5,
                        textY + textBounds.bottom + 5,
                        paintBackground
                    )
                    
                    canvas.drawText(text, textX, textY, paintText)
                }
            }
            
            if (timeSinceUpdate < FADE_DURATION) {
                postInvalidateDelayed(16)
            }
        }
    }
}
