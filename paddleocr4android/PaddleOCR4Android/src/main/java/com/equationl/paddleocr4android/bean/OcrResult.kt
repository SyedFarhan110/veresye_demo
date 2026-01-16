package com.equationl.paddleocr4android.bean

import android.graphics.Bitmap
import com.equationl.paddleocr4android.Util.paddle.OcrResultModel
import java.util.ArrayList

data class OcrResult(
    /**
     * Simple Recognition Result
     * */
    val simpleText: String,
    /**
    * Inference Time
    * */
    val inferenceTime: Float,
    /**
     * Image with text box
     * */
    val imgWithBox: Bitmap,
    /**
     * Original Recognition Result
     * */
    val outputRawResult: ArrayList<OcrResultModel>,
    )
