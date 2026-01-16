package com.equationl.paddleocr4android.app

import android.graphics.BitmapFactory
import android.os.Bundle
import android.util.Log
import android.widget.Button
import android.widget.ImageView
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import com.equationl.paddleocr4android.CpuPowerMode
import com.equationl.paddleocr4android.OCR
import com.equationl.paddleocr4android.OcrConfig
import com.equationl.paddleocr4android.bean.OcrResult
import com.equationl.paddleocr4android.callback.OcrInitCallback
import com.equationl.paddleocr4android.callback.OcrRunCallback

class MainActivity : AppCompatActivity() {
    private val TAG = "el, Main"

    private lateinit var ocr: OCR

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        ocr = OCR(this)

        val initBtn = findViewById<Button>(R.id.init_model)
        val startBtn = findViewById<Button>(R.id.start_model)
        val resultImg = findViewById<ImageView>(R.id.result_img)
        val resultText = findViewById<TextView>(R.id.result_text)

        initBtn.setOnClickListener {
            // Configuration
            val config = OcrConfig()
            //config.labelPath = null

            // Model path: without "/" means assets folder inside APK
            config.modelPath = "models/ch_PP-OCRv4" 
            //config.modelPath = "/sdcard/Android/data/com.equationl.paddleocr4android.app/files/models"
            // Using "/" means a path on phone storage; put downloaded models in this folder for testing

            // Model filenames
            config.clsModelFilename = "cls.nb" // classification model file
            config.detModelFilename = "det.nb" // detection model file
            config.recModelFilename = "rec.nb" // recognition model file

            // Run all models
            // Using all three models gives the best recognition accuracy
            // If only recognition is enabled, the results may be very poor
            // At minimum, detection or classification should be enabled
            config.isRunDet = true
            config.isRunCls = true
            config.isRunRec = true

            // Use all CPU cores
            config.cpuPowerMode = CpuPowerMode.LITE_POWER_FULL

            // Draw bounding boxes for detected text
            config.isDrwwTextPositionBox = true

            // 1. Synchronous initialization (commented out)
            /*
            ocr.initModelSync(config).fold(
                {
                    if (it) {
                        Log.i(TAG, "onCreate: init success")
                    }
                },
                {
                    it.printStackTrace()
                }
            )
            */

            // 2. Asynchronous initialization
            resultText.text = "Loading model..."
            ocr.initModel(config, object : OcrInitCallback {
                override fun onSuccess() {
                    resultText.text = "Model loaded successfully"
                    Log.i(TAG, "onSuccess: initialization successful")
                }

                override fun onFail(e: Throwable) {
                    resultText.text = "Model loading failed: $e"
                    Log.e(TAG, "onFail: initialization failed", e)
                }
            })
        }

        startBtn.setOnClickListener {
            // 1. Synchronous recognition (commented out)
            /*
            val bitmap = BitmapFactory.decodeResource(resources, R.drawable.test2)
            ocr.runSync(bitmap)

            val bitmap2 = BitmapFactory.decodeResource(resources, R.drawable.test3)
            ocr.runSync(bitmap2)
            */

            // 2. Asynchronous recognition
            resultText.text = "Starting recognition..."
            val bitmap3 = BitmapFactory.decodeResource(resources, R.drawable.test4)
            ocr.run(bitmap3, object : OcrRunCallback {
                override fun onSuccess(result: OcrResult) {
                    val simpleText = result.simpleText
                    val imgWithBox = result.imgWithBox
                    val inferenceTime = result.inferenceTime
                    val outputRawResult = result.outputRawResult

                    var text = "Recognized text=\n$simpleText\nInference time=$inferenceTime ms\nMore details=\n"

                    val wordLabels = ocr.getWordLabels()
                    outputRawResult.forEachIndexed { index, ocrResultModel ->
                        // Text index (crResultModel.wordIndex) corresponds to label in wordLabels
                        ocrResultModel.wordIndex.forEach {
                            Log.i(TAG, "onSuccess: text = ${wordLabels[it]}")
                        }
                        // Text orientation: clsLabel may be "0" or "180"
                        text += "$index: Text direction: ${ocrResultModel.clsLabel}; Direction confidence: ${ocrResultModel.clsConfidence}; Recognition confidence: ${ocrResultModel.confidence}; Word index positions: ${ocrResultModel.wordIndex}; Text position points: ${ocrResultModel.points}\n"
                    }

                    resultText.text = text
                    resultImg.setImageBitmap(imgWithBox)
                }

                override fun onFail(e: Throwable) {
                    resultText.text = "Recognition failed: $e"
                    Log.e(TAG, "onFail: recognition failed", e)
                }
            })
        }
        
        val realtimeBtn = findViewById<Button>(R.id.realtime_btn)
        realtimeBtn.setOnClickListener {
             startActivity(android.content.Intent(this, CameraActivity::class.java))
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        // Release OCR resources
        ocr.releaseModel()
    }
}
