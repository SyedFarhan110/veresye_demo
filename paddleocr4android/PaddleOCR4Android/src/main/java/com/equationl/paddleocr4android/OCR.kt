package com.equationl.paddleocr4android

import android.content.Context
import android.graphics.Bitmap
import android.widget.ImageView
import androidx.annotation.MainThread
import androidx.annotation.WorkerThread
import com.equationl.paddleocr4android.Util.paddle.Predictor
import com.equationl.paddleocr4android.bean.OcrResult
import com.equationl.paddleocr4android.callback.OcrInitCallback
import com.equationl.paddleocr4android.callback.OcrRunCallback
import com.equationl.paddleocr4android.exception.InitModelException
import com.equationl.paddleocr4android.exception.RunModelException
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import java.util.*

class OCR(val context: Context) {

    private var predictor: Predictor = Predictor()

    private var resultImageView: ImageView? = null

    private var modelPath = "models/ocr_v2_for_cpu"
    private var labelPath: String? = "labels/ppocr_keys_v1.txt"
    private var cpuThreadNum = 4
    private var cpuPowerMode = "LITE_POWER_HIGH"
    private var scoreThreshold = 0.1f
    private var modelFileNames = arrayOf<String>()
    private var isRunCls = true
    private var isRunDet = true
    private var isRunRec = true
    private var isUseOpencl = false
    private var detLongSize = 960
    private var isDrwwTextPositionBox = false

    fun getPredictor(): Predictor {
        return predictor
    }

    fun getWordLabels(): Vector<String> {
        return predictor.wordLabels
    }

    /**
     *
     * Initialize Model (Sync)
     *
     * @param config Configuration
     *
     * */
    @WorkerThread
    fun initModelSync(config: OcrConfig? = null): Result<Boolean>{
        if (config != null) {
            setConfig(config)
        }

        return try {
            Result.success(
                predictor.init(
                    context, modelPath, labelPath, isUseOpencl, cpuThreadNum,
                    cpuPowerMode, detLongSize, scoreThreshold,
                    modelFileNames, isDrwwTextPositionBox
                )
            )
        } catch (e: Throwable) {
            Result.failure(e)
        }
    }

    /**
     * Initialize Model (Async)
     *
     * @param config Configuration
     * @param callback Initialization callback
     * */
    @MainThread
    fun initModel(config: OcrConfig? = null, callback: OcrInitCallback) {
        val coroutineScope = CoroutineScope(Dispatchers.IO)
        coroutineScope.launch(Dispatchers.IO) {
            initModelSync(config).fold(
                {
                    coroutineScope.launch(Dispatchers.Main) {
                        if (it) {
                            callback.onSuccess()
                        }
                        else {
                            callback.onFail(InitModelException("Unknown error"))
                        }
                    }
                },
                {
                    coroutineScope.launch(Dispatchers.Main) {
                        callback.onFail(it)
                    }
                }
            )
        }
    }


    /**
     * Start Recognition Model (Sync)
     *
     * @param bitmap Image to recognize
     * */
    @WorkerThread
    fun runSync(bitmap: Bitmap): Result<OcrResult> {

        if (!predictor.isLoaded()) {
            return Result.failure(RunModelException("Please load the model first!"))
        }
        else {
            predictor.setInputImage(bitmap) // Load Image

            runModel().fold({
                return if (it) {
                    val ocrResult = OcrResult(
                        predictor.outputResult(),
                        predictor.inferenceTime(),
                        predictor.outputImage(),
                        predictor.outputRawResult()
                    )
                    Result.success(ocrResult)
                } else {
                    Result.failure(RunModelException("Please check if the model is loaded successfully!"))
                }
            }, {
                return Result.failure(it)
            })
        }
    }

    /**
     * Start Recognition Model (Async)
     *
     * @param bitmap Image to recognize
     * @param callback Recognition Result Callback
     * */
    @MainThread
    fun run(bitmap: Bitmap, callback: OcrRunCallback) {
        val coroutineScope = CoroutineScope(Dispatchers.IO)
        coroutineScope.launch(Dispatchers.IO) {
            runSync(bitmap).fold(
                {
                    coroutineScope.launch(Dispatchers.Main) {
                        callback.onSuccess(it)
                    }
                },
                {
                    coroutineScope.launch(Dispatchers.Main) {
                        callback.onFail(it)
                    }
                })
        }
    }

    /**
     * Release Model
     * */
    fun releaseModel() {
        predictor.releaseModel()
    }

    private fun runModel(): Result<Boolean> {
        return try {
            Result.success(predictor.isLoaded() && predictor.runModel(isRunDet, isRunCls, isRunRec))
        } catch (e: Throwable) {
            Result.failure(e)
        }
    }

    private fun setConfig(config: OcrConfig) {
        this.modelPath = config.modelPath
        this.labelPath = config.labelPath
        this.cpuThreadNum = config.cpuThreadNum
        this.cpuPowerMode = config.cpuPowerMode.name
        this.scoreThreshold = config.scoreThreshold
        this.modelFileNames = arrayOf(
            config.detModelFilename,
            config.recModelFilename,
            config.clsModelFilename)
        this.detLongSize = config.detLongSize
        this.isRunDet = config.isRunDet
        this.isRunCls = config.isRunCls
        this.isRunRec = config.isRunRec
        this.isUseOpencl = config.isUseOpencl
        this.isDrwwTextPositionBox = config.isDrwwTextPositionBox
    }

}