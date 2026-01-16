package com.equationl.paddleocr4android

import com.equationl.paddleocr4android.bean.OcrResult

data class OcrConfig(
    /**
     * Model path (defaults to pre-installed models in assets directory)
     *
     * If the value starts with "/", it is considered a custom path, and the program will load the model directly from that path;
     * Otherwise, it is considered a file under assets, and it will be copied to the cache directory before loading.
     *
     * */
    var modelPath:String = "models/ocr_v2_for_cpu",
    /**
     * Label list path (the recognition result returned by the program is the index of this list)
     * */
    var labelPath: String? = "labels/ppocr_keys_v1.txt",
    /**
     * Number of CPU threads to use
     * */
    var cpuThreadNum: Int = 4,
    /**
     * cpu power model
     * */
    var cpuPowerMode: CpuPowerMode = CpuPowerMode.LITE_POWER_HIGH,
    /**
     * Score Threshold
     * */
    var scoreThreshold: Float = 0.1f,

    var detLongSize: Int = 960,

    /**
     * Detection model filename
     * */
    var detModelFilename: String = "ch_ppocr_mobile_v2.0_det_opt.nb",

    /**
     * Recognition model filename
     * */
    var recModelFilename: String = "ch_ppocr_mobile_v2.0_rec_opt.nb",

    /**
     * Classification model filename
     * */
    var clsModelFilename: String = "ch_ppocr_mobile_v2.0_cls_opt.nb",

    /**
     * Whether to run the detection model
     * */
    var isRunDet: Boolean = true,

    /**
     * Whether to run the classification model
     * */
    var isRunCls: Boolean = true,

    /**
     * Whether to run the recognition model
     * */
    var isRunRec: Boolean = true,

    var isUseOpencl: Boolean = false,

    /**
     * Whether to draw text positions
     *
     * If true, [OcrResult.imgWithBox] returns a Bitmap with text position boxes drawn on the input Bitmap
     *
     * Otherwise, [OcrResult.imgWithBox] will directly return the input Bitmap
     * */
    var isDrwwTextPositionBox: Boolean = false
)

enum class CpuPowerMode {
    /**
     * HIGH(only big cores)
     * */
    LITE_POWER_HIGH,
    /**
     * LOW(only LITTLE cores)
     * */
    LITE_POWER_LOW,
    /**
     * FULL(all cores)
     * */
    LITE_POWER_FULL,
    /**
     * NO_BIND(depends on system)
     * */
    LITE_POWER_NO_BIND,
    /**
     * RAND_HIGH
     * */
    LITE_POWER_RAND_HIGH,
    /**
     * RAND_LOW
     * */
    LITE_POWER_RAND_LOW
}