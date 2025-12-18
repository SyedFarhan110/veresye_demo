package com.example.verseye_demo.managers

import android.content.Context
import android.util.Log
import com.example.verseye_demo.models.ModelInfo
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.json.JSONArray
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.CompatibilityList
import org.tensorflow.lite.gpu.GpuDelegate
import java.io.*
import java.net.URL
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

/**
 * ModelManager handles all model-related operations including:
 * - Fetching model list from API
 * - Downloading models and labels
 * - Loading and initializing models
 * - Managing model cache
 */
class ModelManager(private val context: Context) {
    
    companion object {
        private const val TAG = "ModelManager"
        private const val API_URL = "https://raw.githubusercontent.com/SyedFarhan110/Object_detection-/main/transformed_models.json"
    }
    
    private val availableModels = mutableListOf<ModelInfo>()
    private val modelInterpreters = mutableMapOf<String, Interpreter>()
    private var gpuDelegate: GpuDelegate? = null
    
    var currentModelIndex: Int? = null
    var currentInterpreter: Interpreter? = null
    
    // Callbacks
    var onLoadingProgressUpdate: ((String) -> Unit)? = null
    var onModelLoaded: ((ModelInfo) -> Unit)? = null
    var onError: ((String) -> Unit)? = null
    
    /**
     * Fetch available models from the API
     */
    suspend fun fetchModelList(): List<ModelInfo> {
        return withContext(Dispatchers.IO) {
            try {
                val jsonString = URL(API_URL).readText()
                val jsonArray = JSONArray(jsonString)
                availableModels.clear()
                
                for (i in 0 until jsonArray.length()) {
                    val modelObj = jsonArray.getJSONObject(i)
                    val fileName = modelObj.getString("model_url").substringAfterLast("/")
                    val labelsFileName = modelObj.getString("labels_url").substringAfterLast("/")
                    
                    // Check if model is already downloaded
                    val modelFile = File(context.filesDir, fileName)
                    val labelsFile = File(context.filesDir, labelsFileName)
                    val isDownloaded = modelFile.exists() && labelsFile.exists()
                    
                    val modelInfo = ModelInfo(
                        name = modelObj.getString("name"),
                        modelUrl = modelObj.getString("model_url"),
                        labelsUrl = modelObj.getString("labels_url"),
                        type = modelObj.getString("type"),
                        fileName = fileName,
                        labelsFileName = labelsFileName,
                        isDownloaded = isDownloaded
                    )
                    availableModels.add(modelInfo)
                }
                Log.d(TAG, "Found ${availableModels.size} models from API")
                availableModels.toList()
            } catch (e: Exception) {
                Log.e(TAG, "Error fetching model list: ${e.message}")
                throw e
            }
        }
    }
    
    /**
     * Get list of available models
     */
    fun getAvailableModels(): List<ModelInfo> = availableModels.toList()
    
    /**
     * Download model if not already downloaded
     */
    suspend fun downloadModelIfNeeded(modelInfo: ModelInfo): Boolean {
        return try {
            if (!modelInfo.isDownloaded) {
                downloadModel(modelInfo)
                downloadLabels(modelInfo)
                modelInfo.isDownloaded = true
            }
            true
        } catch (e: Exception) {
            Log.e(TAG, "Failed to download ${modelInfo.name}: ${e.message}")
            onError?.invoke("Failed to download ${modelInfo.name}")
            false
        }
    }
    
    /**
     * Download model file from URL
     */
    private suspend fun downloadModel(modelInfo: ModelInfo) {
        val modelFile = File(context.filesDir, modelInfo.fileName)
        
        withContext(Dispatchers.IO) {
            try {
                Log.d(TAG, "Downloading ${modelInfo.name}")
                val connection = URL(modelInfo.modelUrl).openConnection()
                connection.connect()
                
                val input = BufferedInputStream(connection.getInputStream())
                val output = FileOutputStream(modelFile)
                val buffer = ByteArray(8192)
                var bytesRead: Int
                var totalBytes = 0L
                
                while (input.read(buffer).also { bytesRead = it } != -1) {
                    output.write(buffer, 0, bytesRead)
                    totalBytes += bytesRead
                    
                    // Update progress every 100KB
                    if (totalBytes % (100 * 1024) == 0L) {
                        val mb = totalBytes / (1024.0 * 1024.0)
                        onLoadingProgressUpdate?.invoke("Downloading ${modelInfo.name}... ${String.format("%.1f", mb)} MB")
                    }
                }
                
                output.flush()
                output.close()
                input.close()
                Log.d(TAG, "${modelInfo.name} downloaded successfully")
            } catch (e: Exception) {
                Log.e(TAG, "Error downloading ${modelInfo.name}: ${e.message}")
                throw e
            }
        }
    }
    
    /**
     * Download labels file from URL
     */
    private suspend fun downloadLabels(modelInfo: ModelInfo) {
        val labelsFile = File(context.filesDir, modelInfo.labelsFileName)
        
        withContext(Dispatchers.IO) {
            try {
                Log.d(TAG, "Downloading labels")
                val labelsText = URL(modelInfo.labelsUrl).readText()
                labelsFile.writeText(labelsText)
                Log.d(TAG, "Labels downloaded successfully")
            } catch (e: Exception) {
                Log.e(TAG, "Error downloading labels: ${e.message}")
                throw e
            }
        }
    }
    
    /**
     * Load labels from file
     */
    fun loadLabelsForModel(modelInfo: ModelInfo): List<String> {
        val labelsFile = File(context.filesDir, modelInfo.labelsFileName)
        val labels = labelsFile.readLines().filter { it.isNotBlank() }
        Log.d(TAG, "Loaded ${labels.size} labels for ${modelInfo.name}")
        return labels
    }
    
    /**
     * Load model file as MappedByteBuffer
     */
    private fun loadModelFile(fileName: String): MappedByteBuffer {
        val modelFile = File(context.filesDir, fileName)
        val inputStream = FileInputStream(modelFile)
        val fileChannel = inputStream.channel
        val declaredLength = fileChannel.size()
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, 0L, declaredLength)
    }
    
    /**
     * Load and initialize a model with GPU acceleration if available
     */
    fun loadAndInitializeModel(modelInfo: ModelInfo): Boolean {
        try {
            // Check if already loaded
            if (modelInterpreters.containsKey(modelInfo.fileName)) {
                currentInterpreter = modelInterpreters[modelInfo.fileName]
                Log.d(TAG, "Model ${modelInfo.name} already loaded from cache")
                return true
            }
            
            val compatList = CompatibilityList()
            val interpreterOptions = Interpreter.Options()
            
            if(compatList.isDelegateSupportedOnThisDevice) {
                Log.d(TAG, "✅ GPU acceleration enabled")
                if (gpuDelegate == null) {
                    try {
                        gpuDelegate = GpuDelegate()
                        interpreterOptions.addDelegate(gpuDelegate)
                    } catch (e: Throwable) {
                        Log.w(TAG, "GPU delegate fallback to CPU: ${e.message}")
                    }
                } else {
                    interpreterOptions.addDelegate(gpuDelegate)
                }
            } else {
                Log.d(TAG, "⚠️ Using CPU")
            }
            
            val interpreter = Interpreter(loadModelFile(modelInfo.fileName), interpreterOptions)
            modelInterpreters[modelInfo.fileName] = interpreter
            currentInterpreter = interpreter
            
            Log.d(TAG, "✅ Loaded ${modelInfo.name}")
            onModelLoaded?.invoke(modelInfo)
            
            return true
        } catch (e: Exception) {
            Log.e(TAG, "Error loading model: ${e.message}")
            e.printStackTrace()
            onError?.invoke("Error loading model: ${e.message}")
            return false
        }
    }
    
    /**
     * Get model file for a specific model info
     */
    fun getModelFile(modelInfo: ModelInfo): File {
        return File(context.filesDir, modelInfo.fileName)
    }
    
    /**
     * Get labels file for a specific model info
     */
    fun getLabelsFile(modelInfo: ModelInfo): File {
        return File(context.filesDir, modelInfo.labelsFileName)
    }
    
    /**
     * Clean up all resources
     */
    fun cleanup() {
        modelInterpreters.values.forEach { it.close() }
        modelInterpreters.clear()
        gpuDelegate?.close()
        gpuDelegate = null
        currentInterpreter = null
    }
    
    /**
     * Get specific model info by index
     */
    fun getModelInfo(index: Int): ModelInfo? {
        return availableModels.getOrNull(index)
    }
}
