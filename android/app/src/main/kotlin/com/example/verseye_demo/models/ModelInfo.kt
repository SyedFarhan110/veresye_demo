package com.example.verseye_demo.models

/**
 * Data class representing information about a detection model
 */
data class ModelInfo(
    val name: String,
    val modelUrl: String,
    val labelsUrl: String,
    val type: String,
    val fileName: String,
    val labelsFileName: String,
    var isDownloaded: Boolean = false
)
