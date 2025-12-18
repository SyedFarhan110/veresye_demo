package com.example.verseye_demo

import android.content.Intent
import io.flutter.embedding.android.FlutterActivity
import io.flutter.embedding.engine.FlutterEngine
import io.flutter.plugin.common.MethodChannel

class MainActivity : FlutterActivity() {
	private val CHANNEL = "com.example.verseye_demo/object_detection"

	override fun configureFlutterEngine(flutterEngine: FlutterEngine) {
		super.configureFlutterEngine(flutterEngine)
		MethodChannel(flutterEngine.dartExecutor.binaryMessenger, CHANNEL)
			.setMethodCallHandler { call, result ->
				when (call.method) {
					"startObjectDetection" -> {
						try {
							val intent = Intent(this, com.example.verseye_demo.ui.MainActivity::class.java)
							startActivity(intent)
							result.success(null)
						} catch (e: Exception) {
							result.error("ACTIVITY_ERROR", e.message, null)
						}
					}
					else -> result.notImplemented()
				}
			}
	}
}
