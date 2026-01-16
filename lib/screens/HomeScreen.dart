import 'dart:io' show Platform;

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:verseye_demo/screens/OcrMainScreen.dart';
import 'package:verseye_demo/ultralytics/presentation/screens/camera_inference_screen.dart';

import 'WebView.dart';

class HomeScreen extends StatelessWidget {
  const HomeScreen({super.key});

  static const platform = MethodChannel(
    'com.example.verseye_demo/object_detection',
  );

  Future<void> _startObjectDetection(BuildContext context) async {
    if (Platform.isAndroid) {
      try {
        await platform.invokeMethod('startObjectDetection');
      } on PlatformException catch (e) {
        if (context.mounted) {
          ScaffoldMessenger.of(context).showSnackBar(
            SnackBar(
              content: Text('Failed to start Object Detection: ${e.message}'),
              duration: const Duration(seconds: 3),
            ),
          );
        }
      }
    } else {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(
          content: Text('This feature is only available on Android'),
        ),
      );
    }
  }

  Future<void> _startOCRDetection(BuildContext context) async {
    if (Platform.isAndroid) {
      try {
        await platform.invokeMethod('startOCRDetection');
      } on PlatformException catch (e) {
        if (context.mounted) {
          ScaffoldMessenger.of(context).showSnackBar(
            SnackBar(
              content: Text('Failed to start OCR Detection: ${e.message}'),
              duration: const Duration(seconds: 3),
            ),
          );
        }
      }
    } else {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(
          content: Text('This feature is only available on Android'),
        ),
      );
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Verseye'),
        backgroundColor: Theme.of(context).colorScheme.inversePrimary,
      ),
      body: Center(
        child: Padding(
          padding: const EdgeInsets.all(20.0),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              ElevatedButton(
                onPressed: () {
                  _startObjectDetection(context);
                },
                style: ElevatedButton.styleFrom(
                  minimumSize: const Size(double.infinity, 60),
                  textStyle: const TextStyle(fontSize: 18),
                ),
                child: const Text('CV using Custom Models on Device'),
              ),
              const SizedBox(height: 20),
              ElevatedButton(
                onPressed: () {
                  Navigator.push(
                    context,
                    MaterialPageRoute(
                      builder: (context) => const OcrMainScreen(),
                    ),
                  );
                  // _startOCRDetection(context);
                },
                style: ElevatedButton.styleFrom(
                  minimumSize: const Size(double.infinity, 60),
                  textStyle: const TextStyle(fontSize: 18),
                ),
                child: const Text('OCR using Custom Model on Device'),
              ),
              const SizedBox(height: 20),
              ElevatedButton(
                onPressed: () {
                  Navigator.push(
                    context,
                    MaterialPageRoute(
                      builder: (context) => const WebViewScreen(),
                    ),
                  );
                },
                style: ElevatedButton.styleFrom(
                  minimumSize: const Size(double.infinity, 60),
                  textStyle: const TextStyle(fontSize: 18),
                ),
                child: const Text('CV using Verseye on Server'),
              ),
              const SizedBox(height: 20),
              ElevatedButton(
                onPressed: () {
                  Navigator.push(
                    context,
                    MaterialPageRoute(
                      builder: (context) => const CameraInferenceScreen(),
                    ),
                  );
                },
                style: ElevatedButton.styleFrom(
                  minimumSize: const Size(double.infinity, 60),
                  textStyle: const TextStyle(fontSize: 18),
                ),
                child: const Text('CV using Pretrained Models on Device'),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
