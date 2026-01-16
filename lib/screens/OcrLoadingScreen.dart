import 'dart:io';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:lottie/lottie.dart';

class OcrLoadingScreen extends StatefulWidget {
  final String username;

  const OcrLoadingScreen({Key? key, required this.username}) : super(key: key);

  @override
  State<OcrLoadingScreen> createState() => _OcrLoadingScreenState();
}

class _OcrLoadingScreenState extends State<OcrLoadingScreen> {
  bool isFrame = false;
  bool visualization = false;

  static const MethodChannel platform = MethodChannel('com.example.verseye_demo/object_detection');

  Future<void> _startOCRDetection(BuildContext context, String username) async {
    if (Platform.isAndroid) {
      try {
        await platform.invokeMethod('startOCRDetection', {
          'username': username,
        });
      } on PlatformException catch (e) {
        if (context.mounted) {
          ScaffoldMessenger.of(context).showSnackBar(
            SnackBar(
              content: Text('Failed to start OCR Detection: ${e.message}'),
            ),
          );
        }
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: SafeArea(
        child: Stack(
          children: [
            Center(
              child: Lottie.asset(
                'assets/animations/Sync_Icon.json',
                width: 200,
                height: 200,
                fit: BoxFit.fill,
                repeat: true,
              ),
            ),
            Positioned(
              top: 16,
              right: 16,
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.end,
                children: [
                  _buildToggle(
                    label: 'isFrame',
                    value: isFrame,
                    onChanged: (val) {
                      setState(() => isFrame = val);
                    },
                  ),
                  const SizedBox(height: 12),
                  _buildToggle(
                    label: 'visualization',
                    value: visualization,
                    onChanged: (val) {
                      setState(() => visualization = val);

                      /// 🔥 Start OCR when visualization is ON
                      if (val) {
                        _startOCRDetection(context, widget.username);
                      }
                    },
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildToggle({
    required String label,
    required bool value,
    required ValueChanged<bool> onChanged,
  }) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.end,
      children: [
        Text(
          label,
          style: const TextStyle(fontSize: 12, fontWeight: FontWeight.w600),
        ),
        Switch(value: value, onChanged: onChanged),
      ],
    );
  }
}
