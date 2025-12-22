// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import 'package:flutter/material.dart';
import '../controllers/camera_inference_controller.dart';
import '../widgets/camera_inference_content.dart';
import '../widgets/camera_inference_overlay.dart';
import '../widgets/camera_logo_overlay.dart';
import '../widgets/camera_controls.dart';
import '../widgets/threshold_slider.dart';

/// A screen that demonstrates real-time YOLO inference using the device camera.
///
/// This screen provides:
/// - Live camera feed with YOLO object detection
/// - Model selection (detect, segment, classify, pose, obb)
/// - Adjustable thresholds (confidence, IoU, max detections)
/// - Camera controls (flip, zoom)
/// - Performance metrics (FPS)
class CameraInferenceScreen extends StatefulWidget {
  const CameraInferenceScreen({super.key});

  @override
  State<CameraInferenceScreen> createState() => _CameraInferenceScreenState();
}

class _CameraInferenceScreenState extends State<CameraInferenceScreen> {
  late final CameraInferenceController _controller;
  bool _isSwipingBack = false;
  double _swipeProgress = 0.0;
  double? _dragStartX;

  @override
  void initState() {
    super.initState();
    _controller = CameraInferenceController();
    _controller.initialize().catchError((error) {
      if (mounted) {
        _showError('Model Loading Error', error.toString());
      }
    });
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final isLandscape =
        MediaQuery.of(context).orientation == Orientation.landscape;

    return Scaffold(
      body: GestureDetector(
        behavior: HitTestBehavior.translucent,
        onHorizontalDragStart: (details) {
          _dragStartX = details.globalPosition.dx;
          _isSwipingBack = (_dragStartX ?? 0) < 32;
          if (_isSwipingBack) {
            setState(() {
              _swipeProgress = 0.0;
            });
          }
        },
        onHorizontalDragUpdate: (details) {
          if (!_isSwipingBack || _dragStartX == null) return;
          final dx = details.globalPosition.dx - _dragStartX!;
          final progress = (dx / 140).clamp(0.0, 1.0);
          setState(() {
            _swipeProgress = progress;
          });
        },
        onHorizontalDragEnd: (details) {
          if (_isSwipingBack) {
            final shouldPop =
                _swipeProgress > 0.5 ||
                (details.primaryVelocity != null &&
                    details.primaryVelocity! > 300);
            if (shouldPop && Navigator.of(context).canPop()) {
              Navigator.of(context).pop();
            }
          }
          setState(() {
            _isSwipingBack = false;
            _swipeProgress = 0.0;
            _dragStartX = null;
          });
        },
        child: ListenableBuilder(
          listenable: _controller,
          builder: (context, child) {
            return Stack(
              children: [
                CameraInferenceContent(controller: _controller),
                CameraInferenceOverlay(
                  controller: _controller,
                  isLandscape: isLandscape,
                ),
                CameraLogoOverlay(
                  controller: _controller,
                  isLandscape: isLandscape,
                ),
                CameraControls(
                  currentZoomLevel: _controller.currentZoomLevel,
                  isFrontCamera: _controller.isFrontCamera,
                  activeSlider: _controller.activeSlider,
                  onZoomChanged: _controller.setZoomLevel,
                  onSliderToggled: _controller.toggleSlider,
                  onCameraFlipped: _controller.flipCamera,
                  isLandscape: isLandscape,
                ),
                ThresholdSlider(
                  activeSlider: _controller.activeSlider,
                  confidenceThreshold: _controller.confidenceThreshold,
                  iouThreshold: _controller.iouThreshold,
                  numItemsThreshold: _controller.numItemsThreshold,
                  onValueChanged: _controller.updateSliderValue,
                  isLandscape: isLandscape,
                ),
                if (_isSwipingBack)
                  Positioned(
                    left: 16 + (_swipeProgress * 56),
                    top: MediaQuery.of(context).size.height / 2 - 28,
                    child: AnimatedOpacity(
                      opacity: _swipeProgress,
                      duration: const Duration(milliseconds: 80),
                      child: Container(
                        width: 56,
                        height: 56,
                        decoration: BoxDecoration(
                          color: Colors.black.withOpacity(0.5),
                          shape: BoxShape.circle,
                        ),
                        alignment: Alignment.center,
                        child: Icon(
                          Icons.chevron_left,
                          color: Colors.white.withOpacity(_swipeProgress),
                          size: 32,
                        ),
                      ),
                    ),
                  ),
                Positioned(
                  top: MediaQuery.of(context).padding.top + 8,
                  left: 8,
                  child: Container(
                    // decoration: BoxDecoration(
                    //   color: Colors.black.withOpacity(0.3),
                    //   shape: BoxShape.circle,
                    // ),
                    // child: IconButton(
                    //   icon: const Icon(Icons.arrow_back, color: Colors.white),
                    //   onPressed: () => Navigator.of(context).pop(),
                    // ),
                  ),
                ),
              ],
            );
          },
        ),
      ),
    );
  }

  void _showError(String title, String message) => showDialog(
    context: context,
    builder: (context) => AlertDialog(
      title: Text(title),
      content: Text(message),
      actions: [
        TextButton(
          onPressed: () => Navigator.pop(context),
          child: const Text('OK'),
        ),
      ],
    ),
  );
}
