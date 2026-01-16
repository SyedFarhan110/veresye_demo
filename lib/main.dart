import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:webview_flutter/webview_flutter.dart';
import 'package:webview_flutter_android/webview_flutter_android.dart';
import 'package:file_picker/file_picker.dart';
import 'dart:io' show Platform;
import 'ultralytics/presentation/screens/camera_inference_screen.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();

  await SystemChrome.setPreferredOrientations([
    DeviceOrientation.portraitUp,
    DeviceOrientation.portraitDown, // optional
  ]);

  // Hide status and navigation bars across the app
  SystemChrome.setEnabledSystemUIMode(SystemUiMode.immersiveSticky);

  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Verseye Demo',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: const Color(0xFF027E70)),
        useMaterial3: true,
      ),
      home: const HomeScreen(),
    );
  }
}

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
                  _startOCRDetection(context);
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

class WebViewScreen extends StatefulWidget {
  const WebViewScreen({super.key});

  @override
  State<WebViewScreen> createState() => _WebViewScreenState();
}

class _WebViewScreenState extends State<WebViewScreen>
    with SingleTickerProviderStateMixin {
  late final WebViewController controller;
  bool isLoading = true;
  bool showSplash = true;
  // Swipe-back state
  bool _isSwipingBack = false;
  double _swipeProgress = 0.0;
  double? _dragStartX;
  late AnimationController _animationController;
  late Animation<double> _pulseAnimation;
  late Animation<double> _rotateAnimation;
  int _tipIndex = 0;

  final List<String> _loadingTips = [
    "🔍 AI-powered object detection",
    "🚀 Processing your request...",
    "🧠 Neural networks at work",
    "✨ Almost there...",
    "📊 Preparing your dashboard",
  ];

  @override
  void initState() {
    super.initState();

    debugPrint('🚀 WebViewScreen initializing...');

    _animationController = AnimationController(
      duration: const Duration(seconds: 2),
      vsync: this,
    )..repeat(reverse: true);

    _pulseAnimation = Tween<double>(begin: 0.8, end: 1.2).animate(
      CurvedAnimation(parent: _animationController, curve: Curves.easeInOut),
    );

    _rotateAnimation = Tween<double>(begin: 0, end: 0.1).animate(
      CurvedAnimation(parent: _animationController, curve: Curves.easeInOut),
    );

    _startTipCycle();

    Future.delayed(const Duration(milliseconds: 2500), () {
      if (mounted && !isLoading) {
        setState(() {
          showSplash = false;
        });
      }
    });

    debugPrint('🌐 Creating WebViewController...');
    controller = WebViewController()
      ..setJavaScriptMode(JavaScriptMode.unrestricted)
      ..setNavigationDelegate(
        NavigationDelegate(
          onPageStarted: (String url) {
            debugPrint('📄 Page started loading: $url');
            setState(() {
              isLoading = true;
            });
          },
          onPageFinished: (String url) {
            debugPrint('✅ Page finished loading: $url');
            setState(() {
              isLoading = false;
            });
            Future.delayed(const Duration(milliseconds: 500), () {
              if (mounted) {
                setState(() {
                  showSplash = false;
                });
              }
            });
            _fillVideoUrl();
            Future.delayed(
              const Duration(milliseconds: 1500),
              () => _fillVideoUrl(),
            );
            Future.delayed(
              const Duration(milliseconds: 3000),
              () => _fillVideoUrl(),
            );
          },
          onWebResourceError: (error) {
            debugPrint('❌ WebView error: ${error.description}');
            debugPrint('   Error code: ${error.errorCode}');
            debugPrint('   Error type: ${error.errorType}');
          },
        ),
      );

    // Enable file upload for Android
    if (Platform.isAndroid) {
      debugPrint('📱 Platform is Android, setting up file upload...');
      _setupAndroidFileUpload();
    } else {
      debugPrint('📱 Platform is NOT Android: ${Platform.operatingSystem}');
    }

    debugPrint(
      '🔗 Loading URL: http://ai-srv.qbscocloud.net:3000/demo-dashboard',
    );
    controller.loadRequest(
      Uri.parse('http://ai-srv.qbscocloud.net:3000/demo-dashboard'),
    );
  }

  void _setupAndroidFileUpload() {
    // For Android, set the file selector on the Android-specific controller
    final platformController = controller.platform;
    if (platformController is AndroidWebViewController) {
      platformController.setOnShowFileSelector(_androidFilePicker);
      // Enable gesture navigation to allow scrolling
      platformController.setGeolocationPermissionsPromptCallbacks(
        onShowPrompt: (request) async {
          return GeolocationPermissionsResponse(allow: true, retain: true);
        },
      );
    }
  }

  Future<List<String>> _androidFilePicker(FileSelectorParams params) async {
    try {
      final result = await FilePicker.platform.pickFiles(
        type: FileType.video,
        allowMultiple: params.mode == FileSelectorMode.openMultiple,
        withData: false,
      );

      if (result != null && result.files.isNotEmpty) {
        return result.files
            .where((file) => file.identifier != null)
            .map((file) => file.identifier!) // 👈 content:// URI
            .toList();
      }
    } catch (e) {
      debugPrint('Error picking file: $e');
    }
    return <String>[];
  }

  void _fillVideoUrl() {
    debugPrint('🎥 Attempting to auto-fill video URL...');
    const videoUrl =
        'https://cdn.pixabay.com/video/2019/09/20/27091-361827476_small.mp4';

    controller
        .runJavaScript('''
    (function() {
      console.log('[Flutter] Auto-fill script running...');
      if (window.__videoUrlAutoFillInstalled) {
        console.log('[Flutter] Auto-fill already installed, skipping');
        return;
      }
      window.__videoUrlAutoFillInstalled = true;
      console.log('[Flutter] Installing auto-fill listener');

      function isVideoInput(input) {
        const placeholder = (input.placeholder || '').toLowerCase();
        return placeholder.includes('rtsp') || placeholder.includes('http');
      }

      document.addEventListener('focusin', function(e) {
        const input = e.target;
        if (!input || input.tagName !== 'INPUT') return;

        if (!isVideoInput(input)) return;

        if (input.value && input.value.trim() !== '') return;

        console.log('[Flutter] Auto-filling video input:', input);
        try {
          const setter = Object.getOwnPropertyDescriptor(
            window.HTMLInputElement.prototype,
            'value'
          ).set;
          setter.call(input, '$videoUrl');
        } catch (e) {
          input.value = '$videoUrl';
        }

        input.dispatchEvent(new Event('input', { bubbles: true }));
        input.dispatchEvent(new Event('change', { bubbles: true }));
        console.log('[Flutter] Video URL auto-filled successfully');
      }, true);
      
      console.log('[Flutter] Auto-fill listener installed successfully');
    })();
  ''')
        .then((_) {
          debugPrint('✅ Auto-fill JavaScript executed successfully');
        })
        .catchError((error) {
          debugPrint('❌ Error executing auto-fill JavaScript: $error');
        });
  }

  void _startTipCycle() {
    Future.doWhile(() async {
      await Future.delayed(const Duration(seconds: 2));
      if (mounted && showSplash) {
        setState(() {
          _tipIndex = (_tipIndex + 1) % _loadingTips.length;
        });
        return true;
      }
      return false;
    });
  }

  Future<void> _onRefresh() async {
    debugPrint('🔄 Refreshing WebView...');
    await controller.reload();
  }

  @override
  void dispose() {
    _animationController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      // appBar: AppBar(
      //   title: const Text('CV using Verseye'),
      //   backgroundColor: const Color(0xFF027E70),
      //   actions: [
      //     IconButton(
      //       icon: const Icon(Icons.refresh),
      //       tooltip: 'Reload',
      //       onPressed: () async {
      //         await controller.reload();
      //       },
      //     ),
      //   ],
      // ),
      body: Stack(
        children: [
          // WebView layer - receives all gestures by default
          WebViewWidget(controller: controller),
          // Swipe-back gesture layer - only intercepts gestures from left edge
          Positioned(
            left: 0,
            top: 0,
            bottom: 0,
            width: 32, // Only capture gestures from the left 32px edge
            child: GestureDetector(
              behavior: HitTestBehavior.opaque,
              onHorizontalDragStart: (details) {
                _dragStartX = details.globalPosition.dx;
                _isSwipingBack = true;
                setState(() {
                  _swipeProgress = 0.0;
                });
              },
              onHorizontalDragUpdate: (details) {
                if (!_isSwipingBack || _dragStartX == null) return;
                final dx = details.globalPosition.dx - _dragStartX!;
                // Only consider rightward drags
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
            ),
          ),
          // Visual feedback for swipe-back
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
          // Splash screen overlay
          if (showSplash) _buildSplashScreen(),
        ],
      ),
    );
  }

  Widget _buildSplashScreen() {
    return Container(
      color: Colors.white,
      child: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            AnimatedBuilder(
              animation: _animationController,
              builder: (context, child) {
                return Transform.scale(
                  scale: _pulseAnimation.value,
                  child: Transform.rotate(
                    angle: _rotateAnimation.value,
                    child: Container(
                      width: 120,
                      height: 120,
                      decoration: BoxDecoration(
                        shape: BoxShape.circle,
                        gradient: const LinearGradient(
                          colors: [Color(0xFF03A695), Color(0xFF027E70)],
                          begin: Alignment.topLeft,
                          end: Alignment.bottomRight,
                        ),
                        boxShadow: [
                          BoxShadow(
                            color: Color(0xFF027E70).withOpacity(0.4),
                            blurRadius: 20,
                            spreadRadius: 5,
                          ),
                        ],
                      ),
                      child: const Icon(
                        Icons.visibility,
                        size: 60,
                        color: Colors.white,
                      ),
                    ),
                  ),
                );
              },
            ),
            const SizedBox(height: 40),
            ShaderMask(
              shaderCallback: (bounds) => const LinearGradient(
                colors: [Color(0xFF03A695), Color(0xFF027E70)],
              ).createShader(bounds),
              child: const Text(
                'VERSEYE',
                style: TextStyle(
                  fontSize: 36,
                  fontWeight: FontWeight.bold,
                  color: Colors.white,
                  letterSpacing: 8,
                ),
              ),
            ),
            const SizedBox(height: 8),
            Text(
              'AI Vision Platform',
              style: TextStyle(
                fontSize: 14,
                color: Colors.grey.shade600,
                letterSpacing: 2,
              ),
            ),
            const SizedBox(height: 50),
            SizedBox(
              width: 200,
              child: LinearProgressIndicator(
                backgroundColor: Colors.grey.shade200,
                valueColor: const AlwaysStoppedAnimation<Color>(
                  Color(0xFF027E70),
                ),
                minHeight: 4,
              ),
            ),
            const SizedBox(height: 30),
            AnimatedSwitcher(
              duration: const Duration(milliseconds: 500),
              child: Text(
                _loadingTips[_tipIndex],
                key: ValueKey<int>(_tipIndex),
                style: TextStyle(fontSize: 16, color: Colors.grey.shade700),
              ),
            ),
          ],
        ),
      ),
    );
  }
}
