import 'dart:io';

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:verseye_demo/screens/OcrLoadingScreen.dart';

class OcrMainScreen extends StatelessWidget {
  const OcrMainScreen({Key? key}) : super(key: key);

  static const platform = MethodChannel(
    'com.example.verseye_demo/object_detection',
  );

  Future<void> _startOCRDetection(BuildContext context, String username) async {
    if (Platform.isAndroid) {
      try {
        await platform.invokeMethod('startOCRDetection', {
          'username': username, // pass username to native side
        });
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

  void _showUsernameDialog(BuildContext context) {
    final TextEditingController usernameController = TextEditingController();

    showDialog(
      context: context,
      builder: (context) {
        return AlertDialog(
          title: const Text('Enter Your Username'),
          content: TextField(
            controller: usernameController,
            decoration: const InputDecoration(hintText: 'Username'),
          ),
          actions: [
            TextButton(
              onPressed: () {
                Navigator.of(context).pop(); // Close dialog
              },
              child: const Text('Cancel'),
            ),
            ElevatedButton(
              onPressed: () {
                final username = usernameController.text.trim();
                if (username.isNotEmpty) {
                  Navigator.push(
                    context,
                    MaterialPageRoute(
                      builder: (_) => OcrLoadingScreen(username: username),
                    ),
                  );
                } else {
                  // Show error if username is empty
                  ScaffoldMessenger.of(context).showSnackBar(
                    const SnackBar(content: Text('Please enter a username')),
                  );
                }
              },
              child: const Text('Submit'),
            ),
          ],
        );
      },
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('OCR Main Screen')),
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            ElevatedButton(
              onPressed: () {
                // Handle past data click
                ScaffoldMessenger.of(context).showSnackBar(
                  const SnackBar(content: Text('Showing past data...')),
                );
              },
              child: const Text('Show Past Data'),
            ),
            const SizedBox(height: 20),
            ElevatedButton(
              onPressed: () => _showUsernameDialog(context),
              child: const Text('Start Fetching Text'),
            ),
          ],
        ),
      ),
    );
  }
}
