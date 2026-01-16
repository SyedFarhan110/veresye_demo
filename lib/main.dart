import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:verseye_demo/screens/HomeScreen.dart';

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
