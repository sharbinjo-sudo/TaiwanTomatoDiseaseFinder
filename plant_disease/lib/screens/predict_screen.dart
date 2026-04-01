// ignore_for_file: unused_element

import 'dart:io';
import 'dart:typed_data';
import 'package:flutter/foundation.dart' show kIsWeb;
import 'package:flutter/material.dart';
import 'package:file_picker/file_picker.dart';
import '../widgets/navbar.dart';
import 'package:image_picker/image_picker.dart';
import '../services/api_service.dart';
import 'package:permission_handler/permission_handler.dart';
import 'dart:math';
enum PredictStage {
  upload,
  original,
  preprocess,
  segment,
  features,
  confusionMatrix,
  result,
  suggestion,
}

class PredictScreen extends StatefulWidget {
  const PredictScreen({super.key});

  @override
  State<PredictScreen> createState() => _PredictScreenState();
}

class _PredictScreenState extends State<PredictScreen> {
  late final ScrollController _horizontalScrollController;
  
@override
  void initState() {
    super.initState();
    _horizontalScrollController = ScrollController();
  }

  @override
  void dispose() {
    _horizontalScrollController.dispose();
    super.dispose();
  }
  final ImagePicker _picker = ImagePicker();
  PredictStage _stage = PredictStage.upload;
  File? _selectedFile;
  Uint8List? _webImageBytes;
  bool _loading = false;
  Map<String, dynamic>? _result;

  // ===============================
  // 🪄 Pick image
  // ===============================
  Future<void> _pickFromFile() async {
  // Only request storage permission on Android, skip on web/other platforms
  if (!kIsWeb && Platform.isAndroid) {
    var status = await Permission.storage.request();
    if (!status.isGranted) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text("Storage permission required")),
      );
      return;
    }
  }

  final result = await FilePicker.platform.pickFiles(
    type: FileType.image,
  );
  if (result == null) return;

  setState(() {
    if (kIsWeb) {
      _webImageBytes = result.files.single.bytes;
    } else {
      _selectedFile = File(result.files.single.path!);
    }
    _stage = PredictStage.original;
  });

  await _predictWithBackend();
}

Future<void> _pickFromCamera() async {
  final XFile? image =
      await _picker.pickImage(source: ImageSource.camera);

  if (image == null) return;

  if (kIsWeb) {
    _webImageBytes = await image.readAsBytes();
  } else {
    _selectedFile = File(image.path);
  }

  setState(() {
    _stage = PredictStage.original;
  });

  await _predictWithBackend();
}
Future<void> _pickFromGallery() async {
  final XFile? image =
      await _picker.pickImage(source: ImageSource.gallery);

  if (image == null) return;

  if (kIsWeb) {
    _webImageBytes = await image.readAsBytes();
  } else {
    _selectedFile = File(image.path);
  }

  setState(() {
    _stage = PredictStage.original;
  });

  await _predictWithBackend();
}

  // ===============================
  // 🧠 Call Django backend
  // ===============================
  Future<void> _predictWithBackend() async {
    setState(() => _loading = true);
    try {
      Map<String, dynamic> response;
      if (kIsWeb && _webImageBytes != null) {
        response = await ApiService.uploadImageWeb(
          _webImageBytes!,
          fileName: "leaf.jpg",
        );
      } else if (_selectedFile != null) {
        response = await ApiService.uploadImage(_selectedFile!);
      } else {
        throw Exception("No image selected");
      }

      setState(() {
        _result = response;
        _stage = PredictStage.original;
      });
    } catch (e) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text("Error connecting to backend: $e")),
      );
    } finally {
      setState(() => _loading = false);
    }
  }

  // ===============================
  // 🧱 BUILD
  // ===============================
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFFF8FAF9),
      body: Column(
        children: [
          const NavBar(activePage: "Predict"),
          if (_stage != PredictStage.upload) _buildStageSelectorBar(),
          Expanded(
            child: Center(
              child: Padding(
                padding: const EdgeInsets.all(20),
                child: _loading
                    ? const CircularProgressIndicator()
                    : SingleChildScrollView(child: _buildStageContent()),
              ),
            ),
          ),
          if (_stage != PredictStage.upload) _buildBottomNav(),
        ],
      ),
    );
  }

  // ===============================
  // 🧭 Stage Selector Dropdown
  // ===============================
  Widget _buildStageSelectorBar() {
    return Container(
      width: double.infinity,
      color: Colors.white,
      padding: const EdgeInsets.symmetric(vertical: 12, horizontal: 20),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          const Text(
            "View Stage:",
            style: TextStyle(
                fontSize: 16, fontWeight: FontWeight.w600, color: Colors.black87),
          ),
          const SizedBox(width: 10),
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 12),
            decoration: BoxDecoration(
              color: const Color(0xFFF1F8E9),
              borderRadius: BorderRadius.circular(10),
              border: Border.all(color: const Color.fromRGBO(255, 99, 71, 1)),
            ),
            child: DropdownButtonHideUnderline(
              child: DropdownButton<PredictStage>(
                value: _stage,
                icon: const Icon(Icons.arrow_drop_down, color: Color.fromRGBO(255, 99, 71, 1)),
                dropdownColor: Colors.white,
                items: PredictStage.values
                    .where((s) => s != PredictStage.upload)
                    .map(
                      (stage) => DropdownMenuItem(
                        value: stage,
                        child: Text(
                          _getStageName(stage),
                          style: const TextStyle(color: Colors.black87),
                        ),
                      ),
                    )
                    .toList(),
                onChanged: (stage) {
                  if (stage != null) setState(() => _stage = stage);
                },
              ),
            ),
          ),
        ],
      ),
    );
  }

  // ===============================
  // 🔘 Bottom Navigation
  // ===============================
  Widget _buildBottomNav() {
    return Container(
      color: Colors.white,
      padding: const EdgeInsets.symmetric(vertical: 12, horizontal: 16),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          if (_stage != PredictStage.original)
            ElevatedButton(
              onPressed: _previousStage,
              style: ElevatedButton.styleFrom(
                backgroundColor: Colors.white,
                side: BorderSide(color: const Color.fromRGBO(255, 99, 71, 1), width: 1.4),
                foregroundColor: const Color.fromRGBO(255, 99, 71, 1),
                textStyle: const TextStyle(fontWeight: FontWeight.bold),
                padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 14),
                shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(10)),
              ),
              child: const Text("Previous"),
            ),
          const SizedBox(width: 20),
          ElevatedButton(
            onPressed: _nextStage,
            style: ElevatedButton.styleFrom(
              backgroundColor: const Color.fromRGBO(255, 99, 71, 1),
              foregroundColor: Colors.white,
              textStyle: const TextStyle(fontWeight: FontWeight.bold),
              padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 14),
              shape:
                  RoundedRectangleBorder(borderRadius: BorderRadius.circular(10)),
            ),
            child: Text(_getNextButtonText()),
          ),
        ],
      ),
    );
  }

  // ===============================
  // 🧭 Stage Content
  // ===============================
  Widget _buildStageContent() {
    Widget content;
    switch (_stage) {
      case PredictStage.upload:
        content = _buildUploadStage();
        break;
      case PredictStage.original:
        content = _buildImageStage("Original Image", _result?['original_url']);
        break;
      case PredictStage.preprocess:
        content = _buildImageStage("Preprocessed Image", _result?['preprocessed_url']);
        break;
      case PredictStage.segment:
        content = _buildImageStage("Segmented Image", _result?['segmented_url']);
        break;
      case PredictStage.features:
        content = _buildFeatureStage();
        break;
      case PredictStage.confusionMatrix:
        content = _buildConfusionMatrixStage();
        break;
      case PredictStage.result:
        content = _buildResultStage();
        break;
      case PredictStage.suggestion:
        content = _buildSuggestionStage();
        break;
    }

return AnimatedSwitcher(
  duration: const Duration(milliseconds: 300),
  transitionBuilder: (child, animation) {
    return FadeTransition(opacity: animation, child: child);
  },
  child: DefaultTextStyle.merge( // <== fixes the inherit issue
    style: const TextStyle(inherit: true),
    child: Card(
      key: ValueKey(_stage),
      elevation: 3,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
      margin: const EdgeInsets.all(16),
      child: Padding(
        padding: const EdgeInsets.all(24),
        child: content,
      ),
    ),
  ),
);

  }

  // ===============================
  // 📤 Upload
  // ===============================
Widget _buildUploadStage() => Column(
  mainAxisAlignment: MainAxisAlignment.center,
  children: [
    const Icon(Icons.eco, color: Color.fromRGBO(255, 99, 71, 1), size: 48),
    const SizedBox(height: 10),
    const Text(
      "Leaf Disease Detection using Vision Transformer (ViT)",
      style: TextStyle(fontSize: 22, fontWeight: FontWeight.bold),
      textAlign: TextAlign.center,
    ),
    const SizedBox(height: 30),

    Wrap(
      spacing: 15,
      runSpacing: 15,
      alignment: WrapAlignment.center,
      children: [

        if (!kIsWeb && Platform.isAndroid)
          ElevatedButton.icon(
            onPressed: _pickFromCamera,
            icon: const Icon(Icons.camera_alt),
            label: const Text("Camera"),
            style: ElevatedButton.styleFrom(
              backgroundColor: const Color.fromRGBO(255, 99, 71, 1),
              padding: const EdgeInsets.symmetric(
                  horizontal: 22, vertical: 14),
            ),
          ),

        if (!kIsWeb && Platform.isAndroid)
          ElevatedButton.icon(
            onPressed: _pickFromGallery,
            icon: const Icon(Icons.photo),
            label: const Text("Gallery"),
            style: ElevatedButton.styleFrom(
              backgroundColor: const Color.fromRGBO(255, 99, 71, 1),
              padding: const EdgeInsets.symmetric(
                  horizontal: 22, vertical: 14),
            ),
          ),

        // ✅ Always show file upload
        ElevatedButton.icon(
          onPressed: _pickFromFile,
          icon: const Icon(Icons.upload_file),
          label: const Text("Upload File"),
          style: ElevatedButton.styleFrom(
            backgroundColor: const Color.fromRGBO(255, 99, 71, 1),
            padding: const EdgeInsets.symmetric(
                horizontal: 22, vertical: 14),
          ),
        ),
      ],
    ),
  ],
);

  // ===============================
  // 🖼️ Image Display
  // ===============================
  Widget _buildImageStage(String title, String? imageUrl) {
    return Column(
      mainAxisAlignment: MainAxisAlignment.center,
      children: [
        Text(title,
            style: const TextStyle(fontSize: 20, fontWeight: FontWeight.w600)),
        const SizedBox(height: 20),
        if (imageUrl != null)
          Image.network(imageUrl, width: 320, height: 320, fit: BoxFit.cover)
        else if (_selectedFile != null)
          Image.file(_selectedFile!, width: 320, height: 320, fit: BoxFit.cover)
        else if (_webImageBytes != null)
          Image.memory(_webImageBytes!,
              width: 320, height: 320, fit: BoxFit.cover),
      ],
    );
  }

  // ===============================
  // 📊 Features
  // ===============================
  Widget _buildFeatureStage() {
    final features = _result?['features'] ?? {};
    if (features is! Map<String, dynamic>) {
      return const Text("No features available");
    }

    return Column(
      children: [
        const Text("Extracted Features",
            style: TextStyle(fontSize: 20, fontWeight: FontWeight.w600)),
        const SizedBox(height: 20),
        SingleChildScrollView(
          scrollDirection: Axis.horizontal,
          child: DataTable(
            columns: const [
              DataColumn(label: Text("Feature")),
              DataColumn(label: Text("Value")),
            ],
            rows: List<DataRow>.from(
              features.entries.map(
                (entry) => DataRow(cells: [
                  DataCell(Text(entry.key)),
                  DataCell(Text(entry.value.toString())),
                ]),
              ),
            ),
          ),
        ),
      ],
    );
  }

// ===============================
// 🧮 Confusion Matrix + Graph + Metrics
// ===============================
Widget _buildConfusionMatrixStage() {
  final confusionData =
      _result?['confusion_matrix_data'] as Map<String, dynamic>?;

  final labels = (confusionData?['labels'] as List?)
          ?.map((e) => e.toString())
          .toList() ??
      [];

  final matrix = (confusionData?['matrix'] as List?)
          ?.map(
            (row) => (row as List)
                .map((v) => (v as num).toDouble())
                .toList(),
          )
          .toList() ??
      [];

  final metrics = confusionData?['svm_metrics'] as Map<String, dynamic>?;
  final message = confusionData?['message']?.toString();

  if (labels.isEmpty || matrix.isEmpty) {
    return Padding(
      padding: const EdgeInsets.all(20),
      child: Text(
        message ?? "No confusion matrix available.",
        style: const TextStyle(fontSize: 18, color: Colors.black54),
        textAlign: TextAlign.center,
      ),
    );
  }

  final allValues = matrix.expand((row) => row).toList();
  final minValue = allValues.isEmpty ? 0.0 : allValues.reduce(min);
  final maxValue = allValues.isEmpty ? 0.0 : allValues.reduce(max);

  const double cellSize = 100;
  const double labelWidth = 110;
  const double legendWidth = 28;

  final ScrollController verticalController = ScrollController();
  final ScrollController horizontalController = ScrollController();

  return LayoutBuilder(
    builder: (context, constraints) {
      return Center(
        child: Padding(
          padding: const EdgeInsets.symmetric(vertical: 20, horizontal: 16),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              const Text(
                "Confusion Matrix",
                style: TextStyle(
                  fontSize: 24,
                  fontWeight: FontWeight.bold,
                  color: Color(0xFF91150C),
                ),
              ),
              const SizedBox(height: 20),
              Scrollbar(
                thumbVisibility: true,
                controller: verticalController,
                child: SingleChildScrollView(
                  controller: verticalController,
                  scrollDirection: Axis.vertical,
                  child: Scrollbar(
                    thumbVisibility: true,
                    controller: horizontalController,
                    child: SingleChildScrollView(
                      controller: horizontalController,
                      scrollDirection: Axis.horizontal,
                      child: Column(
                        children: [
                          Row(
                            crossAxisAlignment: CrossAxisAlignment.center,
                            children: [
                              RotatedBox(
                                quarterTurns: 3,
                                child: const Text(
                                  "True Labels ↓",
                                  style: TextStyle(
                                    fontWeight: FontWeight.bold,
                                    fontSize: 16,
                                    color: Color(0xFFFF6347),
                                  ),
                                ),
                              ),
                              const SizedBox(width: 16),
                              Column(
                                children: [
                                  const Padding(
                                    padding: EdgeInsets.only(bottom: 12),
                                    child: Text(
                                      "Predicted Labels →",
                                      style: TextStyle(
                                        fontWeight: FontWeight.bold,
                                        fontSize: 16,
                                        color: Color(0xFFFF6347),
                                      ),
                                    ),
                                  ),
                                  Row(
                                    crossAxisAlignment: CrossAxisAlignment.center,
                                    children: [
                                      Column(
                                        children: [
                                          const SizedBox(height: 40),
                                          for (final actual in labels)
                                            SizedBox(
                                              width: labelWidth,
                                              height: cellSize,
                                              child: Align(
                                                alignment: Alignment.centerRight,
                                                child: Padding(
                                                  padding: const EdgeInsets.only(right: 10),
                                                  child: Text(
                                                    actual,
                                                    textAlign: TextAlign.right,
                                                    style: const TextStyle(
                                                      fontWeight: FontWeight.bold,
                                                      fontSize: 16,
                                                      color: Color(0xFFFF6347),
                                                    ),
                                                  ),
                                                ),
                                              ),
                                            ),
                                        ],
                                      ),
                                      Column(
                                        children: [
                                          Row(
                                            children: labels
                                                .map(
                                                  (c) => SizedBox(
                                                    width: cellSize,
                                                    height: 40,
                                                    child: Center(
                                                      child: Text(
                                                        c,
                                                        textAlign: TextAlign.center,
                                                        style: const TextStyle(
                                                          fontWeight: FontWeight.bold,
                                                          fontSize: 16,
                                                          color: Color(0xFFFF8C70),
                                                        ),
                                                      ),
                                                    ),
                                                  ),
                                                )
                                                .toList(),
                                          ),
                                          Column(
                                            children: List.generate(labels.length, (i) {
                                              return Row(
                                                children: List.generate(labels.length, (j) {
                                                  final value = matrix[i][j];
                                                  final color = _heatColor(
                                                    value,
                                                    minValue,
                                                    maxValue,
                                                  );

                                                  return Container(
                                                    width: cellSize,
                                                    height: cellSize,
                                                    alignment: Alignment.center,
                                                    decoration: BoxDecoration(
                                                      color: color,
                                                      border: Border.all(
                                                        color: Colors.white,
                                                        width: 1,
                                                      ),
                                                    ),
                                                    child: Text(
                                                      value.toStringAsFixed(0),
                                                      style: TextStyle(
                                                        fontWeight: FontWeight.bold,
                                                        fontSize: 18,
                                                        color: value > (maxValue * 0.55)
                                                            ? Colors.white
                                                            : Colors.black87,
                                                      ),
                                                    ),
                                                  );
                                                }),
                                              );
                                            }),
                                          ),
                                        ],
                                      ),
                                      const SizedBox(width: 32),
                                      Column(
                                        children: [
                                          SizedBox(
                                            height: (cellSize * labels.length) + 40,
                                            child: Stack(
                                              alignment: Alignment.center,
                                              children: [
                                                Container(
                                                  width: legendWidth,
                                                  height: cellSize * labels.length,
                                                  decoration: BoxDecoration(
                                                    borderRadius: BorderRadius.circular(8),
                                                    gradient: const LinearGradient(
                                                      begin: Alignment.bottomCenter,
                                                      end: Alignment.topCenter,
                                                      colors: [
                                                        Color(0xFFFFF1ED),
                                                        Color(0xFFFFC6B8),
                                                        Color(0xFFFF8C70),
                                                        Color(0xFFFF6347),
                                                        Color(0xFFCC3F2E),
                                                      ],
                                                    ),
                                                  ),
                                                ),
                                                Positioned.fill(
                                                  child: Column(
                                                    mainAxisAlignment:
                                                        MainAxisAlignment.spaceBetween,
                                                    children: List.generate(6, (index) {
                                                      final tickValue =
                                                          maxValue - ((maxValue / 5) * index);
                                                      return Row(
                                                        mainAxisSize: MainAxisSize.min,
                                                        children: [
                                                          Container(
                                                            width: 6,
                                                            height: 1.2,
                                                            color: Colors.black87,
                                                          ),
                                                          const SizedBox(width: 4),
                                                          Text(
                                                            tickValue
                                                                .clamp(0, maxValue)
                                                                .toStringAsFixed(0),
                                                            style: const TextStyle(
                                                              fontSize: 12,
                                                              color: Colors.black54,
                                                            ),
                                                          ),
                                                        ],
                                                      );
                                                    }),
                                                  ),
                                                ),
                                              ],
                                            ),
                                          ),
                                        ],
                                      ),
                                    ],
                                  ),
                                ],
                              ),
                            ],
                          ),
                        ],
                      ),
                    ),
                  ),
                ),
              ),
              const SizedBox(height: 40),
              if (metrics != null) ...[
                const Text(
                  "Performance Metrics",
                  style: TextStyle(
                    fontSize: 22,
                    fontWeight: FontWeight.bold,
                    color: Color(0xFFFF6347),
                  ),
                ),
                const SizedBox(height: 20),
                Wrap(
                  alignment: WrapAlignment.center,
                  spacing: 20,
                  runSpacing: 12,
                  children: metrics.entries.map((entry) {
                    return Container(
                      padding: const EdgeInsets.symmetric(
                        horizontal: 16,
                        vertical: 10,
                      ),
                      decoration: BoxDecoration(
                        color: const Color(0xFFFFF1ED),
                        borderRadius: BorderRadius.circular(10),
                        border: Border.all(
                          color: const Color(0xFFFFC6B8),
                          width: 1,
                        ),
                      ),
                      child: Text(
                        "${entry.key}: ${entry.value}",
                        style: const TextStyle(
                          fontWeight: FontWeight.w600,
                          color: Colors.black87,
                          fontSize: 16,
                        ),
                      ),
                    );
                  }).toList(),
                ),
              ],
            ],
          ),
        ),
      );
    },
  );
}

//================================
// 🧬 Prediction (Result Stage)
// ===============================
Widget _buildResultStage() {
  final label = _result?['prediction'] ?? "Unknown";
  final confidence = ((_result?['confidence'] ?? 0)).toDouble();

  return Column(
    mainAxisAlignment: MainAxisAlignment.center,
    children: [
      const Text(
        "Prediction Result",
        style: TextStyle(
          fontSize: 22,
          fontWeight: FontWeight.bold,
          color: Color.fromRGBO(255, 99, 71, 1),
        ),
      ),
      const SizedBox(height: 20),
      Text(
        label,
        style: const TextStyle(
          fontSize: 26,
          fontWeight: FontWeight.w700,
          color: Color.fromRGBO(255, 99, 71, 1),
        ),
      ),
      const SizedBox(height: 10),
      Text(
        "Confidence: ${confidence.toStringAsFixed(2)}%",
        style: const TextStyle(fontSize: 18),
      ),
      const SizedBox(height: 20),
      if (_result?['description'] != null)
        Padding(
          padding: const EdgeInsets.symmetric(horizontal: 20),
          child: Text(
            _result!['description'],
            textAlign: TextAlign.center,
            style: const TextStyle(
                fontSize: 16, color: Colors.black87, height: 1.4),
          ),
        ),
    ],
  );
}
// ===============================
//  Suggestion (Final Stage)
// ===============================
Widget _buildSuggestionStage() {
  final label = (_result?['prediction'] ?? '').toString();
final Map<String, Map<String, dynamic>> suggestions = {
  'Bacterial spot': {
    'icon': Icons.water_drop,
    'color': Color(0xFF1565C0),
    'severity': 'High',
    'cause': 'Caused by Xanthomonas vesicatoria bacteria. Spreads through rain splash, wind, and contaminated tools.',
    'symptoms': [
      'Small dark brown/black water-soaked spots on leaves',
      'Yellow halo surrounding the spots',
      'Spots may merge causing large dead areas',
      'Raised, scab-like spots on fruits',
    ],
    'treatments': [
      '🧪 Apply copper-based bactericide (Copper Oxychloride) every 7–10 days.',
      '✂️ Prune and destroy all infected leaves and stems immediately.',
      '💧 Switch to drip irrigation — avoid wetting foliage.',
      '🌱 Use certified disease-free seeds before next planting.',
      '🔄 Rotate crops — avoid tomato in the same field for 2–3 seasons.',
    ],
    'prevention': 'Use resistant varieties. Maintain proper spacing for air circulation. Disinfect tools between uses.',
  },

  'Black mold': {
    'icon': Icons.dark_mode,
    'color': Color(0xFF37474F),
    'severity': 'Medium',
    'cause': 'Caused by Alternaria alternata fungus. Thrives in warm, humid conditions and on overripe or damaged fruit.',
    'symptoms': [
      'Black velvety mold patches on leaves and stems',
      'Dark sunken lesions on fruit surface',
      'Yellowing and wilting of affected leaves',
      'Soft rot developing under mold patches on fruit',
    ],
    'treatments': [
      '🍄 Apply fungicide: Mancozeb, Iprodione, or Chlorothalonil.',
      '🗑️ Remove and bag infected plant material — do NOT compost.',
      '🌬️ Improve airflow by pruning dense canopy growth.',
      '📅 Spray preventively every 10–14 days in humid weather.',
      '🧹 Clear all debris and fallen fruit from the field regularly.',
    ],
    'prevention': 'Avoid injuring plants during cultivation. Harvest fruit on time. Reduce humidity by proper spacing.',
  },

  'Gray spot': {
    'icon': Icons.grain,
    'color': Color(0xFF546E7A),
    'severity': 'Medium',
    'cause': 'Caused by Stemphylium solani fungus. Favored by warm days, cool nights, and high humidity.',
    'symptoms': [
      'Small circular gray or tan spots with dark brown borders',
      'Spots turn gray in center as they age',
      'Heavily infected leaves turn yellow and drop',
      'Spots mostly on lower and middle leaves first',
    ],
    'treatments': [
      '🍄 Apply Chlorothalonil, Mancozeb, or Azoxystrobin fungicide.',
      '🌿 Remove infected lower leaves to limit spore spread.',
      '💧 Avoid overhead irrigation; water at the base.',
      '📅 Apply fungicide on a 7–10 day schedule during wet seasons.',
      '🧹 Destroy all infected plant debris after harvest.',
    ],
    'prevention': 'Increase plant spacing. Avoid excessive nitrogen fertilization. Use drip irrigation.',
  },

  'Late blight': {
    'icon': Icons.warning_amber_rounded,
    'color': Color(0xFFB71C1C),
    'severity': 'Very High',
    'cause': 'Caused by Phytophthora infestans (water mould). Spreads rapidly in cool, wet and humid conditions.',
    'symptoms': [
      'Greasy/water-soaked patches on leaves and stems',
      'White fuzzy mold visible on leaf undersides in humid conditions',
      'Dark brown/black lesions spreading quickly across the plant',
      'Firm brown rot on fruit with greasy appearance',
    ],
    'treatments': [
      '🚨 Act immediately — this disease can destroy the entire crop within days!',
      '🧴 Apply Metalaxyl-M, Cymoxanil, or Mancozeb systemic fungicide.',
      '🗑️ Remove and bag ALL infected material — never compost it.',
      '🌬️ Prune canopy to improve air circulation and reduce humidity.',
      '📅 Repeat fungicide application every 5–7 days in wet weather.',
    ],
    'prevention': 'Plant resistant varieties (e.g., Mountain Magic). Monitor weather forecasts. Avoid overhead irrigation.',
  },

  'health': {
    'icon': Icons.check_circle,
    'color': Color(0xFF2E7D32),
    'severity': 'None',
    'cause': 'No disease detected. Your tomato plant appears perfectly healthy!',
    'symptoms': [
      'Deep green, firm, and well-shaped leaves',
      'No discoloration, spots, or lesions observed',
      'Normal and vigorous plant growth',
    ],
    'treatments': [
      '✅ Continue regular deep watering — avoid waterlogging.',
      '🌱 Maintain balanced fertilization (N-P-K as per growth stage).',
      '🔍 Scout weekly for early signs of disease or pest activity.',
      '🌬️ Prune lower leaves periodically for good airflow.',
      '🧹 Keep field clean — remove dead leaves and fallen fruit.',
    ],
    'prevention': 'Maintain a preventive fungicide/pesticide spray schedule during high-risk (wet/humid) seasons.',
  },

  'powdery mildew': {
    'icon': Icons.blur_on,
    'color': Color(0xFF6A1B9A),
    'severity': 'Medium',
    'cause': 'Caused by Leveillula taurica or Oidium neolycopersici fungi. Favored by dry weather with moderate humidity.',
    'symptoms': [
      'White powdery coating on upper leaf surfaces',
      'Yellow patches on leaves below the white powder',
      'Leaves curl upward and eventually turn brown and drop',
      'Reduced fruit size and quality in severe infections',
    ],
    'treatments': [
      '🧴 Apply sulfur-based fungicide or Potassium bicarbonate spray.',
      '🍄 Use systemic fungicide: Myclobutanil, Trifloxystrobin, or Tebuconazole.',
      '✂️ Remove heavily infected leaves and dispose of them safely.',
      '🌿 Neem oil (5ml/L) spray is an effective organic option.',
      '📅 Repeat applications every 10–14 days until symptoms clear.',
    ],
    'prevention': 'Avoid excessive nitrogen (promotes tender growth). Ensure good air circulation. Use resistant varieties.',
  },
};
  // Direct match first, then case-insensitive
   final matchedKey = suggestions.keys.firstWhere(
  (k) => k.toLowerCase() == label.toLowerCase() ||
      label.toLowerCase().contains(k.toLowerCase()),
  orElse: () => '',
);

  final info = suggestions[matchedKey] ??
      suggestions[label] ?? {
        'icon': Icons.help_outline,
        'color': Colors.grey,
        'severity': 'Unknown',
        'cause': 'Disease information not available for: $label',
        'symptoms': ['No symptom data available'],
        'treatments': ['Please consult an agricultural expert.'],
        'prevention': 'No data available.',
      };

  final isHealthy = matchedKey == 'Tomato_healthy';
  final Color accentColor = info['color'] as Color;
  final IconData iconData = info['icon'] as IconData;
  final String severity = info['severity'] as String;

  Color severityColor = severity == 'None' || severity == 'Unknown'
      ? Colors.green
      : severity == 'Medium'
          ? Colors.orange
          : Colors.red;

  return Column(
    crossAxisAlignment: CrossAxisAlignment.start,
    children: [
      // Header
      Center(
        child: Column(
          children: [
            Icon(iconData, size: 52, color: accentColor),
            const SizedBox(height: 10),
            Text(
              isHealthy ? '🌿 Your Plant is Healthy!' : '🔬 Treatment Suggestions',
              style: TextStyle(
                fontSize: 22,
                fontWeight: FontWeight.bold,
                color: accentColor,
              ),
            ),
            const SizedBox(height: 6),
            Container(
              padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 6),
              decoration: BoxDecoration(
                color: severityColor.withOpacity(0.12),
                borderRadius: BorderRadius.circular(20),
                border: Border.all(color: severityColor),
              ),
              child: Text(
                'Severity: $severity',
                style: TextStyle(
                  fontWeight: FontWeight.bold,
                  color: severityColor,
                  fontSize: 14,
                ),
              ),
            ),
          ],
        ),
      ),
      const SizedBox(height: 24),
      const Divider(),

      // Cause
      _suggestionSection(
        icon: Icons.info_outline,
        title: 'Cause',
        color: accentColor,
        child: Text(
          info['cause'] as String,
          style: const TextStyle(fontSize: 15, height: 1.5, color: Colors.black87),
        ),
      ),
      const SizedBox(height: 16),

      // Symptoms
      _suggestionSection(
        icon: Icons.visibility,
        title: 'Symptoms to Look For',
        color: accentColor,
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: (info['symptoms'] as List<String>)
              .map((s) => Padding(
                    padding: const EdgeInsets.symmetric(vertical: 3),
                    child: Row(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Icon(Icons.circle, size: 8, color: accentColor),
                        const SizedBox(width: 8),
                        Expanded(
                          child: Text(s,
                              style: const TextStyle(fontSize: 15, height: 1.4)),
                        ),
                      ],
                    ),
                  ))
              .toList(),
        ),
      ),
      const SizedBox(height: 16),

      // Treatment Steps
      _suggestionSection(
        icon: Icons.healing,
        title: 'Recommended Actions',
        color: accentColor,
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: (info['treatments'] as List<String>)
              .asMap()
              .entries
              .map((e) => Padding(
                    padding: const EdgeInsets.symmetric(vertical: 5),
                    child: Row(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        CircleAvatar(
                          radius: 12,
                          backgroundColor: accentColor.withOpacity(0.15),
                          child: Text(
                            '${e.key + 1}',
                            style: TextStyle(
                                fontSize: 11,
                                fontWeight: FontWeight.bold,
                                color: accentColor),
                          ),
                        ),
                        const SizedBox(width: 10),
                        Expanded(
                          child: Text(e.value,
                              style: const TextStyle(fontSize: 15, height: 1.4)),
                        ),
                      ],
                    ),
                  ))
              .toList(),
        ),
      ),
      const SizedBox(height: 16),

      // Prevention
      _suggestionSection(
        icon: Icons.shield_outlined,
        title: 'Prevention Tips',
        color: accentColor,
        child: Text(
          info['prevention'] as String,
          style: const TextStyle(fontSize: 15, height: 1.5, color: Colors.black87),
        ),
      ),
      const SizedBox(height: 8),
    ],
  );
}

// Helper widget for suggestion sections
Widget _suggestionSection({
  required IconData icon,
  required String title,
  required Color color,
  required Widget child,
}) {
  return Container(
    width: double.infinity,
    padding: const EdgeInsets.all(16),
    decoration: BoxDecoration(
      color: color.withOpacity(0.05),
      borderRadius: BorderRadius.circular(12),
      border: Border.all(color: color.withOpacity(0.25)),
    ),
    child: Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Row(
          children: [
            Icon(icon, size: 20, color: color),
            const SizedBox(width: 8),
            Text(
              title,
              style: TextStyle(
                  fontSize: 16, fontWeight: FontWeight.bold, color: color),
            ),
          ],
        ),
        const SizedBox(height: 10),
        child,
      ],
    ),
  );
}
  // ===============================
  // 🎨 Helpers
  // ===============================
  String _getStageName(PredictStage s) {
    switch (s) {
      case PredictStage.original:
        return "Original";
      case PredictStage.preprocess:
        return "Preprocess";
      case PredictStage.segment:
        return "Segmentation";
      case PredictStage.features:
        return "Features";
      case PredictStage.confusionMatrix:
        return "Confusion Matrix";
      case PredictStage.result:
        return "Prediction";
      case PredictStage.suggestion:
        return "Suggestions";
      default:
        return "Upload";
    }
  }

 void _nextStage() async {
  if (_loading) return;

  // ✅ Confirmation dialog before reset
  if (_stage == PredictStage.suggestion) {
    final confirm = await showDialog<bool>(
      context: context,
      builder: (ctx) => AlertDialog(
        title: const Text("Confirm Reset"),
        content: const Text(
          "Are you sure you want to test another leaf?\n\n"
          "Your current results and graphs will be cleared.",
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.of(ctx).pop(false),
            child: const Text("Cancel"),
          ),
          ElevatedButton(
            style: ElevatedButton.styleFrom(
              backgroundColor: Color(0xFFFF6347),
              foregroundColor: Colors.white,
            ),
            onPressed: () => Navigator.of(ctx).pop(true),
            child: const Text("Yes, Reset"),
          ),
        ],
      ),
    );

    if (confirm != true) return;

    setState(() {
      _stage = PredictStage.upload;
      _selectedFile = null;
      _webImageBytes = null;
      _result = null;
    });
    return;
  }

  // Normal navigation between stages
  setState(() {
    final order = PredictStage.values;
    final nextIndex = (order.indexOf(_stage) + 1) % order.length;
    _stage = order[nextIndex];
  });
}

  void _previousStage() {
    if (_loading) return;
    setState(() {
      final order = PredictStage.values;
      final prevIndex =
          (order.indexOf(_stage) - 1) % order.length;
      _stage = order[prevIndex < 0 ? order.length - 1 : prevIndex];
    });
  }

  String _getNextButtonText() {
    switch (_stage) {
      case PredictStage.original:
        return "Next: Preprocess";
      case PredictStage.preprocess:
        return "Next: Segment";
      case PredictStage.segment:
        return "Next: Features";
      case PredictStage.features:
        return "Next: Confusion Matrix";
      case PredictStage.confusionMatrix:
        return "Next: Prediction";
      case PredictStage.result:
        return "Next: Suggestions";
      case PredictStage.suggestion:
        return "Test Another Leaf";
      default:
        return "Next";
    }
  }

  Color _getStageColor() {
    switch (_stage) {
      case PredictStage.original:
        return Colors.red;
      case PredictStage.preprocess:
        return Colors.amber;
      case PredictStage.segment:
        return Colors.cyan;
      case PredictStage.features:
        return Colors.blue;
      case PredictStage.confusionMatrix:
        return Colors.teal;
      case PredictStage.result:
        return Colors.black87;
      default:
        return Colors.green;
    }
  }

  Color _heatColor(double value, double minValue, double maxValue) {
  final denom =
      (maxValue - minValue).abs() < 1e-9 ? 1.0 : (maxValue - minValue);
  final t = ((value - minValue) / denom).clamp(0.0, 1.0);

  return Color.lerp(
    const Color(0xFFFFF1ED),
    const Color(0xFFCC3F2E),
    t,
  )!;
}

}

// ===============================
// 🎨 Heatmap Legend Widget
// ===============================
class HeatmapLegend extends StatelessWidget {
  const HeatmapLegend({
    super.key,
    required this.minValue,
    required this.maxValue,
    required this.colors,
    this.height = 180,
    this.width = 16,
  });

  final double minValue;
  final double maxValue;
  final List<Color> colors;
  final double height;
  final double width;

  @override
  Widget build(BuildContext context) {
    return Column(
      children: [
        Container(
          width: width,
          height: height,
          decoration: BoxDecoration(
            borderRadius: BorderRadius.circular(6),
            border: Border.all(color: Colors.black26, width: 0.8),
            gradient: LinearGradient(
              begin: Alignment.bottomCenter,
              end: Alignment.topCenter,
              colors: colors,
              stops: List.generate(colors.length, (i) => i / (colors.length - 1)),
            ),
          ),
        ),
        const SizedBox(height: 8),
        Text(maxValue.toStringAsFixed(0),
            style: const TextStyle(fontSize: 12, color: Colors.black87)),
        Text(minValue.toStringAsFixed(0),
            style: const TextStyle(fontSize: 12, color: Colors.black54)),
      ],
    );
  }
}
