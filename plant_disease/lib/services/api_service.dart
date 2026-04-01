import 'dart:convert';
import 'dart:typed_data';
import 'dart:io' show File, Platform;
import 'package:flutter/foundation.dart' show kIsWeb;
import 'package:http/http.dart' as http;

class ApiService {
  static const String _apiHost = String.fromEnvironment(
    'API_HOST',
    defaultValue: '',
  );

  static String get baseUrl {
    if (_apiHost.isNotEmpty) {
      return 'http://$_apiHost:8000/api';
    }

    if (kIsWeb) {
      return 'http://127.0.0.1:8000/api';
    } else if (Platform.isAndroid) {
      return 'http://192.168.43.227:8000/api';
    } else {
      return 'http://127.0.0.1:8000/api';
    }
  }

  static Future<Map<String, dynamic>> uploadImage(File file) async {
    final uri = Uri.parse('$baseUrl/predict/');
    final request = http.MultipartRequest('POST', uri);
    request.files.add(await http.MultipartFile.fromPath('image', file.path));

    final response = await request.send();
    final res = await http.Response.fromStream(response);

    if (res.statusCode == 200) {
      return jsonDecode(res.body) as Map<String, dynamic>;
    } else {
      throw Exception('Upload failed: ${res.statusCode} - ${res.body}');
    }
  }

  static Future<Map<String, dynamic>> uploadImageWeb(
    Uint8List fileBytes, {
    required String fileName,
  }) async {
    final uri = Uri.parse('$baseUrl/predict/');
    final request = http.MultipartRequest('POST', uri);
    request.files.add(
      http.MultipartFile.fromBytes('image', fileBytes, filename: fileName),
    );

    final response = await request.send();
    final res = await http.Response.fromStream(response);

    if (res.statusCode == 200) {
      return jsonDecode(res.body) as Map<String, dynamic>;
    } else {
      throw Exception('Upload failed: ${res.statusCode} - ${res.body}');
    }
  }

  static Future<Map<String, dynamic>> upload(
    dynamic file, {
    Uint8List? bytes,
    String? fileName,
  }) async {
    if (kIsWeb) {
      if (bytes == null || fileName == null) {
        throw Exception('Web upload requires bytes and fileName');
      }
      return uploadImageWeb(bytes, fileName: fileName);
    } else {
      return uploadImage(file as File);
    }
  }

  static Future<Map<String, dynamic>> confirmLabel({
    required int recordId,
    required String trueLabel,
  }) async {
    final uri = Uri.parse('$baseUrl/predict/$recordId/confirm/');
    final res = await http.patch(
      uri,
      headers: {'Content-Type': 'application/json'},
      body: jsonEncode({'true_label': trueLabel}),
    );

    if (res.statusCode == 200) {
      return jsonDecode(res.body) as Map<String, dynamic>;
    } else {
      throw Exception('Confirm failed: ${res.statusCode} - ${res.body}');
    }
  }

  static Future<Map<String, dynamic>> fetchConfusionMatrix() async {
    final uri = Uri.parse('$baseUrl/confusion-matrix/');
    final res = await http.get(uri);

    if (res.statusCode == 200) {
      return jsonDecode(res.body) as Map<String, dynamic>;
    } else {
      throw Exception(
        'Fetch confusion matrix failed: ${res.statusCode} - ${res.body}',
      );
    }
  }
}
