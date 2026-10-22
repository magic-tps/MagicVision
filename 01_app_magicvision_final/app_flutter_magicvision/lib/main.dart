import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'dart:math' as math;

import 'package:cactus/models/types.dart'
    show
        CactusCompletionParams,
        CactusCompletionResult,
        ChatMessage,
        CompletionMode;
import 'package:cactus/services/lm.dart' show CactusLM;
// ignore: implementation_imports
import 'package:cactus/src/services/context.dart' show CactusContext;
// ignore: implementation_imports
import 'package:cactus/src/utils/models/download.dart'
    show DownloadService, DownloadTask;
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_tts/flutter_tts.dart';
import 'package:geocoding/geocoding.dart';
import 'package:geolocator/geolocator.dart';
import 'package:google_mlkit_text_recognition/google_mlkit_text_recognition.dart';
import 'package:http/http.dart' as http;
import 'package:path_provider/path_provider.dart';
import 'package:permission_handler/permission_handler.dart';
import 'package:share_plus/share_plus.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:speech_to_text/speech_recognition_result.dart';
import 'package:speech_to_text/speech_to_text.dart';
import 'package:url_launcher/url_launcher.dart';

import 'audio/audio_input_source.dart';
import 'camera/camera_source.dart';
import 'camera/usb_camera_source.dart';
import 'camera/internal_camera_source.dart';
import 'core/models/analysis_result.dart';
import 'core/services/vision_analysis_service.dart';
import 'core/services/walking_assist.dart';
import 'core/config/campus_contract.dart';

const double confidenceThreshold = 0.30;
const int nearbyDangerMeters = 4;
const Duration detectionFreshnessDuration = Duration(seconds: 2);
const Duration speechCommandCooldown = Duration(seconds: 1);
const Duration speechRestartDelay = Duration(seconds: 5);
const Duration speechAfterResponseRestartDelay = Duration(seconds: 5);
const Duration speechListenDuration = Duration(minutes: 3);
const Duration speechPauseDuration = Duration(seconds: 5);
const Duration repeatedCommandCooldown = Duration(seconds: 3);
const Duration navigationCueInterval = Duration(seconds: 12);
const double navigationArrivedMeters = 18;
const double navigationMovingAwayToleranceMeters = 9;
const double navigationGoodAccuracyMeters = 35;
const int navigationDistanceFilterMeters = 4;
const String mobileVlmBundledAssetRoot = 'assets/mobilevlm';
const String contactsPrefsKey = 'voice_contacts_v1';
const String commandHistoryPrefsKey = 'command_history_v1';
const int maxCommandHistoryEntries = 250;
const String mobileVlmPreferredModelName = 'lfm2.5-vl-450m';
const String mobileVlmLocalModelName = 'lfm2-vl-1.6b';
const String mobileVlmModelId = 'LiquidAI/LFM2.5-VL-450M';
const String mobileVlmLocalModelId = 'LiquidAI/LFM2-VL-1.6B';
const String mobileVlmEndpoint = String.fromEnvironment(
  'MAGICVISION_VLM_ENDPOINT',
);
const String mobileVlmApiKey = String.fromEnvironment(
  'MAGICVISION_VLM_API_KEY',
);
const int mobileVlmContextSize = 2048;
const int mobileVlmMaxTokens = 110;
const int mobileVlmQuantizationBytes = 8;
const Duration mobileVlmBackgroundResultWindow = Duration(seconds: 4);
const Duration mobileVlmHttpTimeout = Duration(seconds: 85);
// Cloud detections already passed the configured detector threshold.
const double sceneSpeechMinConfidence = 0.0;
const double sceneSpeechPersonMinConfidence = 0.0;

class MobileVlmBundledModelSpec {
  const MobileVlmBundledModelSpec({
    required this.fileCount,
    required this.totalBytes,
  });

  final int fileCount;
  final int totalBytes;
}

const Map<String, MobileVlmBundledModelSpec> mobileVlmBundledModelSpecs = {
  'lfm2-vl-1.6b': MobileVlmBundledModelSpec(
    fileCount: 694,
    totalBytes: 2122555588,
  ),
};

const Map<String, String> mobileVlmDirectDownloadUrls = {
  'lfm2-vl-450m':
      'https://vlqqczxwyaodtcdmdmlw.supabase.co/storage/v1/object/public/cactus-models/lfm2-vl-450m.zip',
  'lfm2-vl-1.6b':
      'https://vlqqczxwyaodtcdmdmlw.supabase.co/storage/v1/object/public/cactus-models/lfm2-vl-1.6b.zip',
};

enum AppMode { danger, reading }

enum ContactWizardStep { name, phone, emergency }

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  runApp(const YoloRealtimeApp());
}

class YoloRealtimeApp extends StatelessWidget {
  const YoloRealtimeApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      title: 'MagicVision',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(
          seedColor: const Color(0xff13b8a6),
          brightness: Brightness.dark,
        ),
        scaffoldBackgroundColor: const Color(0xff06080d),
        useMaterial3: true,
      ),
      home: const DetectorPage(),
    );
  }
}

class DetectorPage extends StatefulWidget {
  const DetectorPage({super.key});

  @override
  State<DetectorPage> createState() => _DetectorPageState();
}

class _DetectorPageState extends State<DetectorPage>
    with WidgetsBindingObserver {
  late final WalkingAssistLoop _walking;
  bool _closing = false;
  final CameraSource _cameraSource = const String.fromEnvironment('CAMERA_SOURCE', defaultValue: 'usb') == 'internal'
      ? InternalCameraSource() : UsbCameraSource();
  final AudioInputSource _audioInput = UsbMicrophoneSource();
  InferenceState? _lastAnnouncedCameraState;
  Timer? _captureDisplayTimer;
  bool _voiceReady = false;
  late final TextRecognizer _textRecognizer;
  late final MobileVlmClient _mobileVlmClient;
  late final VisionAnalysisService _visionAnalysisService;
  final SpeechToText _speech = SpeechToText();
  final FlutterTts _tts = FlutterTts();
  List<Detection> _detections = const [];
  AnalysisResult? _lastEnvironmentAnalysis;
  final List<VoiceReminder> _reminders = [];
  final List<VoiceContact> _contacts = [];
  final List<CommandHistoryEntry> _commandHistory = [];
  ContactWizardStep? _contactWizardStep;
  NavigationTarget? _navigationTarget;
  NavigationSnapshot? _navigationSnapshot;
  StreamSubscription<Position>? _navigationSubscription;
  String _status = 'Inicializando camara...';
  String _voiceStatus = 'Preparando comandos de voz';
  String _lastWords = '';
  String _ocrText = '';
  String? _lastVisualDescription;
  String _lastVisualSource = mobileVlmModelId;
  String _lastSpokenText = '';
  String _lastHandledCommand = '';
  String _pendingContactName = '';
  String _pendingContactPhone = '';
  String? _speechLocaleId;
  AppMode _mode = AppMode.danger;
  bool _isReading = false;
  bool _isDescribingScene = false;
  bool _isAnalyzingEnvironment = false;
  bool _isInstallingMobileVlm = false;
  bool _isPreparingBundledMobileVlm = false;
  bool _isWarmingMobileVlm = false;
  bool _isSpeaking = false;
  bool _speechAvailable = false;
  bool _restartSpeechPending = false;
  DateTime _ignoreSpeechUntil = DateTime.fromMillisecondsSinceEpoch(0);
  DateTime _lastCommandAt = DateTime.fromMillisecondsSinceEpoch(0);
  DateTime _lastNavigationCueAt = DateTime.fromMillisecondsSinceEpoch(0);
  int _visualAnalysisRequestId = 0;

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
    _textRecognizer = TextRecognizer(script: TextRecognitionScript.latin);
    _mobileVlmClient = MobileVlmClient(
      modelName: mobileVlmPreferredModelName,
      modelId: mobileVlmModelId,
      localModelName: mobileVlmLocalModelName,
      localModelId: mobileVlmLocalModelId,
      remoteEndpoint: mobileVlmEndpoint,
      remoteApiKey: mobileVlmApiKey,
    );
    _visionAnalysisService = VisionAnalysisService(camera: _cameraSource);
    _visionAnalysisService.addListener(_inferenceChanged);
    _walking = WalkingAssistLoop(
      intervalMs: const int.fromEnvironment(
        'WALKING_ANALYSIS_INTERVAL_MS',
        defaultValue: 1000,
      ),
      capture: () async {
        if (_isReading || _isDescribingScene || _isSpeaking) return null;
        await _analyzeEnvironmentFromCamera();
        final capture = _visionAnalysisService.lastCapture;
        return capture == null
            ? null
            : (capture.response.detections, capture.capturedAt);
      },
      available: () =>
          mounted &&
          _cameraSource.status == CameraStatus.ready &&
          _mode != AppMode.reading,
      cancel: _visionAnalysisService.cancelAnalysis,
      deliver: (hazards) async {
        if (!mounted || !_walking.running) return;
        final text = hazards.map((h) => h.message).join(' ');
        setState(() => _status = text);
        await _speak(text);
      },
      onStopped: () {
        if (mounted && !_closing) setState(() {});
      },
    );
    _start();
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    if (state != AppLifecycleState.resumed) _walking.stop();
  }

  void _toggleWalking() {
    if (_walking.running) {
      _walking.stop();
    } else {
      _walking.start();
    }
    if (mounted) setState(() {});
  }

  Future<void> _start() async {
    try {
      await _loadContacts();
      await _loadCommandHistory();
      await _setupVoiceOutput();
      _voiceReady = true;
      final microphoneStatus = await Permission.microphone.request();
      if (!mounted) return;
      if (microphoneStatus.isGranted) {
        await _setupSpeechCommands();
      } else {
        setState(() => _voiceStatus = 'Permiso de microfono denegado');
      }
      final cameraStatus = await Permission.camera.request();
      if (!mounted) return;
      if (!cameraStatus.isGranted) {
        setState(() => _status = 'Permiso de camara denegado');
        return;
      }
      await _visionAnalysisService.initialize();
      if (mounted) unawaited(_prepareMobileVlmForInstantUse());
    } catch (_) {
      if (mounted) {
        setState(
          () => _status = 'No se pudo iniciar la camara o los servicios de voz',
        );
      }
    }
  }

  void _inferenceChanged() {
    if (_walking.running && _cameraSource.status != CameraStatus.ready) {
      _walking.stop();
    }
    if (!mounted) return;
    final state = _visionAnalysisService.state;
    final message = switch (state) {
      InferenceState.cameraDisconnected => 'Camara desconectada',
      InferenceState.cameraConnecting => 'Conectando camara USB...',
      InferenceState.cameraReady =>
        'Camara conectada. Usa Que ves para analizar',
      InferenceState.capturing => 'Capturando entorno...',
      InferenceState.preparing =>
        'Preparando servicio de vision. Puedes cancelar la espera',
      InferenceState.cancelled => 'Analisis cancelado',
      InferenceState.uploading ||
      InferenceState.processing => 'Analizando entorno',
      InferenceState.resultReady => 'Analisis listo',
      InferenceState.error =>
        _visionAnalysisService.errorMessage ?? 'No se pudo analizar la imagen',
    };
    setState(() {
      _status = message;
      if (state == InferenceState.cameraDisconnected ||
          state == InferenceState.cameraConnecting ||
          state == InferenceState.capturing ||
          state == InferenceState.preparing ||
          state == InferenceState.cancelled ||
          state == InferenceState.error) {
        _detections = const [];
        _lastEnvironmentAnalysis = null;
        _visualAnalysisRequestId++;
      }
    });
    if (state != _lastAnnouncedCameraState &&
        (state == InferenceState.cameraDisconnected ||
            state == InferenceState.preparing ||
            state == InferenceState.cameraReady)) {
      if (_voiceReady) {
        unawaited(
          _speak(
            state == InferenceState.cameraReady
                ? 'Camara conectada'
                : state == InferenceState.preparing
                ? 'Preparando servicio de vision. Puedes cancelar la espera'
                : 'Camara desconectada',
          ),
        );
      }
    }
    _lastAnnouncedCameraState = state;
  }

  Future<void> _setupVoiceOutput() async {
    await _tts.awaitSpeakCompletion(true);
    if (Platform.isAndroid) {
      await _tts.setQueueMode(0);
      await _tts.setAudioAttributesForNavigation();
    }

    final preferredVoice = await _preferredSpanishVoice();
    if (preferredVoice == null) {
      await _tts.setLanguage('es-ES');
    } else {
      await _tts.setVoice(preferredVoice);
      final locale = preferredVoice['locale'];
      if (locale != null && locale.isNotEmpty) {
        await _tts.setLanguage(locale);
      }
    }
    await _tts.setSpeechRate(0.51);
    await _tts.setVolume(1);
    await _tts.setPitch(1.04);
    _tts.setStartHandler(() {
      if (mounted) setState(() => _isSpeaking = true);
    });
    _tts.setCompletionHandler(() {
      _markSpeechIdle();
    });
    _tts.setErrorHandler((_) {
      _markSpeechIdle();
    });
    _tts.setCancelHandler(() {
      _markSpeechIdle(cooldown: const Duration(milliseconds: 150));
    });
  }

  Future<Map<String, String>?> _preferredSpanishVoice() async {
    try {
      final rawVoices = await _tts.getVoices;
      if (rawVoices is! List) return null;

      final voices = rawVoices.whereType<Map>().map(_stringVoiceMap).where((
        voice,
      ) {
        final name = voice['name'] ?? '';
        final locale = voice['locale'] ?? '';
        return name.isNotEmpty && locale.toLowerCase().startsWith('es');
      }).toList();
      if (voices.isEmpty) return null;

      voices.sort(
        (a, b) => _spanishVoiceScore(b).compareTo(_spanishVoiceScore(a)),
      );
      final selected = voices.first;
      return {'name': selected['name']!, 'locale': selected['locale']!};
    } catch (_) {
      return null;
    }
  }

  void _markSpeechIdle({Duration cooldown = speechCommandCooldown}) {
    if (!mounted) return;
    _ignoreSpeechUntil = DateTime.now().add(cooldown);
    setState(() => _isSpeaking = false);
    _scheduleSpeechRestart(delay: speechAfterResponseRestartDelay);
  }

  Future<void> _setupSpeechCommands() async {
    _speechAvailable = await _speech.initialize(
      onStatus: (status) {
        if (!mounted) return;
        setState(() {
          _voiceStatus = status == 'listening'
              ? 'Di "analiza imagen", "estado modelo visual", "alto" o "SOS"'
              : 'Comandos de voz en espera';
        });
        if ((status == 'done' || status == 'notListening') && !_isSpeaking) {
          _scheduleSpeechRestart();
        }
      },
      onError: (error) {
        if (!mounted) return;
        setState(() {
          _voiceStatus = speechRecognitionStatusMessage(error.errorMsg);
        });
        if (!_isSpeaking) _scheduleSpeechRestart();
      },
    );

    if (!_speechAvailable) {
      if (!mounted) return;
      setState(() => _voiceStatus = 'Reconocimiento de voz no disponible');
      return;
    }

    _speechLocaleId = await _preferredSpeechLocaleId();
    if (mounted && _speechLocaleId != null) {
      setState(() => _voiceStatus = 'Voz lista en $_speechLocaleId');
    }
    await _listenForCommands();
  }

  Future<String?> _preferredSpeechLocaleId() async {
    try {
      final systemLocale = await _speech.systemLocale();
      final locales = await _speech.locales();
      final candidates = [
        systemLocale?.localeId,
        'es_PE',
        'es_US',
        'es_MX',
        'es_419',
        'es_ES',
        'es',
      ].whereType<String>();

      for (final candidate in candidates) {
        final match = _findSpeechLocale(locales, candidate);
        if (match != null) return match.localeId;
      }

      for (final locale in locales) {
        if (_normalizedSpeechLocale(locale.localeId).startsWith('es_')) {
          return locale.localeId;
        }
      }
      return systemLocale?.localeId;
    } catch (error) {
      debugPrint('No pude leer los idiomas de voz: $error');
      return null;
    }
  }

  LocaleName? _findSpeechLocale(List<LocaleName> locales, String wanted) {
    final normalizedWanted = _normalizedSpeechLocale(wanted);
    for (final locale in locales) {
      if (_normalizedSpeechLocale(locale.localeId) == normalizedWanted) {
        return locale;
      }
    }
    return null;
  }

  String _normalizedSpeechLocale(String localeId) {
    return localeId.toLowerCase().replaceAll('-', '_');
  }

  void _scheduleSpeechRestart({Duration delay = speechRestartDelay}) {
    if (_restartSpeechPending || !_speechAvailable || !mounted) return;
    if (_isSpeaking) return;
    _restartSpeechPending = true;
    Future<void>.delayed(delay, () async {
      _restartSpeechPending = false;
      if (!mounted || _speech.isListening || _isSpeaking) {
        return;
      }
      await _listenForCommands();
    });
  }

  Future<void> _listenForCommands() async {
    if (!_speechAvailable || _speech.isListening) {
      return;
    }
    await _audioInput.prepare();
    if (!mounted) return;
    await _speech.listen(
      localeId: _speechLocaleId,
      listenFor: speechListenDuration,
      pauseFor: speechPauseDuration,
      listenOptions: SpeechListenOptions(
        partialResults: true,
        listenMode: ListenMode.dictation,
        cancelOnError: false,
      ),
      onResult: _onSpeechResult,
    );
  }

  void _onSpeechResult(SpeechRecognitionResult result) {
    final words = result.recognizedWords.trim();
    if (words.isEmpty) return;

    final command = _normalize(words);
    if (_isAnalyzingEnvironment && isCancelAnalysisCommand(command)) {
      if (!_shouldHandleCommand('cancelar-analisis')) return;
      if (mounted) setState(() => _lastWords = words);
      _recordCommand(words, 'Cancelar analisis');
      _visionAnalysisService.cancelAnalysis();
      unawaited(_stopSpeaking());
      return;
    }
    final stopSpeechCommand = isStopSpeakingCommand(
      command,
      whileSpeaking: _isSpeaking,
    );

    if (_isSpeaking) {
      if (!stopSpeechCommand) return;
      if (mounted) setState(() => _lastWords = words);
      if (!_shouldHandleCommand('silencio:$command')) return;
      if (_contactWizardStep != null) {
        _recordCommand(words, 'Cancelar contacto');
        unawaited(_cancelContactWizard());
        return;
      }
      _recordCommand(words, 'Silencio');
      unawaited(_stopSpeaking());
      return;
    }

    if (DateTime.now().isBefore(_ignoreSpeechUntil)) return;
    if (mounted) setState(() => _lastWords = words);

    if (stopSpeechCommand) {
      if (!_shouldHandleCommand('silencio:$command')) return;
      if (_contactWizardStep != null) {
        _recordCommand(words, 'Cancelar contacto');
        unawaited(_cancelContactWizard());
        return;
      }
      _recordCommand(words, 'Silencio');
      unawaited(_stopSpeaking());
      return;
    }

    if (_contactWizardStep != null) {
      if (!result.finalResult) return;
      final step = _contactWizardStep!;
      if (!_shouldHandleCommand('contacto:${step.name}:$command')) return;
      _recordCommand(words, 'Guardar contacto por pasos');
      unawaited(_handleContactWizardInput(words, command));
      return;
    }

    final acceptsPartial = _acceptsPartialCommand(command);
    if (!result.finalResult && !acceptsPartial) return;
    if (!_shouldHandleCommand(_commandDedupeKey(command))) return;
    _recordCommand(
      words,
      result.finalResult
          ? _commandCategory(command)
          : '${_commandCategory(command)} parcial',
    );

    final handled = _handleRecognizedCommand(command);
    if (!handled && result.finalResult && mounted) {
      setState(() => _voiceStatus = 'Te escuche, pero no entendi el comando');
    }
  }

  bool _acceptsPartialCommand(String command) {
    return _isSosCommand(command) ||
        isStopSpeakingCommand(command, whileSpeaking: _isSpeaking);
  }

  String _commandDedupeKey(String command) {
    if (_isMobileVlmInstallCommand(command)) return 'mobile-vlm-install';
    if (_isMobileVlmStatusCommand(command)) return 'mobile-vlm-status';
    if (_isMobileVlmCommand(command)) return 'mobile-vlm-analyze';
    if (_isSosCommand(command)) return 'sos';
    if (command.contains('comandos') || command.contains('que puedes hacer')) {
      return 'help';
    }
    return command;
  }

  bool _handleRecognizedCommand(String command) {
    if (command.contains('activar asistencia') ||
        command.contains('modo caminar')) {
      if (!_walking.running) _toggleWalking();
      return true;
    }
    if (command.contains('detener asistencia')) {
      _walking.stop();
      return true;
    }
    if (_isAnalyzingEnvironment && isCancelAnalysisCommand(command)) {
      _visionAnalysisService.cancelAnalysis();
      return true;
    }
    if (_isSosCommand(command)) {
      _triggerSos();
      return true;
    }

    if (_isVoiceContactCreationCommand(command)) {
      _createContactFromCommand(command);
      return true;
    }

    if (_isListContactsCommand(command)) {
      _listContacts();
      return true;
    }

    if (_isListHistoryCommand(command)) {
      _listCommandHistory();
      return true;
    }

    if (_isSavedSummaryCommand(command)) {
      _speakSavedSummary();
      return true;
    }

    if (_isContactActionCommand(command)) {
      _handleContactAction(command);
      return true;
    }

    if (command.contains('comandos') || command.contains('que puedes hacer')) {
      _speakHelp();
      return true;
    }

    if (_isStopNavigationCommand(command)) {
      _stopNavigation(speak: true);
      return true;
    }

    if (_isOpenMapsNavigationCommand(command)) {
      _openCurrentRouteInMaps();
      return true;
    }

    if (_isMobileVlmInstallCommand(command)) {
      _installMobileVlmModel();
      return true;
    }

    if (_isMobileVlmStatusCommand(command)) {
      _speakMobileVlmStatus();
      return true;
    }

    if (_isMobileVlmCommand(command)) {
      _describeSceneDetections(explicitMobileVlm: true);
      return true;
    }

    if (_isCurrentLocationCommand(command)) {
      _speakCurrentLocation();
      return true;
    }

    if (_isNavigationCommand(command)) {
      _startNavigationFromCommand(command);
      return true;
    }

    if (command.contains('leer ahora') ||
        command.contains('lee ahora') ||
        command.contains('capturar') ||
        command.contains('escanea')) {
      _captureAndReadText();
      return true;
    }

    if (command.contains('leer') || command.contains('lectura')) {
      _enterReadingMode();
      return true;
    }

    if (command.contains('ver peligro') ||
        command.contains('peligro') ||
        command.contains('bache') ||
        command.contains('hueco') ||
        command.contains('grieta')) {
      _describeDangerDetections();
      return true;
    }

    if (command.contains('reconocer entorno') ||
        command.contains('entorno') ||
        command.contains('ambiente') ||
        command.contains('reconocer alrededor')) {
      _describeSceneDetections();
      return true;
    }

    if (command.contains('reconocimiento') ||
        command.contains('reconocer') ||
        command.contains('detectar')) {
      _describeSceneDetections();
      return true;
    }

    if (_hasWord(command, 'hora')) {
      _speakTime();
      return true;
    }

    if (command.contains('fecha') ||
        command.contains('dia es') ||
        command.contains('que dia')) {
      _speakDate();
      return true;
    }

    if (command.contains('repite') ||
        command.contains('repetir') ||
        command.contains('ultimo texto') ||
        command.contains('ultima lectura')) {
      _repeatLastSpeech();
      return true;
    }

    if (command.contains('que ves') ||
        command.contains('que vez') ||
        command.contains('que hay') ||
        command.contains('describe') ||
        command.contains('al frente') ||
        command.contains('objetos')) {
      _describeSceneDetections();
      return true;
    }

    if (command.contains('lista recordatorios') ||
        command.contains('mis recordatorios') ||
        command.contains('que recordatorios tengo') ||
        command.contains('recordatorios pendientes')) {
      _listReminders();
      return true;
    }

    if (command.contains('cancelar recordatorios') ||
        command.contains('borra recordatorios') ||
        command.contains('elimina recordatorios')) {
      _clearReminders();
      return true;
    }

    if (command.contains('recordatorio') ||
        command.contains('recuerdame') ||
        command.contains('recordarme')) {
      _createReminderFromCommand(command);
      return true;
    }

    if (command.contains('temporizador') ||
        command.contains('cronometro') ||
        command.contains('alarma en')) {
      _createTimerFromCommand(command);
      return true;
    }
    return false;
  }

  String _commandCategory(String command) {
    if (_isSosCommand(command)) return 'SOS';
    if (_isVoiceContactCreationCommand(command)) {
      return 'Guardar contacto por voz';
    }
    if (_isListContactsCommand(command)) return 'Lista de contactos';
    if (_isListHistoryCommand(command)) return 'Historial';
    if (_isSavedSummaryCommand(command)) return 'Resumen guardado';
    if (_isContactActionCommand(command)) return 'Contacto';
    if (_isStopNavigationCommand(command)) return 'Cancelar ruta';
    if (_isOpenMapsNavigationCommand(command)) return 'Abrir ruta externa';
    if (_isMobileVlmInstallCommand(command)) {
      return 'Preparar descripcion visual';
    }
    if (_isMobileVlmStatusCommand(command)) return 'Estado visual local';
    if (_isMobileVlmCommand(command)) return 'Descripcion visual';
    if (_isCurrentLocationCommand(command)) return 'Ubicacion';
    if (_isNavigationCommand(command)) return 'Ruta';
    if (command.contains('leer ahora') ||
        command.contains('lee ahora') ||
        command.contains('capturar') ||
        command.contains('escanea')) {
      return 'Lectura OCR';
    }
    if (command.contains('leer') || command.contains('lectura')) {
      return 'Preparar lectura';
    }
    if (command.contains('recordatorio') ||
        command.contains('recuerdame') ||
        command.contains('recordarme')) {
      return 'Recordatorio';
    }
    if (command.contains('temporizador') ||
        command.contains('cronometro') ||
        command.contains('alarma en')) {
      return 'Temporizador';
    }
    if (_hasWord(command, 'hora')) return 'Hora';
    if (command.contains('fecha') ||
        command.contains('dia es') ||
        command.contains('que dia')) {
      return 'Fecha';
    }
    if (command.contains('que ves') ||
        command.contains('que hay') ||
        command.contains('describe') ||
        command.contains('entorno') ||
        command.contains('ambiente') ||
        command.contains('peligro') ||
        command.contains('bache') ||
        command.contains('hueco') ||
        command.contains('grieta')) {
      return command.contains('peligro') ||
              command.contains('bache') ||
              command.contains('hueco') ||
              command.contains('grieta')
          ? 'Peligros'
          : 'Escena';
    }
    if (command.contains('comandos') || command.contains('que puedes hacer')) {
      return 'Ayuda';
    }
    return 'Comando';
  }

  bool _shouldHandleCommand(String command) {
    final now = DateTime.now();
    if (command == _lastHandledCommand &&
        now.difference(_lastCommandAt) < repeatedCommandCooldown) {
      return false;
    }
    _lastHandledCommand = command;
    _lastCommandAt = now;
    return true;
  }

  String _normalize(String value) {
    return normalizeSpeechText(value);
  }

  bool _hasWord(String command, String word) {
    return hasSpeechWord(command, word);
  }

  bool _isSosCommand(String command) {
    return command == 'sos' ||
        command == 'ayuda' ||
        command.contains('emergencia') ||
        command.contains('auxilio') ||
        command.contains('necesito ayuda');
  }

  bool _isContactActionCommand(String command) {
    return command.contains('llamar') ||
        command.contains('llama') ||
        command.contains('llamada') ||
        command.contains('whatsapp') ||
        command.contains('wsp') ||
        command.contains('guasap') ||
        command.contains('mensaje');
  }

  bool _isVoiceContactCreationCommand(String command) {
    return command.contains('agrega contacto') ||
        command.contains('agregar contacto') ||
        command.contains('guarda contacto') ||
        command.contains('guardar contacto') ||
        command.contains('registra contacto') ||
        command.contains('registrar contacto');
  }

  bool _isListContactsCommand(String command) {
    return command.contains('mis contactos') ||
        command.contains('lista contactos') ||
        command.contains('lista de contactos') ||
        command.contains('contactos guardados') ||
        command.contains('que contactos tengo') ||
        command.contains('contacto sos') ||
        command.contains('quien es mi sos');
  }

  bool _isListHistoryCommand(String command) {
    return command.contains('historial') ||
        command.contains('ultimos comandos') ||
        command.contains('comandos usados') ||
        command.contains('que comandos use');
  }

  bool _isSavedSummaryCommand(String command) {
    return command.contains('que tengo guardado') ||
        command.contains('que hay guardado') ||
        command.contains('resumen guardado') ||
        command.contains('mis datos guardados');
  }

  bool _isCurrentLocationCommand(String command) {
    return command.contains('donde estoy') ||
        command.contains('mi ubicacion') ||
        command.contains('ubicacion actual') ||
        command.contains('en que lugar estoy') ||
        command.contains('dime mi ubicacion');
  }

  bool _isNavigationCommand(String command) {
    return command.contains('quiero ir a') ||
        command.contains('quiero llegar a') ||
        command.contains('guiame a') ||
        command.contains('guia me a') ||
        command.contains('llevame a') ||
        command.contains('como llegar a') ||
        command.contains('ruta a') ||
        command.contains('ir a ');
  }

  bool _isStopNavigationCommand(String command) {
    return command.contains('cancelar ruta') ||
        command.contains('detener ruta') ||
        command.contains('parar ruta') ||
        command.contains('cancelar guia') ||
        command.contains('detener guia') ||
        command.contains('dejar de guiar');
  }

  bool _isOpenMapsNavigationCommand(String command) {
    return command.contains('abrir maps') ||
        command.contains('abrir mapa') ||
        command.contains('ruta en maps') ||
        command.contains('ruta en google') ||
        command.contains('navegacion externa') ||
        command.contains('abrir navegacion');
  }

  bool _isMobileVlmCommand(String command) {
    return isMobileVlmAnalyzeCommandForSpeech(command);
  }

  bool _isMobileVlmInstallCommand(String command) {
    return isMobileVlmInstallCommandForSpeech(command);
  }

  bool _isMobileVlmStatusCommand(String command) {
    return isMobileVlmStatusCommandForSpeech(command);
  }

  Future<void> _loadContacts() async {
    final prefs = await SharedPreferences.getInstance();
    final raw = prefs.getString(contactsPrefsKey);
    if (raw == null || raw.isEmpty) return;
    try {
      final values = jsonDecode(raw);
      if (values is! List) return;
      final loaded = values
          .whereType<Map<String, dynamic>>()
          .map(VoiceContact.fromJson)
          .where((contact) => contact.phone.trim().isNotEmpty)
          .toList();
      if (!mounted) return;
      setState(() {
        _contacts
          ..clear()
          ..addAll(loaded);
      });
    } catch (_) {
      // Ignore damaged saved data and let the user register contacts again.
    }
  }

  Future<void> _saveContacts() async {
    final prefs = await SharedPreferences.getInstance();
    final raw = jsonEncode(
      _contacts.map((contact) => contact.toJson()).toList(),
    );
    await prefs.setString(contactsPrefsKey, raw);
  }

  Future<void> _loadCommandHistory() async {
    final prefs = await SharedPreferences.getInstance();
    final raw = prefs.getString(commandHistoryPrefsKey);
    if (raw == null || raw.isEmpty) return;
    try {
      final values = jsonDecode(raw);
      if (values is! List) return;
      final loaded = values
          .whereType<Map<String, dynamic>>()
          .map(CommandHistoryEntry.fromJson)
          .toList();
      if (!mounted) return;
      setState(() {
        _commandHistory
          ..clear()
          ..addAll(loaded);
      });
    } catch (_) {
      // Ignore damaged saved history.
    }
  }

  Future<void> _saveCommandHistory() async {
    final prefs = await SharedPreferences.getInstance();
    final raw = jsonEncode(
      _commandHistory.map((entry) => entry.toJson()).toList(),
    );
    await prefs.setString(commandHistoryPrefsKey, raw);
  }

  void _recordCommand(String spokenText, String action) {
    final entry = CommandHistoryEntry(
      spokenText: spokenText.trim(),
      action: action,
      at: DateTime.now(),
    );
    if (mounted) {
      setState(() {
        _commandHistory.insert(0, entry);
        if (_commandHistory.length > maxCommandHistoryEntries) {
          _commandHistory.removeRange(
            maxCommandHistoryEntries,
            _commandHistory.length,
          );
        }
      });
    } else {
      _commandHistory.insert(0, entry);
    }
    unawaited(_saveCommandHistory());
  }

  void _recordEnvironmentAnalysis(AnalysisResult result) {
    final entry = CommandHistoryEntry(
      spokenText: result.displayText,
      action: 'Analisis de entorno (${result.source})',
      at: result.timestamp,
      type: 'environment',
      description: result.description,
      riskLevel: result.riskLevel,
      detectedObjects: result.detectedObjects,
      recommendation: result.recommendation,
      source: result.source,
    );
    if (mounted) {
      setState(() {
        _commandHistory.insert(0, entry);
        if (_commandHistory.length > maxCommandHistoryEntries) {
          _commandHistory.removeRange(
            maxCommandHistoryEntries,
            _commandHistory.length,
          );
        }
      });
    } else {
      _commandHistory.insert(0, entry);
    }
    unawaited(_saveCommandHistory());
  }

  Future<void> _clearCommandHistory() async {
    setState(() => _commandHistory.clear());
    await _saveCommandHistory();
  }

  Future<void> _exportCommandHistory() async {
    if (_commandHistory.isEmpty) {
      await _speak('No hay historial de comandos para descargar.');
      return;
    }
    final directory = await getApplicationDocumentsDirectory();
    final file = File(
      '${directory.path}/magicvision_historial_${DateTime.now().millisecondsSinceEpoch}.txt',
    );
    await file.writeAsString(_commandHistoryAsText(), flush: true);
    await SharePlus.instance.share(
      ShareParams(
        files: [XFile(file.path)],
        text: 'Historial de uso de MagicVision',
        subject: 'Historial MagicVision',
      ),
    );
  }

  String _commandHistoryAsText() {
    final buffer = StringBuffer('Historial de uso - MagicVision\n\n');
    for (final entry in _commandHistory) {
      buffer.writeln('${formatDateTimeForHistory(entry.at)} | ${entry.action}');
      if (entry.type == 'environment') {
        buffer.writeln('Tipo: environment');
        buffer.writeln('Descripcion: ${entry.description}');
        buffer.writeln('Riesgo: ${entry.riskLevel}');
        buffer.writeln('Objetos: ${entry.detectedObjects.join(', ')}');
        buffer.writeln('Recomendacion: ${entry.recommendation}');
        buffer.writeln('Fuente: ${entry.source}');
      } else {
        buffer.writeln('Comando: ${entry.spokenText}');
      }
      buffer.writeln();
    }
    return buffer.toString();
  }

  Future<void> _addContact({
    required String name,
    required String phone,
    required bool emergency,
  }) async {
    final cleanName = name.trim();
    final cleanPhone = phone.trim();
    if (cleanName.isEmpty || cleanPhone.isEmpty) return;
    setState(() {
      if (emergency) {
        for (final contact in _contacts) {
          contact.isEmergency = false;
        }
      }
      _contacts.add(
        VoiceContact(
          name: cleanName,
          phone: cleanPhone,
          aliases: _contactAliases(cleanName),
          isEmergency: emergency || _contacts.isEmpty,
        ),
      );
    });
    await _saveContacts();
  }

  Future<void> _startContactWizard() async {
    if (!mounted) return;
    setState(() {
      _contactWizardStep = ContactWizardStep.name;
      _pendingContactName = '';
      _pendingContactPhone = '';
      _voiceStatus = 'Guardando contacto: esperando nombre';
    });
    await _speak('Listo. Cual es el nombre del contacto?');
  }

  Future<void> _cancelContactWizard() async {
    if (!mounted) return;
    setState(() {
      _contactWizardStep = null;
      _pendingContactName = '';
      _pendingContactPhone = '';
      _voiceStatus = 'Guardado de contacto cancelado';
    });
    await _stopSpeaking();
    await _speak('Cancele el guardado del contacto.');
  }

  Future<void> _handleContactWizardInput(
    String spokenText,
    String command,
  ) async {
    final step = _contactWizardStep;
    if (step == null) return;

    switch (step) {
      case ContactWizardStep.name:
        final name = contactNameFromWizardSpeech(spokenText);
        if (name.isEmpty) {
          await _speak(
            'No escuche el nombre. Dime solo el nombre del contacto.',
          );
          return;
        }
        if (!mounted) return;
        setState(() {
          _pendingContactName = name;
          _contactWizardStep = ContactWizardStep.phone;
          _voiceStatus = 'Guardando contacto: esperando numero';
        });
        await _speak(
          'Perfecto, $name. Ahora dime los nueve numeros del celular peruano, uno por uno. Yo agregare el mas cincuenta y uno.',
        );
        return;

      case ContactWizardStep.phone:
        final phone = peruMobilePhoneFromSpeech(command);
        if (phone == null) {
          await _speak(
            'No escuche un celular peruano valido. Dime nueve numeros, empezando con nueve. Por ejemplo: nueve ocho siete seis cinco cuatro tres dos uno.',
          );
          return;
        }
        if (!mounted) return;
        setState(() {
          _pendingContactPhone = phone;
          _contactWizardStep = ContactWizardStep.emergency;
          _voiceStatus = 'Guardando contacto: esperando confirmacion SOS';
        });
        await _speak(
          'Escuche ${formatLongNumbersForSpeech(phone)}. Quieres guardarlo como contacto SOS? Di si o no.',
        );
        return;

      case ContactWizardStep.emergency:
        final emergency = yesNoFromSpeech(command);
        if (emergency == null) {
          await _speak('Dime si o no. Este contacto sera SOS?');
          return;
        }
        final name = _pendingContactName;
        final phone = _pendingContactPhone;
        if (name.isEmpty || phone.isEmpty) {
          await _startContactWizard();
          return;
        }
        if (!mounted) return;
        setState(() {
          _contactWizardStep = null;
          _pendingContactName = '';
          _pendingContactPhone = '';
          _voiceStatus = 'Contacto guardado por voz';
        });
        await _addContact(name: name, phone: phone, emergency: emergency);
        final sosText = emergency ? ' Lo marque como contacto SOS.' : '';
        await _speak(
          'Listo. Guarde a $name, numero ${formatLongNumbersForSpeech(phone)}.$sosText',
        );
        return;
    }
  }

  Future<void> _createContactFromCommand(String command) async {
    final parsed = parseVoiceContactCommand(command);
    if (parsed == null) {
      await _startContactWizard();
      return;
    }
    await _addContact(
      name: parsed.name,
      phone: parsed.phone,
      emergency: parsed.isEmergency,
    );
    await _speak(
      'Contacto guardado: ${parsed.name}, numero ${formatLongNumbersForSpeech(parsed.phone)}.',
    );
  }

  Future<void> _listContacts() async {
    if (_contacts.isEmpty) {
      await _speak('No tienes contactos guardados.');
      return;
    }
    final emergency = _emergencyContact();
    final contactsText = _contacts
        .map((contact) {
          final sosText = contact.isEmergency ? ', contacto SOS' : '';
          return '${contact.name}, numero ${formatLongNumbersForSpeech(contact.phone)}$sosText';
        })
        .join('. ');
    final prefix = emergency == null
        ? 'Tus contactos son: '
        : 'Tu contacto SOS es ${emergency.name}. Tus contactos son: ';
    await _speak('$prefix$contactsText.');
  }

  Future<void> _listCommandHistory() async {
    if (_commandHistory.isEmpty) {
      await _speak('Aun no hay historial de comandos.');
      return;
    }
    final latest = _commandHistory
        .take(5)
        .map((entry) {
          return '${entry.action}: ${entry.spokenText}';
        })
        .join('. ');
    await _speak('Ultimos comandos: $latest.');
  }

  Future<void> _speakSavedSummary() async {
    final contactCount = _contacts.length;
    final reminderCount = _reminders.length;
    final historyCount = _commandHistory.length;
    final emergency = _emergencyContact();
    final sosText = emergency == null
        ? 'No tienes contacto SOS.'
        : 'Tu contacto SOS es ${emergency.name}.';
    await _speak(
      'Tienes $contactCount contacto${contactCount == 1 ? '' : 's'}, '
      '$reminderCount recordatorio${reminderCount == 1 ? '' : 's'} pendiente${reminderCount == 1 ? '' : 's'}, '
      'y $historyCount comando${historyCount == 1 ? '' : 's'} en el historial. $sosText',
    );
  }

  Future<void> _deleteContact(VoiceContact contact) async {
    setState(() => _contacts.remove(contact));
    await _saveContacts();
  }

  Future<void> _setEmergencyContact(VoiceContact contact) async {
    setState(() {
      for (final item in _contacts) {
        item.isEmergency = identical(item, contact);
      }
    });
    await _saveContacts();
  }

  List<String> _contactAliases(String name) {
    final normalized = _normalize(name);
    final aliases = <String>{normalized};
    if (normalized.contains('mama') || normalized.contains('madre')) {
      aliases.addAll(['mama', 'mi mama', 'madre', 'mi madre']);
    }
    if (normalized.contains('papa') || normalized.contains('padre')) {
      aliases.addAll(['papa', 'mi papa', 'padre', 'mi padre']);
    }
    return aliases.toList();
  }

  VoiceContact? _findContactInCommand(String command) {
    for (final contact in _contacts) {
      if (contact.matches(command)) return contact;
    }
    return null;
  }

  VoiceContact? _emergencyContact() {
    for (final contact in _contacts) {
      if (contact.isEmergency) return contact;
    }
    return _contacts.isEmpty ? null : _contacts.first;
  }

  Future<void> _handleContactAction(String command) async {
    if (_contacts.isEmpty) {
      await _speak('Primero registra un contacto desde el boton Contactos.');
      return;
    }
    final contact = _findContactInCommand(command);
    if (contact == null) {
      await _speak(
        'No encontre ese contacto. Puedes registrarlo o decir su nombre guardado.',
      );
      return;
    }
    final wantsWhatsApp =
        command.contains('whatsapp') ||
        command.contains('wsp') ||
        command.contains('guasap') ||
        command.contains('mensaje');
    if (wantsWhatsApp) {
      await _openWhatsApp(contact, message: 'Hola ${contact.name}.');
      return;
    }
    await _openPhoneCall(contact);
  }

  Future<void> _openPhoneCall(VoiceContact contact) async {
    final uri = Uri(scheme: 'tel', path: contact.phone);
    await _openExternal(uri, 'No pude abrir la llamada para ${contact.name}.');
  }

  Future<void> _openSms(VoiceContact contact, String message) async {
    final uri = Uri(
      scheme: 'sms',
      path: contact.phone,
      queryParameters: {'body': message},
    );
    await _openExternal(
      uri,
      'No pude abrir el mensaje SMS para ${contact.name}.',
    );
  }

  Future<void> _openWhatsApp(
    VoiceContact contact, {
    required String message,
  }) async {
    final phone = contact.phone.replaceAll(RegExp(r'[^0-9]'), '');
    final uri = Uri.https('wa.me', '/$phone', {'text': message});
    final opened = await launchUrl(uri, mode: LaunchMode.externalApplication);
    if (!opened) {
      await _openSms(contact, message);
    }
  }

  Future<void> _openExternal(Uri uri, String failureMessage) async {
    final opened = await launchUrl(uri, mode: LaunchMode.externalApplication);
    if (!opened) await _speak(failureMessage);
  }

  Future<Position?> _currentPosition({bool navigation = false}) async {
    var permission = await Geolocator.checkPermission();
    if (permission == LocationPermission.denied) {
      permission = await Geolocator.requestPermission();
    }
    if (permission == LocationPermission.denied ||
        permission == LocationPermission.deniedForever) {
      return null;
    }
    final serviceEnabled = await Geolocator.isLocationServiceEnabled();
    if (!serviceEnabled) return null;

    Position? cached;
    try {
      cached = await Geolocator.getLastKnownPosition();
    } catch (_) {
      cached = null;
    }

    try {
      final fresh = await Geolocator.getCurrentPosition(
        locationSettings: LocationSettings(
          accuracy: navigation
              ? LocationAccuracy.bestForNavigation
              : LocationAccuracy.best,
          timeLimit: navigation
              ? const Duration(seconds: 12)
              : const Duration(seconds: 9),
        ),
      );
      return chooseBestPosition(
        fresh: fresh,
        cached: cached,
        now: DateTime.now(),
        navigation: navigation,
      );
    } catch (_) {
      return chooseBestPosition(
        fresh: null,
        cached: cached,
        now: DateTime.now(),
        navigation: navigation,
      );
    }
  }

  Future<Placemark?> _placemarkForPosition(Position position) async {
    try {
      final places = await placemarkFromCoordinates(
        position.latitude,
        position.longitude,
      );
      return places.firstOrNull;
    } catch (_) {
      return null;
    }
  }

  Future<String> _addressForPosition(Position position) async {
    final place = await _placemarkForPosition(position);
    return place == null ? '' : addressFromPlacemark(place);
  }

  Future<void> _speakCurrentLocation() async {
    if (mounted) {
      setState(() => _status = 'Afinando ubicacion GPS...');
    }
    await _speak('Dame un momento. Estoy afinando tu ubicacion.');
    final position = await _currentPosition(navigation: true);
    if (position == null) {
      await _speak(
        'No pude obtener tu ubicacion. Revisa que el GPS y el permiso de ubicacion esten activos.',
      );
      return;
    }
    final address = await _addressForPosition(position);
    if (mounted) {
      setState(() => _status = 'Ubicacion actual lista');
    }
    await _speak(currentLocationSpeech(position: position, address: address));
  }

  Future<void> _startNavigationFromCommand(String command) async {
    final destination = _navigationDestinationFromCommand(command);
    if (destination == null || destination.trim().length < 3) {
      await _speak('Dime el destino. Por ejemplo: quiero ir a la farmacia.');
      return;
    }
    await _startInternalNavigation(destination);
  }

  String? _navigationDestinationFromCommand(String command) {
    final patterns = [
      RegExp(r'\bquiero ir a\s+(.+)$'),
      RegExp(r'\bquiero llegar a\s+(.+)$'),
      RegExp(r'\bguiame a\s+(.+)$'),
      RegExp(r'\bguia me a\s+(.+)$'),
      RegExp(r'\bllevame a\s+(.+)$'),
      RegExp(r'\bcomo llegar a\s+(.+)$'),
      RegExp(r'\bruta a\s+(.+)$'),
      RegExp(r'\bir a\s+(.+)$'),
    ];
    for (final pattern in patterns) {
      final match = pattern.firstMatch(command);
      if (match != null) return match.group(1)?.trim();
    }
    return null;
  }

  Future<void> _startInternalNavigation(String destination) async {
    final cleanDestination = cleanNavigationDestination(destination);
    await _speak('Estoy ubicandote y buscando $cleanDestination cerca de ti.');
    if (mounted) {
      setState(() => _status = 'Buscando ruta a $cleanDestination...');
    }

    final current = await _currentPosition(navigation: true);
    if (current == null) {
      await _speak(
        'No pude obtener tu ubicacion. Activa el GPS y el permiso de ubicacion.',
      );
      return;
    }

    final originPlace = await _placemarkForPosition(current);
    final target = await _resolveNavigationTarget(
      cleanDestination,
      near: originPlace,
    );
    if (target == null) {
      await _speak(
        'No pude encontrar ese destino con suficiente claridad. Prueba con un nombre mas especifico, por ejemplo farmacia central con distrito, o una avenida con numero.',
      );
      return;
    }

    await _navigationSubscription?.cancel();
    _navigationTarget = target;
    _navigationSnapshot = _buildNavigationSnapshot(current, target);
    _lastNavigationCueAt = DateTime.fromMillisecondsSinceEpoch(0);
    if (mounted) {
      setState(() => _status = 'Guiando a ${target.name}');
    }
    await _speakNavigationCue(current, force: true, opening: true);
    _navigationSubscription =
        Geolocator.getPositionStream(
          locationSettings: const LocationSettings(
            accuracy: LocationAccuracy.bestForNavigation,
            distanceFilter: navigationDistanceFilterMeters,
          ),
        ).listen(
          (position) => _speakNavigationCue(position),
          onError: (_) {
            if (!mounted) return;
            setState(() => _status = 'Esperando mejor senal GPS...');
          },
        );
  }

  Future<NavigationTarget?> _resolveNavigationTarget(
    String destination, {
    Placemark? near,
  }) async {
    final queries = navigationSearchQueries(destination, near: near);
    for (final query in queries) {
      try {
        final locations = await locationFromAddress(query);
        if (locations.isEmpty) continue;
        final location = locations.first;
        final targetAddress = await _addressForCoordinates(
          location.latitude,
          location.longitude,
        );
        return NavigationTarget(
          name: destination,
          latitude: location.latitude,
          longitude: location.longitude,
          address: targetAddress,
        );
      } catch (_) {
        continue;
      }
    }
    return null;
  }

  Future<String?> _addressForCoordinates(
    double latitude,
    double longitude,
  ) async {
    try {
      final places = await placemarkFromCoordinates(latitude, longitude);
      final address = places.isEmpty ? '' : addressFromPlacemark(places.first);
      return address.isEmpty ? null : address;
    } catch (_) {
      return null;
    }
  }

  Future<void> _speakNavigationCue(
    Position position, {
    bool force = false,
    bool opening = false,
  }) async {
    final target = _navigationTarget;
    if (target == null) return;
    final now = DateTime.now();
    if (!force &&
        now.difference(_lastNavigationCueAt) < navigationCueInterval) {
      return;
    }
    _lastNavigationCueAt = now;
    final snapshot = _buildNavigationSnapshot(position, target);
    _navigationSnapshot = snapshot;
    if (mounted) {
      setState(
        () => _status =
            '${navigationDistanceCompact(snapshot.distanceMeters)} hasta ${target.name}',
      );
    }
    final distance = snapshot.distanceMeters;
    if (distance <= navigationArrivedMeters) {
      await _speak(
        'Llegaste a la zona de ${target.name}. Estoy deteniendo la guia.',
      );
      await _stopNavigation();
      return;
    }
    await _speak(
      navigationCueForSpeech(
        targetName: target.name,
        direction: snapshot.direction,
        distanceMeters: snapshot.distanceMeters,
        accuracyMeters: snapshot.accuracyMeters,
        movingAway: snapshot.movingAway,
        opening: opening,
      ),
    );
  }

  NavigationSnapshot _buildNavigationSnapshot(
    Position position,
    NavigationTarget target,
  ) {
    final previousDistance = _navigationSnapshot?.distanceMeters;
    final distance = Geolocator.distanceBetween(
      position.latitude,
      position.longitude,
      target.latitude,
      target.longitude,
    );
    final bearing = Geolocator.bearingBetween(
      position.latitude,
      position.longitude,
      target.latitude,
      target.longitude,
    );
    return NavigationSnapshot(
      targetName: target.name,
      targetAddress: target.address,
      distanceMeters: distance,
      direction: _navigationDirectionForSpeech(bearing, position),
      bearing: bearing,
      accuracyMeters: normalizedAccuracyMeters(position),
      movingAway:
          previousDistance != null &&
          distance > previousDistance + navigationMovingAwayToleranceMeters,
      updatedAt: DateTime.now(),
    );
  }

  String _navigationDirectionForSpeech(double bearing, Position position) {
    final heading = position.heading;
    if (isReliableHeading(position)) {
      final diff = ((bearing - heading + 540) % 360) - 180;
      if (diff.abs() < 25) return 'de frente';
      if (diff >= 25 && diff < 75) return 'un poco a la derecha';
      if (diff <= -25 && diff > -75) return 'un poco a la izquierda';
      if (diff >= 75 && diff < 135) return 'hacia la derecha';
      if (diff <= -75 && diff > -135) return 'hacia la izquierda';
      return 'en direccion contraria';
    }
    return 'hacia el ${_cardinalDirection(bearing)}';
  }

  String _cardinalDirection(double bearing) {
    const directions = [
      'norte',
      'noreste',
      'este',
      'sureste',
      'sur',
      'suroeste',
      'oeste',
      'noroeste',
    ];
    final index = ((bearing % 360) / 45).round() % directions.length;
    return directions[index];
  }

  Future<void> _stopNavigation({bool speak = false}) async {
    await _navigationSubscription?.cancel();
    _navigationSubscription = null;
    _navigationTarget = null;
    _navigationSnapshot = null;
    if (mounted) {
      setState(() => _status = 'Escaneando peligros del piso');
    }
    if (speak) await _speak('Ruta cancelada.');
  }

  Future<void> _openCurrentRouteInMaps() async {
    final target = _navigationTarget;
    if (target == null) {
      await _speak(
        'Todavia no hay una ruta activa. Di, por ejemplo, quiero ir a la farmacia.',
      );
      return;
    }

    final current = await _currentPosition(navigation: true);
    final queryParameters = <String, String>{
      'api': '1',
      'destination': '${target.latitude},${target.longitude}',
      'travelmode': 'walking',
    };
    if (current != null) {
      queryParameters['origin'] = '${current.latitude},${current.longitude}';
    }

    await _speak('Abriendo ruta peatonal en Maps hacia ${target.name}.');
    final uri = Uri.https('www.google.com', '/maps/dir/', queryParameters);
    final opened = await launchUrl(uri, mode: LaunchMode.externalApplication);
    if (!opened) {
      await _speak('No pude abrir Maps en este dispositivo.');
    }
  }

  Future<void> _triggerSos() async {
    final contact = _emergencyContact();
    if (contact == null) {
      await _speak(
        'No hay contacto SOS. Pulsa Contactos y registra una persona de emergencia.',
      );
      return;
    }
    await _speak(
      'Activando SOS para ${contact.name}. Abrire el mensaje con tu ubicacion y luego la llamada.',
    );
    final locationText = await _currentLocationText();
    final message = 'SOS. Necesito ayuda. Mi ubicacion: $locationText';
    await _openWhatsApp(contact, message: message);
    await Future<void>.delayed(const Duration(seconds: 1));
    await _openPhoneCall(contact);
  }

  Future<String> _currentLocationText() async {
    final position = await _currentPosition();
    if (position == null) return 'ubicacion no disponible';
    return 'https://maps.google.com/?q=${position.latitude},${position.longitude}';
  }

  void _showContactsDialog() {
    showDialog<void>(
      context: context,
      builder: (context) => _ContactsDialog(
        contacts: _contacts,
        onAdd: _addContact,
        onDelete: _deleteContact,
        onEmergency: _setEmergencyContact,
        onCall: _openPhoneCall,
        onWhatsApp: (contact) =>
            _openWhatsApp(contact, message: 'Hola ${contact.name}.'),
        onSos: () {
          Navigator.of(context).pop();
          _triggerSos();
        },
      ),
    );
  }

  void _showHistoryDialog() {
    showDialog<void>(
      context: context,
      builder: (context) => _HistoryDialog(
        entries: _commandHistory,
        onExport: _exportCommandHistory,
        onClear: _clearCommandHistory,
      ),
    );
  }

  Future<void> _enableDetectionMode(AppMode mode) async {
    await _tts.stop();
    if (!mounted) return;
    setState(() {
      _mode = mode;
      _detections = const [];
      _ocrText = '';
      _isSpeaking = false;
      _status = _cameraSource.status == CameraStatus.ready
          ? 'Camara lista. Usa Que ves para analizar entorno'
          : 'Camara desconectada';
    });
  }

  Future<void> _enableDangerRecognition() async {
    await _enableDetectionMode(AppMode.danger);
  }

  Future<void> _enterReadingMode() async {
    _walking.stop();
    if (_cameraSource.status != CameraStatus.ready) {
      await _speak('Camara desconectada');
      return;
    }
    await _enableDetectionMode(AppMode.reading);
    await _speak('Modo lectura. Enfoca el texto y di leer ahora.');
  }

  // ML Kit and Cactus require a path. Keep their temporary capture only until
  // the consumer finishes, including error and disconnect paths.
  Future<T> _withCameraFile<T>(Future<T> Function(String path) action) async {
    final bytes = await _cameraSource.captureFrame();
    return _withImageFile(bytes, action);
  }

  Future<T> _withImageFile<T>(
    Uint8List bytes,
    Future<T> Function(String path) action,
  ) async {
    final directory = await Directory(
      '${(await getTemporaryDirectory()).path}/magicvision-captures',
    ).create(recursive: true);
    final file = File(
      '${directory.path}/frame-${DateTime.now().microsecondsSinceEpoch}.jpg',
    );
    try {
      await file.writeAsBytes(bytes);
      return await action(file.path);
    } finally {
      if (await file.exists()) await file.delete();
    }
  }

  Future<void> _captureAndReadText() async {
    _walking.stop();
    if (_isReading || _isAnalyzingEnvironment || _isDescribingScene) return;
    _isReading = true;
    if (mounted) {
      setState(() {
        _mode = AppMode.reading;
        _status = 'Capturando texto...';
      });
    }
    try {
      final text = await _withCameraFile(
        (path) async => (await _textRecognizer.processImage(
          InputImage.fromFilePath(path),
        )).text.trim(),
      );
      if (!mounted) return;
      setState(() {
        _ocrText = text.isEmpty ? 'No encontre texto en la imagen.' : text;
        _status = 'Lectura lista';
      });
      await _speak(_ocrText);
    } catch (_) {
      if (mounted) {
        setState(() => _status = 'No se pudo procesar el texto');
        await _speak(
          _cameraSource.status == CameraStatus.ready
              ? 'No se pudo procesar el texto'
              : 'Camara desconectada',
        );
      }
    } finally {
      _isReading = false;
      if (mounted) setState(() {});
    }
  }

  Future<({String text, AnalysisResult analysis, List<Detection> detections})?>
  _describeSceneWithMobileVlm({
    required MobileVlmModelStatus modelStatus,
  }) async {
    if (_cameraSource.status != CameraStatus.ready ||
        _isDescribingScene ||
        _isReading ||
        !modelStatus.isReady) {
      return null;
    }
    var capture = _visionAnalysisService.lastCapture;
    if (capture == null ||
        !_visionAnalysisService.isResultCurrent(capture.analysis)) {
      await _analyzeEnvironmentFromCamera();
      capture = _visionAnalysisService.lastCapture;
    }
    if (capture == null ||
        !_visionAnalysisService.isResultCurrent(capture.analysis)) {
      return null;
    }
    final currentCapture = capture;
    String? sentPrompt;
    final size = Size(
      capture.response.imageWidth.toDouble(),
      capture.response.imageHeight.toDouble(),
    );
    final detectionHints = capture.response.detections
        .map(
          (d) => Detection(
            label: d.label,
            confidence: d.confidence,
            sourceSize: size,
            rect: Rect.fromLTRB(
              d.bbox.left * size.width,
              d.bbox.top * size.height,
              d.bbox.right * size.width,
              d.bbox.bottom * size.height,
            ),
          ),
        )
        .toList();
    _isDescribingScene = true;
    try {
      if (mounted) {
        setState(() => _status = 'Analizando descripcion completa...');
      }
      final description = await _withImageFile(
        currentCapture.image.bytes,
        (path) => _mobileVlmClient.describeImage(
          path,
          prompt: sentPrompt = mobileVlmPromptForDetections(detectionHints),
          modelStatus: modelStatus,
        ),
      );
      final clean = description.trim();
      if (clean.isEmpty ||
          !_visionAnalysisService.isResultCurrent(currentCapture.analysis)) {
        return null;
      }
      if (!await _visionAnalysisService.recordVisualResult(
        currentCapture,
        prompt: sentPrompt!,
        text: clean,
        model: modelStatus.displayName,
        speechText: groundedMobileVlmSpeech(
          vlmDescription: clean,
          analysis: currentCapture.analysis,
          detections: detectionHints,
        ),
      )) {
        return null;
      }
      if (mounted) {
        setState(() {
          _lastVisualDescription = clean;
          _lastVisualSource = modelStatus.displayName;
          _status = 'Descripcion completa lista';
        });
      }
      return (
        text: clean,
        analysis: currentCapture.analysis,
        detections: detectionHints,
      );
    } catch (_) {
      if (mounted) {
        setState(() => _status = 'Descripcion completa no disponible');
      }
      return null;
    } finally {
      _isDescribingScene = false;
      if (mounted) setState(() {});
    }
  }

  Future<void> _speak(String text, {bool Function()? isCurrent}) async {
    if (isCurrent != null && !isCurrent()) return;
    final clean = formatLongNumbersForSpeech(text.trim());
    if (clean.isEmpty) return;
    _lastSpokenText = clean;
    _ignoreSpeechUntil = DateTime.now().add(speechCommandCooldown);
    await _tts.stop();
    if (!mounted || (isCurrent != null && !isCurrent())) return;
    if (mounted) setState(() => _isSpeaking = true);
    await _tts.speak(
      clean.length > 900 ? clean.substring(0, 900) : clean,
      focus: false,
    );
  }

  Future<void> _stopSpeaking() async {
    await _tts.stop();
    _markSpeechIdle(cooldown: const Duration(milliseconds: 150));
  }

  Future<void> _speakHelp() async {
    await _speak(
      'Puedes decir: que hora es, que fecha es, leer, leer ahora, repetir, '
      'que ves para describir lo detectado con posicion, '
      'cancelar analisis para cancelar la espera de vision, '
      'ver peligro para revisar personas, vehiculos, escaleras, huecos, postes y obstaculos, temporizador en cinco minutos, '
      'recuerdame tomar medicina en diez minutos, mis recordatorios, '
      'analiza imagen o modelo visual para una descripcion completa con LFM dos punto cinco, '
      'estado modelo visual para saber si LFM dos punto cinco esta listo, '
      'descargar modelo visual para revisar la preparacion del motor visual, '
      'dime donde estoy, quiero ir a una farmacia, abrir Maps para ruta peatonal exacta, '
      'llamar a mama, '
      'quiero guardar contacto para que te pregunte nombre, numero y si es SOS, '
      'agrega contacto mama numero nueve ocho siete seis cinco cuatro tres dos uno, '
      'mis contactos, historial, que tengo guardado, cancelar ruta, '
      'llamar a mama por WhatsApp, SOS, comandos, silencio, alto o cancelar para cortar la voz.',
    );
  }

  void _onVisualModelPressed() {
    debugPrint('Boton Visual presionado');
    if (mounted) {
      setState(() => _status = 'Analisis visual solicitado...');
    }
    _runButtonAction(
      'descripcion visual',
      () => _describeSceneDetections(explicitMobileVlm: true),
    );
  }

  void _onMobileVlmStatusPressed() {
    debugPrint('Boton Estado visual presionado');
    if (mounted) {
      setState(() => _status = 'Revisando modelo visual local...');
    }
    _runButtonAction('estado visual', _speakMobileVlmStatus);
  }

  void _runButtonAction(String label, Future<void> Function() action) {
    unawaited(() async {
      try {
        await action();
      } catch (error, stackTrace) {
        final message = friendlyMobileVlmError(error);
        debugPrint('Accion de $label fallo: $error');
        debugPrintStack(stackTrace: stackTrace);
        if (!mounted) return;
        setState(() => _status = 'No pude completar $label');
        try {
          await _speak('No pude completar $label. $message');
        } catch (speechError, speechStackTrace) {
          debugPrint('No pude avisar el error de $label: $speechError');
          debugPrintStack(stackTrace: speechStackTrace);
        }
      }
    }());
  }

  Future<void> _speakMobileVlmStatus() async {
    debugPrint('Solicitud de estado MobileVLM');
    if (mounted) {
      setState(() => _status = 'Revisando LFM 2.5...');
    }
    await _speak('Revisando LFM dos punto cinco.');
    final status = await _mobileVlmClient.status();
    debugPrint(
      'Estado MobileVLM: listo=${status.isReady}, ruta=${status.modelPath ?? 'ninguna'}',
    );
    if (!status.isReady && await _mobileVlmClient.hasBundledModel()) {
      if (mounted) {
        setState(() => _status = 'Preparando descripcion visual local...');
      }
      await _prepareBundledMobileVlmModel();
      await _speakVerifiedMobileVlmStatus();
      return;
    }
    await _speakVerifiedMobileVlmStatus();
  }

  Future<void> _speakVerifiedMobileVlmStatus() async {
    final status = await _mobileVlmClient.status();
    if (status.isReady && mounted) {
      setState(
        () => _status = status.isRemote
            ? 'LFM 2.5 conectado'
            : 'Probando descripcion visual local...',
      );
    }
    final message = await _mobileVlmClient.runtimeStatusMessage().timeout(
      const Duration(seconds: 45),
      onTimeout: () =>
          'Encontre los archivos del modelo visual en el telefono, '
          'pero la prueba de carga esta tardando demasiado. '
          'Vuelve a tocar Estado en un momento.',
    );
    if (mounted && (message.contains('lista en memoria') || status.isRemote)) {
      setState(
        () => _status = status.isRemote
            ? 'LFM 2.5 listo'
            : 'Descripcion visual lista',
      );
    }
    await _speak(message);
  }

  Future<void> _migrateExternalMobileVlmModel() async {
    if (_isPreparingBundledMobileVlm) return;
    _isPreparingBundledMobileVlm = true;
    if (mounted) {
      setState(() => _status = 'Preparando descripcion visual local...');
    }
    try {
      await _mobileVlmClient.migrateExternalModel(
        onProgress: (progress, statusMessage) {
          if (!mounted) return;
          final percent = progress == null
              ? ''
              : ' ${(progress * 100).clamp(0, 100).toStringAsFixed(0)}%';
          setState(() => _status = '$statusMessage$percent');
        },
      );
      if (mounted) {
        setState(() => _status = 'Descripcion visual local lista');
      }
    } finally {
      _isPreparingBundledMobileVlm = false;
    }
  }

  Future<void> _prepareBundledMobileVlmModel() async {
    if (_isPreparingBundledMobileVlm || _isInstallingMobileVlm) return;
    final currentStatus = await _mobileVlmClient.status();
    if (currentStatus.isReady) return;
    if (!await _mobileVlmClient.hasBundledModel()) return;

    _isPreparingBundledMobileVlm = true;
    if (mounted) {
      setState(() => _status = 'Preparando descripcion visual local...');
    }
    try {
      await _mobileVlmClient.installBundledModel(
        onProgress: (progress, statusMessage) {
          if (!mounted) return;
          final percent = progress == null
              ? ''
              : ' ${(progress * 100).clamp(0, 100).toStringAsFixed(0)}%';
          setState(() => _status = '$statusMessage$percent');
        },
      );
      if (mounted) {
        setState(() => _status = 'Descripcion visual local lista');
      }
    } catch (_) {
      if (mounted) {
        setState(() => _status = 'No se pudo preparar descripcion visual');
      }
    } finally {
      _isPreparingBundledMobileVlm = false;
    }
  }

  Future<void> _prepareMobileVlmForInstantUse() async {
    final status = await _mobileVlmClient.status();
    if (!status.isRemote &&
        !status.isReady &&
        !await _mobileVlmClient.hasBundledModel()) {
      return;
    }
    await _prepareBundledMobileVlmModel();
    await _warmUpMobileVlm();
  }

  Future<bool> _warmUpMobileVlm() async {
    if (_mobileVlmClient.isRuntimeLoaded) return true;
    if (_isWarmingMobileVlm) return false;

    _isWarmingMobileVlm = true;
    if (mounted) {
      setState(() => _status = 'Cargando descripcion visual...');
    }
    try {
      final ready = await _mobileVlmClient.warmUpRuntime();
      if (mounted) {
        setState(
          () => _status = ready
              ? '${_mobileVlmClient.displayName} listo'
              : 'Descripcion visual aun no esta lista',
        );
      }
      return ready;
    } catch (error, stackTrace) {
      debugPrint('Visual description warm up failed: $error');
      debugPrintStack(stackTrace: stackTrace);
      if (mounted) {
        setState(() => _status = 'Descripcion visual no pudo quedar lista');
      }
      return false;
    } finally {
      _isWarmingMobileVlm = false;
    }
  }

  Future<void> _installMobileVlmModel() async {
    if (_isInstallingMobileVlm) {
      await _speak(
        'Ya estoy preparando la descripcion visual. Esto puede tardar un momento.',
      );
      return;
    }

    _isInstallingMobileVlm = true;
    if (mounted) {
      setState(() => _status = 'Preparando descripcion visual...');
    }
    final currentStatus = await _mobileVlmClient.status();
    if (currentStatus.isReady) {
      await _speakVerifiedMobileVlmStatus();
      _isInstallingMobileVlm = false;
      return;
    }
    if (_mobileVlmClient.requiresRemoteEndpoint &&
        !_mobileVlmClient.hasRemoteEndpoint) {
      await _speak(currentStatus.speechMessage);
      _isInstallingMobileVlm = false;
      return;
    }
    final externalModelPath = await _mobileVlmClient.externalModelPath();
    final hasBundledModel = externalModelPath == null
        ? await _mobileVlmClient.hasBundledModel()
        : false;
    await _speak(
      externalModelPath != null
          ? 'Ya encontre los archivos visuales. Los preparare dentro de la app.'
          : hasBundledModel
          ? 'Voy a preparar la descripcion visual local que ya viene dentro de la app. No necesito internet.'
          : 'Voy a descargar la descripcion visual local. Necesito internet solo para esta descarga; despues funcionara localmente.',
    );

    try {
      if (externalModelPath != null) {
        await _migrateExternalMobileVlmModel();
      } else if (hasBundledModel) {
        await _mobileVlmClient.installBundledModel(
          onProgress: (progress, statusMessage) {
            if (!mounted) return;
            final progressText = progress == null
                ? ''
                : ' ${(progress * 100).clamp(0, 100).toStringAsFixed(0)}%';
            setState(() => _status = '$statusMessage$progressText');
          },
        );
      } else {
        await _mobileVlmClient.installModel(
          onProgress: (progress, statusMessage, isError) {
            if (!mounted) return;
            final progressText = progress == null
                ? ''
                : ' ${(progress * 100).clamp(0, 100).toStringAsFixed(0)}%';
            setState(() {
              _status = isError
                  ? 'Error preparando descripcion visual'
                  : '$statusMessage$progressText';
            });
          },
        );
      }
      await _speakVerifiedMobileVlmStatus();
    } catch (_) {
      final status = await _mobileVlmClient.status();
      await _speak(
        hasBundledModel
            ? 'No pude preparar la descripcion visual automaticamente. ${status.speechMessage}'
            : 'No pude descargar la descripcion visual automaticamente. ${status.speechMessage}',
      );
    } finally {
      _isInstallingMobileVlm = false;
    }
  }

  Future<void> _speakTime() async {
    await _speak('Son las ${formatTimeForSpeech(DateTime.now())}.');
  }

  Future<void> _speakDate() async {
    await _speak('Hoy es ${formatDateForSpeech(DateTime.now())}.');
  }

  Future<void> _repeatLastSpeech() async {
    if (_ocrText.trim().isNotEmpty) {
      await _speak(_ocrText);
      return;
    }
    if (_lastSpokenText.trim().isNotEmpty) {
      await _speak(_lastSpokenText);
      return;
    }
    await _speak('Todavia no hay una lectura para repetir.');
  }

  Future<void> _describeDangerDetections() async {
    if (_mode != AppMode.danger) await _enableDangerRecognition();
    final result = await _analyzeEnvironmentFromCamera();
    if (result != null) {
      await _speak(
        result.speechText,
        isCurrent: () => _visionAnalysisService.isResultCurrent(result),
      );
    }
  }

  Future<AnalysisResult?> _analyzeEnvironmentFromCamera() async {
    if (_isAnalyzingEnvironment || _isReading || _isDescribingScene) {
      return null;
    }
    _isAnalyzingEnvironment = true;
    _captureDisplayTimer?.cancel();
    if (mounted) setState(() => _status = 'Analizando entorno');
    try {
      if (!mounted) return null;
      final result = await _visionAnalysisService.analyzeCurrentFrame();
      if (!mounted) return null;
      if (result == null) {
        if (_cameraSource.status != CameraStatus.disconnected) {
          await _speak(
            _visionAnalysisService.errorMessage ??
                'No se pudo analizar la imagen',
          );
        }
        return null;
      }
      final response = _visionAnalysisService.lastResponse;
      if (response == null || _cameraSource.status != CameraStatus.ready) {
        return null;
      }
      final size = Size(
        response.imageWidth.toDouble(),
        response.imageHeight.toDouble(),
      );
      setState(() {
        _lastEnvironmentAnalysis = result;
        _status = _statusForAnalysisResult(result);
        _detections = response.detections
            .map(
              (d) => Detection(
                label: d.label,
                confidence: d.confidence,
                sourceSize: size,
                rect: Rect.fromLTRB(
                  d.bbox.left * size.width,
                  d.bbox.top * size.height,
                  d.bbox.right * size.width,
                  d.bbox.bottom * size.height,
                ),
              ),
            )
            .toList();
      });
      _captureDisplayTimer = Timer(detectionFreshnessDuration, () {
        if (mounted) setState(() => _detections = const []);
      });
      _recordEnvironmentAnalysis(result);
      return result;
    } finally {
      _isAnalyzingEnvironment = false;
      if (mounted) setState(() {});
    }
  }

  String _statusForAnalysisResult(AnalysisResult result) {
    return switch (result.riskLevel) {
      'alto' => 'Riesgo alto: ${result.suggestedAction}',
      'medio' => 'Riesgo medio: ${result.suggestedAction}',
      _ => 'Riesgo bajo: ${result.suggestedAction}',
    };
  }

  Future<void> _describeSceneDetections({
    bool explicitMobileVlm = false,
  }) async {
    if (explicitMobileVlm) {
      debugPrint('Solicitud de descripcion Visual');
      if (mounted) {
        setState(() => _status = 'Preparando analisis visual...');
      }
    }
    if (_mode != AppMode.danger) {
      await _enableDangerRecognition();
      await Future<void>.delayed(const Duration(milliseconds: 900));
    }

    final analysisResult = await _analyzeEnvironmentFromCamera();
    if (analysisResult == null) return;
    final quickSummary = analysisResult.speechText;
    bool isCurrent() => _visionAnalysisService.isResultCurrent(analysisResult);
    if (!explicitMobileVlm) {
      await _speak(quickSummary, isCurrent: isCurrent);
      return;
    }

    final requestId = ++_visualAnalysisRequestId;
    var mobileVlmStatus = await _mobileVlmClient.status();
    if (!mobileVlmStatus.isReady && await _mobileVlmClient.hasBundledModel()) {
      unawaited(_prepareMobileVlmForInstantUse());
      if (mounted) {
        setState(() => _status = 'Descripcion rapida lista');
      }
      await _speak(quickSummary, isCurrent: isCurrent);
      return;
    }

    if (!mobileVlmStatus.isReady) {
      if (mounted) {
        setState(() => _status = 'Preparando modelo visual...');
      }
      await _speak(
        '$quickSummary Para la descripcion completa voy a preparar ${_mobileVlmClient.displayName}. '
        'Esto se hace una vez y despues funcionara sin internet.',
        isCurrent: isCurrent,
      );
      unawaited(_installMobileVlmModel());
      return;
    }

    if (mobileVlmStatus.isReady) {
      if (_mobileVlmClient.isRuntimeLoaded && !_isDescribingScene) {
        unawaited(
          _speakExplicitMobileVlmResult(mobileVlmStatus, requestId: requestId),
        );
      }
      if (!_mobileVlmClient.isRuntimeLoaded) {
        unawaited(() async {
          final ready = await _warmUpMobileVlm();
          if (!ready || requestId != _visualAnalysisRequestId) return;
          final refreshedStatus = await _mobileVlmClient.status();
          if (!refreshedStatus.isReady) return;
          await _speakExplicitMobileVlmResult(
            refreshedStatus,
            requestId: requestId,
          );
        }());
      }
      if (mounted) setState(() => _status = 'Descripcion rapida lista');
      await _speak(quickSummary, isCurrent: isCurrent);
      return;
    }

    if (mounted) setState(() => _status = 'Descripcion rapida lista');
    await _speak(quickSummary, isCurrent: isCurrent);
  }

  Future<void> _speakExplicitMobileVlmResult(
    MobileVlmModelStatus mobileVlmStatus, {
    required int requestId,
  }) async {
    debugPrint('Iniciando descripcion visual completa');
    final description = await _describeSceneWithMobileVlm(
      modelStatus: mobileVlmStatus,
    );
    if (requestId != _visualAnalysisRequestId) return;
    if (!mounted || description == null || description.text.isEmpty) return;
    final grounded = groundedMobileVlmSpeech(
      vlmDescription: description.text,
      analysis: description.analysis,
      detections: description.detections,
    );
    await _speak(
      grounded,
      isCurrent: () =>
          _visionAnalysisService.isResultCurrent(description.analysis),
    );
  }

  Future<void> _createTimerFromCommand(String command) async {
    final duration = parseSpanishDuration(command);
    if (duration == null) {
      await _speak(
        'Dime el tiempo. Por ejemplo: temporizador en cinco minutos.',
      );
      return;
    }
    _addReminder(
      message: 'El temporizador termino.',
      duration: duration,
      isTimer: true,
    );
    await _speak(
      'Temporizador creado por ${formatDurationForSpeech(duration)}.',
    );
  }

  Future<void> _createReminderFromCommand(String command) async {
    final duration = parseSpanishDuration(command);
    if (duration == null) {
      await _speak(
        'Dime cuando quieres el recordatorio. Por ejemplo: recuerdame tomar agua en diez minutos.',
      );
      return;
    }
    final message = extractReminderMessage(command);
    _addReminder(message: message, duration: duration, isTimer: false);
    await _speak(
      'Recordatorio creado: $message, en ${formatDurationForSpeech(duration)}.',
    );
  }

  void _addReminder({
    required String message,
    required Duration duration,
    required bool isTimer,
  }) {
    late final VoiceReminder reminder;
    final dueAt = DateTime.now().add(duration);
    reminder = VoiceReminder(
      message: message,
      dueAt: dueAt,
      isTimer: isTimer,
      timer: Timer(duration, () {
        _reminders.remove(reminder);
        _speak(isTimer ? message : 'Recordatorio: $message');
      }),
    );
    _reminders.add(reminder);
  }

  Future<void> _listReminders() async {
    if (_reminders.isEmpty) {
      await _speak('No tienes recordatorios pendientes.');
      return;
    }
    final now = DateTime.now();
    final pending = _reminders
        .map((reminder) {
          final remaining = reminder.dueAt.difference(now);
          final type = reminder.isTimer ? 'temporizador' : 'recordatorio';
          return '$type: ${reminder.message}, falta ${formatDurationForSpeech(remaining)}';
        })
        .join('. ');
    await _speak(pending);
  }

  Future<void> _clearReminders() async {
    for (final reminder in _reminders) {
      reminder.timer.cancel();
    }
    final count = _reminders.length;
    _reminders.clear();
    await _speak(
      count == 0
          ? 'No habia recordatorios pendientes.'
          : 'Listo. Cancele $count recordatorios.',
    );
  }

  @override
  void dispose() {
    _closing = true;
    WidgetsBinding.instance.removeObserver(this);
    _walking.stop();
    _captureDisplayTimer?.cancel();
    _visionAnalysisService.removeListener(_inferenceChanged);
    _cameraSource.dispose();
    _visionAnalysisService.dispose();
    _mobileVlmClient.dispose();
    _textRecognizer.close();
    _speech.stop();
    _tts.stop();
    _navigationSubscription?.cancel();
    for (final reminder in _reminders) {
      reminder.timer.cancel();
    }
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final ready = _cameraSource.status == CameraStatus.ready;

    return Scaffold(
      backgroundColor: const Color(0xff05070b),
      body: SafeArea(
        child: Stack(
          fit: StackFit.expand,
          children: [
            if (ready)
              _detections.isNotEmpty && _visionAnalysisService.lastImage != null
                  ? Semantics(
                      label: 'Ultima captura analizada',
                      child: Image.memory(
                        _visionAnalysisService.lastImage!.bytes,
                        fit: BoxFit.cover,
                        gaplessPlayback: true,
                      ),
                    )
                  : _cameraSource.buildPreview()
            else
              Center(
                child: Padding(
                  padding: const EdgeInsets.all(32),
                  child: Column(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      const SizedBox(
                        width: 44,
                        height: 44,
                        child: CircularProgressIndicator(strokeWidth: 3),
                      ),
                      TextButton(
                        onPressed: () async {
                          if (await Permission.camera.request().isGranted) {
                            await _visionAnalysisService.initialize();
                          }
                        },
                        child: const Text('Reconectar camara'),
                      ),
                      const SizedBox(height: 20),
                      Text(
                        _status,
                        textAlign: TextAlign.center,
                        style: Theme.of(context).textTheme.titleMedium,
                      ),
                    ],
                  ),
                ),
              ),
            const DecoratedBox(
              decoration: BoxDecoration(
                gradient: LinearGradient(
                  begin: Alignment.topCenter,
                  end: Alignment.bottomCenter,
                  colors: [
                    Color(0xcc05070b),
                    Color(0x2205070b),
                    Color(0xdd05070b),
                  ],
                ),
              ),
            ),
            if (ready && _mode != AppMode.reading)
              CustomPaint(
                painter: DetectionPainter(_detections),
                child: const SizedBox.expand(),
              ),
            Positioned(
              left: 16,
              right: 16,
              top: 16,
              child: _TopStatusBar(
                mode: _mode,
                status: _status,
                voiceStatus: _voiceStatus,
                listening: _speech.isListening,
                detections: _detections,
              ),
            ),
            if (ready)
              Positioned(
                top: 78,
                right: 16,
                child: FilledButton.icon(
                  onPressed: _toggleWalking,
                  icon: Icon(
                    _walking.running ? Icons.pause : Icons.directions_walk,
                  ),
                  label: Text(
                    _walking.running
                        ? 'Detener asistencia'
                        : 'Asistencia al caminar',
                  ),
                ),
              ),
            if (ready && _mode == AppMode.danger)
              Positioned(
                left: 16,
                right: 16,
                top: 118,
                child: IgnorePointer(
                  child: _navigationSnapshot == null
                      ? _SafetyHudOverlay(detections: _detections)
                      : _NavigationHud(snapshot: _navigationSnapshot!),
                ),
              ),
            Positioned(
              left: 16,
              right: 16,
              bottom: 16,
              child: _CommandPanel(
                mode: _mode,
                lastWords: _lastWords,
                ocrText: _ocrText,
                lastAnalysis: _lastEnvironmentAnalysis,
                lastVisualDescription: _lastVisualDescription,
                lastVisualSource: _lastVisualSource,
                isAnalyzingEnvironment: _isAnalyzingEnvironment,
                isDescribingScene: _isDescribingScene,
                isReading: _isReading,
                isSpeaking: _isSpeaking,
                onHelp: _speakHelp,
                onTime: _speakTime,
                onDate: _speakDate,
                onDescribe: () {
                  if (_visionAnalysisService.isInferenceRunning) {
                    _visionAnalysisService.cancelAnalysis();
                  } else {
                    unawaited(_describeSceneDetections());
                  }
                },
                onVisualModel: _onVisualModelPressed,
                onMobileVlmStatus: _onMobileVlmStatusPressed,
                onLocation: _speakCurrentLocation,
                onDanger: _enableDangerRecognition,
                onPrepareRead: _enterReadingMode,
                onReadNow: _captureAndReadText,
                onStopSpeaking: _stopSpeaking,
                onContacts: _showContactsDialog,
                onHistory: _showHistoryDialog,
                onSos: _triggerSos,
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class _TopStatusBar extends StatelessWidget {
  const _TopStatusBar({
    required this.mode,
    required this.status,
    required this.voiceStatus,
    required this.listening,
    required this.detections,
  });

  final AppMode mode;
  final String status;
  final String voiceStatus;
  final bool listening;
  final List<Detection> detections;

  @override
  Widget build(BuildContext context) {
    final accent = mode == AppMode.danger
        ? const Color(0xfff87171)
        : const Color(0xff60a5fa);
    final dangerCount = detections.where(isDangerousDetection).length;
    final totalCount = detections.length;

    return Container(
      decoration: BoxDecoration(
        color: const Color(0xff101722).withValues(alpha: 0.82),
        borderRadius: BorderRadius.circular(8),
        border: Border.all(color: Colors.white.withValues(alpha: 0.12)),
      ),
      child: Padding(
        padding: const EdgeInsets.all(12),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Container(
                  width: 34,
                  height: 34,
                  decoration: BoxDecoration(
                    color: accent.withValues(alpha: 0.18),
                    borderRadius: BorderRadius.circular(8),
                  ),
                  child: Icon(
                    mode == AppMode.danger
                        ? Icons.warning_amber_outlined
                        : Icons.document_scanner_outlined,
                    color: accent,
                    size: 20,
                  ),
                ),
                const SizedBox(width: 10),
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        mode == AppMode.danger ? 'Guardian Pro' : 'Lectura OCR',
                        style: const TextStyle(
                          fontSize: 16,
                          fontWeight: FontWeight.w800,
                        ),
                      ),
                      Text(
                        status,
                        maxLines: 1,
                        overflow: TextOverflow.ellipsis,
                        style: TextStyle(
                          color: Colors.white.withValues(alpha: 0.70),
                          fontSize: 12,
                        ),
                      ),
                    ],
                  ),
                ),
                _MetricPill(
                  icon: dangerCount > 0
                      ? Icons.report_problem_outlined
                      : Icons.radar_outlined,
                  label: mode == AppMode.danger
                      ? '$dangerCount/$totalCount'
                      : '$totalCount',
                ),
              ],
            ),
            const SizedBox(height: 10),
            Row(
              children: [
                Icon(
                  listening ? Icons.mic : Icons.mic_off_outlined,
                  size: 16,
                  color: listening ? const Color(0xfff8d66d) : Colors.white54,
                ),
                const SizedBox(width: 8),
                Expanded(
                  child: Text(
                    voiceStatus,
                    maxLines: 1,
                    overflow: TextOverflow.ellipsis,
                    style: TextStyle(
                      color: Colors.white.withValues(alpha: 0.72),
                      fontSize: 12,
                    ),
                  ),
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }
}

class _NavigationHud extends StatelessWidget {
  const _NavigationHud({required this.snapshot});

  final NavigationSnapshot snapshot;

  @override
  Widget build(BuildContext context) {
    final accent = snapshot.movingAway
        ? const Color(0xfffbbf24)
        : const Color(0xff5eead4);
    final accuracyGood =
        snapshot.accuracyMeters <= navigationGoodAccuracyMeters;
    final address = snapshot.targetAddress?.trim() ?? '';

    return AnimatedContainer(
      duration: const Duration(milliseconds: 180),
      decoration: BoxDecoration(
        color: const Color(0xff08111d).withValues(alpha: 0.86),
        borderRadius: BorderRadius.circular(8),
        border: Border.all(color: accent.withValues(alpha: 0.48)),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withValues(alpha: 0.26),
            blurRadius: 18,
            offset: const Offset(0, 10),
          ),
        ],
      ),
      padding: const EdgeInsets.all(12),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Container(
                width: 34,
                height: 34,
                decoration: BoxDecoration(
                  color: accent.withValues(alpha: 0.18),
                  borderRadius: BorderRadius.circular(8),
                ),
                child: Icon(
                  snapshot.movingAway
                      ? Icons.explore_off_outlined
                      : Icons.navigation_outlined,
                  color: accent,
                  size: 20,
                ),
              ),
              const SizedBox(width: 10),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      snapshot.movingAway ? 'Reorientando ruta' : 'En ruta',
                      maxLines: 1,
                      overflow: TextOverflow.ellipsis,
                      style: const TextStyle(
                        fontSize: 12,
                        fontWeight: FontWeight.w800,
                        color: Colors.white70,
                      ),
                    ),
                    Text(
                      snapshot.targetName,
                      maxLines: 1,
                      overflow: TextOverflow.ellipsis,
                      style: const TextStyle(
                        fontSize: 16,
                        fontWeight: FontWeight.w900,
                      ),
                    ),
                  ],
                ),
              ),
            ],
          ),
          if (address.isNotEmpty) ...[
            const SizedBox(height: 8),
            Text(
              address,
              maxLines: 1,
              overflow: TextOverflow.ellipsis,
              style: TextStyle(
                color: Colors.white.withValues(alpha: 0.70),
                fontSize: 12,
                fontWeight: FontWeight.w600,
              ),
            ),
          ],
          const SizedBox(height: 10),
          Wrap(
            spacing: 8,
            runSpacing: 8,
            children: [
              _NavigationPill(
                icon: Icons.straighten_outlined,
                label: navigationDistanceCompact(snapshot.distanceMeters),
              ),
              _NavigationPill(
                icon: Icons.explore_outlined,
                label: snapshot.direction,
              ),
              _NavigationPill(
                icon: accuracyGood
                    ? Icons.gps_fixed_outlined
                    : Icons.gps_not_fixed_outlined,
                label: accuracyGood
                    ? 'GPS ${snapshot.accuracyMeters.round()} m'
                    : 'GPS aprox.',
                accent: accuracyGood ? const Color(0xff5eead4) : accent,
              ),
            ],
          ),
        ],
      ),
    );
  }
}

class _NavigationPill extends StatelessWidget {
  const _NavigationPill({
    required this.icon,
    required this.label,
    this.accent = Colors.white,
  });

  final IconData icon;
  final String label;
  final Color accent;

  @override
  Widget build(BuildContext context) {
    return Container(
      height: 30,
      padding: const EdgeInsets.symmetric(horizontal: 9),
      decoration: BoxDecoration(
        color: Colors.white.withValues(alpha: 0.08),
        borderRadius: BorderRadius.circular(8),
        border: Border.all(color: Colors.white.withValues(alpha: 0.10)),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(icon, size: 15, color: accent),
          const SizedBox(width: 6),
          Text(
            label,
            maxLines: 1,
            overflow: TextOverflow.ellipsis,
            style: const TextStyle(
              color: Colors.white,
              fontSize: 12,
              fontWeight: FontWeight.w900,
            ),
          ),
        ],
      ),
    );
  }
}

class _SafetyHudOverlay extends StatelessWidget {
  const _SafetyHudOverlay({required this.detections});

  final List<Detection> detections;

  @override
  Widget build(BuildContext context) {
    final dangers = detections.where(isDangerousDetection).toList()
      ..sort(
        (a, b) => _dangerClosenessScore(b).compareTo(_dangerClosenessScore(a)),
      );
    final nearest = dangers.firstOrNull;
    final cautionCount = detections
        .where(
          (detection) =>
              riskLevelForLabel(detection.label) == RiskLevel.caution,
        )
        .length;
    final isNear = nearest != null && isNearbyDanger(nearest);
    final accent = nearest == null
        ? const Color(0xff34d399)
        : isNear
        ? const Color(0xfffb7185)
        : const Color(0xfffbbf24);
    final title = nearest == null
        ? 'Ruta despejada'
        : isNear
        ? 'Riesgo cerca'
        : 'Riesgo detectado';
    final subtitle = nearest == null
        ? (cautionCount > 0
              ? '$cautionCount elemento${cautionCount == 1 ? '' : 's'} de precaucion'
              : 'Sin peligros activos')
        : compactDangerSummary(nearest);

    return AnimatedContainer(
      duration: const Duration(milliseconds: 180),
      decoration: BoxDecoration(
        color: const Color(0xff0b111b).withValues(alpha: 0.78),
        borderRadius: BorderRadius.circular(8),
        border: Border.all(color: accent.withValues(alpha: 0.45)),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withValues(alpha: 0.24),
            blurRadius: 18,
            offset: const Offset(0, 10),
          ),
        ],
      ),
      padding: const EdgeInsets.all(10),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          Row(
            children: [
              Container(
                width: 34,
                height: 34,
                decoration: BoxDecoration(
                  color: accent.withValues(alpha: 0.18),
                  borderRadius: BorderRadius.circular(8),
                ),
                child: Icon(
                  nearest == null
                      ? Icons.verified_outlined
                      : Icons.warning_amber_rounded,
                  color: accent,
                  size: 20,
                ),
              ),
              const SizedBox(width: 10),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      title,
                      maxLines: 1,
                      overflow: TextOverflow.ellipsis,
                      style: const TextStyle(
                        fontSize: 14,
                        fontWeight: FontWeight.w900,
                      ),
                    ),
                    Text(
                      subtitle,
                      maxLines: 1,
                      overflow: TextOverflow.ellipsis,
                      style: TextStyle(
                        color: Colors.white.withValues(alpha: 0.72),
                        fontSize: 12,
                        fontWeight: FontWeight.w600,
                      ),
                    ),
                  ],
                ),
              ),
            ],
          ),
          const SizedBox(height: 8),
          Row(
            children: [
              Expanded(
                child: _DirectionRiskChip(
                  label: 'Izquierda',
                  count: dangers
                      .where((d) => dangerZoneForDetection(d) == 'left')
                      .length,
                ),
              ),
              const SizedBox(width: 8),
              Expanded(
                child: _DirectionRiskChip(
                  label: 'Frente',
                  count: dangers
                      .where((d) => dangerZoneForDetection(d) == 'front')
                      .length,
                ),
              ),
              const SizedBox(width: 8),
              Expanded(
                child: _DirectionRiskChip(
                  label: 'Derecha',
                  count: dangers
                      .where((d) => dangerZoneForDetection(d) == 'right')
                      .length,
                ),
              ),
            ],
          ),
        ],
      ),
    );
  }
}

class _DirectionRiskChip extends StatelessWidget {
  const _DirectionRiskChip({required this.label, required this.count});

  final String label;
  final int count;

  @override
  Widget build(BuildContext context) {
    final active = count > 0;
    final color = active ? const Color(0xfffb7185) : const Color(0xff34d399);
    return AnimatedContainer(
      duration: const Duration(milliseconds: 180),
      height: 30,
      padding: const EdgeInsets.symmetric(horizontal: 8),
      decoration: BoxDecoration(
        color: color.withValues(alpha: active ? 0.18 : 0.10),
        borderRadius: BorderRadius.circular(8),
        border: Border.all(
          color: color.withValues(alpha: active ? 0.52 : 0.28),
        ),
      ),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Icon(
            active ? Icons.priority_high_rounded : Icons.check_rounded,
            size: 14,
            color: color,
          ),
          const SizedBox(width: 5),
          Flexible(
            child: Text(
              active ? '$label $count' : label,
              maxLines: 1,
              overflow: TextOverflow.ellipsis,
              style: TextStyle(
                color: active ? Colors.white : Colors.white70,
                fontSize: 11,
                fontWeight: FontWeight.w900,
              ),
            ),
          ),
        ],
      ),
    );
  }
}

class MobileVlmClient {
  MobileVlmClient({
    required this.modelName,
    required this.modelId,
    required this.localModelName,
    required this.localModelId,
    this.remoteEndpoint = '',
    this.remoteApiKey = '',
    http.Client? httpClient,
    this.contextSize = mobileVlmContextSize,
    this.maxTokens = mobileVlmMaxTokens,
    this.quantizationBytes = mobileVlmQuantizationBytes,
  }) : _httpClient = httpClient ?? http.Client(),
       _ownsHttpClient = httpClient == null;

  final String modelName;
  final String modelId;
  final String localModelName;
  final String localModelId;
  final String remoteEndpoint;
  final String remoteApiKey;
  final int contextSize;
  final int maxTokens;
  final int quantizationBytes;
  final http.Client _httpClient;
  final bool _ownsHttpClient;
  int? _handle;
  String? _loadedModelPath;
  String? _lastInitializationError;
  Future<void>? _initializing;

  bool get isRuntimeLoaded => _handle != null;
  bool get hasRemoteEndpoint => remoteEndpoint.trim().isNotEmpty;
  bool get requiresRemoteEndpoint => false;
  String get activeModelName => hasRemoteEndpoint ? modelName : localModelName;
  String get activeModelId => hasRemoteEndpoint ? modelId : localModelId;
  String get displayName => mobileVlmReadableModelName(activeModelName);
  String get backendLabel =>
      hasRemoteEndpoint ? '$displayName endpoint' : '$displayName local';

  Future<String> describeImage(
    String imagePath, {
    required String prompt,
    MobileVlmModelStatus? modelStatus,
  }) async {
    final currentStatus = modelStatus ?? await status();
    if (currentStatus.isRemote || hasRemoteEndpoint) {
      return _describeImageWithRemoteEndpoint(imagePath, prompt: prompt);
    }
    if (!currentStatus.isReady || currentStatus.modelPath == null) {
      throw StateError(currentStatus.speechMessage);
    }
    await _ensureInitializedWithBundledRepair(currentStatus.modelPath!);
    final handle = _handle;
    if (handle == null) {
      throw StateError('La descripcion visual local no pudo inicializarse.');
    }

    final result = await CactusContext.completion(
      handle,
      [
        ChatMessage(
          role: 'system',
          content:
              'Eres una asistente visual para una persona con baja vision. '
              'Describe solo lo que puedas inferir de la imagen. '
              'Responde en espanol latino con una frase corta y util.',
        ),
        ChatMessage(role: 'user', content: prompt, images: [imagePath]),
      ],
      CactusCompletionParams(
        maxTokens: maxTokens,
        temperature: 0.2,
        topP: 0.9,
        completionMode: CompletionMode.local,
      ),
      quantizationBytes,
    );
    final description = normalizeMobileVlmDescription(_resultText(result));
    if (description.isEmpty) {
      throw const FormatException(
        'La descripcion visual local no devolvio texto.',
      );
    }
    if (!result.success) {
      throw StateError(description);
    }
    return description;
  }

  Future<String> _describeImageWithRemoteEndpoint(
    String imagePath, {
    required String prompt,
  }) async {
    final endpoint = _chatCompletionsUri();
    final image = File(imagePath);
    if (!await image.exists()) {
      throw StateError('No encontre la imagen capturada para LFM 2.5.');
    }

    final bytes = await image.readAsBytes();
    final mimeType = _mimeTypeForImagePath(imagePath);
    final imageDataUrl = 'data:$mimeType;base64,${base64Encode(bytes)}';
    final requestBody = jsonEncode({
      'model': modelId,
      'messages': [
        {
          'role': 'system',
          'content':
              'Eres una asistente visual para una persona con baja vision. '
              'Prioriza seguridad peatonal, obstaculos, personas, vehiculos, '
              'escaleras, huecos, postes y senales. Responde en espanol latino, '
              'natural, concreto y sin inventar.',
        },
        {
          'role': 'user',
          'content': [
            {'type': 'text', 'text': prompt},
            {
              'type': 'image_url',
              'image_url': {'url': imageDataUrl},
            },
          ],
        },
      ],
      'temperature': 0.1,
      'top_p': 0.9,
      'max_tokens': maxTokens,
    });

    final headers = <String, String>{
      HttpHeaders.contentTypeHeader: 'application/json',
    };
    if (remoteApiKey.trim().isNotEmpty) {
      headers[HttpHeaders.authorizationHeader] =
          'Bearer ${remoteApiKey.trim()}';
    }

    final response = await _httpClient
        .post(endpoint, headers: headers, body: requestBody)
        .timeout(mobileVlmHttpTimeout);
    if (response.statusCode < 200 || response.statusCode >= 300) {
      final body = response.body.trim();
      throw StateError(
        'LFM 2.5 respondio ${response.statusCode}'
        '${body.isEmpty ? '' : ': ${body.length > 220 ? body.substring(0, 220) : body}'}',
      );
    }

    final decoded = jsonDecode(response.body);
    final description = normalizeMobileVlmDescription(
      extractMobileVlmText(decoded),
    );
    if (description.isEmpty) {
      throw const FormatException(
        'LFM 2.5 no devolvio texto para esta imagen.',
      );
    }
    return description;
  }

  Uri _chatCompletionsUri() {
    final raw = remoteEndpoint.trim();
    if (raw.isEmpty) {
      throw StateError(
        'Configura MAGICVISION_VLM_ENDPOINT para usar LFM 2.5 VL 450M.',
      );
    }
    final normalized = raw.replaceAll(RegExp(r'/+$'), '');
    if (normalized.endsWith('/v1/chat/completions')) {
      return Uri.parse(normalized);
    }
    if (normalized.endsWith('/v1')) {
      return Uri.parse('$normalized/chat/completions');
    }
    return Uri.parse('$normalized/v1/chat/completions');
  }

  Future<MobileVlmModelStatus> status() async {
    if (hasRemoteEndpoint) {
      return MobileVlmModelStatus(
        isPlatformSupported: true,
        modelName: modelName,
        modelId: modelId,
        modelPath: null,
        searchedPaths: const [],
        backend: MobileVlmBackend.remote,
        endpoint: _chatCompletionsUri().toString(),
      );
    }

    final localName = activeModelName;
    final localId = activeModelId;
    if (!_isCactusPlatformSupported) {
      return MobileVlmModelStatus(
        isPlatformSupported: false,
        modelName: localName,
        modelId: localId,
        modelPath: null,
        searchedPaths: const [],
      );
    }

    final searchedPaths = await _candidateModelPaths(localName);
    await _ensureModelFolders(searchedPaths);
    final hasBundledCopy = await hasBundledModel();
    for (final path in searchedPaths) {
      final isUsableModel = hasBundledCopy
          ? await _directoryMatchesBundledMobileVlmSpec(path, localName)
          : await _directoryLooksLikeMobileVlmModel(path);
      if (isUsableModel) {
        return MobileVlmModelStatus(
          isPlatformSupported: true,
          modelName: localName,
          modelId: localId,
          modelPath: path,
          searchedPaths: searchedPaths,
        );
      }
    }

    return MobileVlmModelStatus(
      isPlatformSupported: true,
      modelName: localName,
      modelId: localId,
      modelPath: null,
      searchedPaths: searchedPaths,
    );
  }

  Future<bool> hasBundledModel() async {
    final assets = await _bundledMobileVlmAssetPaths(activeModelName);
    return assets.isNotEmpty;
  }

  Future<bool> warmUpRuntime() async {
    if (hasRemoteEndpoint) return true;
    var currentStatus = await status();
    if (!currentStatus.isReady && await hasBundledModel()) {
      await installBundledModel();
      currentStatus = await status();
    }
    final modelPath = currentStatus.modelPath;
    if (!currentStatus.isReady || modelPath == null) return false;
    await _ensureInitializedWithBundledRepair(modelPath);
    return _handle != null;
  }

  Future<String> runtimeStatusMessage() async {
    final currentStatus = await status();
    if (currentStatus.isRemote) {
      return '${currentStatus.displayName} esta configurado por endpoint. '
          'Usare ese motor para la descripcion completa y mantendre YOLO como respuesta rapida.';
    }
    if (!currentStatus.isReady || currentStatus.modelPath == null) {
      return currentStatus.speechMessage;
    }

    try {
      await _ensureInitializedWithBundledRepair(currentStatus.modelPath!);
      return 'La descripcion visual local esta lista en memoria. '
          'Ya puedo analizar imagenes sin internet.';
    } catch (error, stackTrace) {
      final message = friendlyMobileVlmError(error);
      debugPrint('Visual description runtime check failed: $error');
      debugPrintStack(stackTrace: stackTrace);
      return 'Encontre los archivos de la descripcion visual en el telefono, '
          'pero todavia no pude cargar el motor local. $message';
    }
  }

  Future<void> installBundledModel({
    void Function(double? progress, String statusMessage)? onProgress,
    bool force = false,
  }) async {
    if (!_isCactusPlatformSupported) {
      throw StateError(
        'La descripcion visual local no esta soportada en esta plataforma.',
      );
    }

    final localName = activeModelName;
    final searchedPaths = await _candidateModelPaths(localName);
    await _ensureModelFolders(searchedPaths);
    final targetPath = searchedPaths.first;
    if (!force &&
        await _directoryMatchesBundledMobileVlmSpec(targetPath, localName)) {
      onProgress?.call(1, 'Descripcion visual local ya estaba lista');
      return;
    }

    final assets = await _bundledMobileVlmAssetPaths(localName);
    if (assets.isEmpty) {
      throw StateError(
        'La descripcion visual local no esta incluida en assets.',
      );
    }

    final prefix = '$mobileVlmBundledAssetRoot/$localName/';
    final directAssets = <String>[];
    final chunkedAssets = <String, List<String>>{};
    for (final assetPath in assets) {
      final relativePath = assetPath.substring(prefix.length);
      final basePath = mobileVlmChunkBasePath(relativePath);
      if (basePath == null) {
        directAssets.add(assetPath);
      } else {
        (chunkedAssets[basePath] ??= []).add(assetPath);
      }
    }

    var copiedAssets = 0;
    Future<void> updateProgress() async {
      copiedAssets++;
      onProgress?.call(
        copiedAssets / assets.length,
        'Preparando descripcion visual local',
      );
    }

    final targetDirectory = Directory(targetPath);
    final temporaryPath = '$targetPath.tmp';
    final temporaryDirectory = Directory(temporaryPath);
    if (await temporaryDirectory.exists()) {
      await temporaryDirectory.delete(recursive: true);
    }
    await temporaryDirectory.create(recursive: true);

    try {
      for (final assetPath in directAssets) {
        final relativePath = assetPath.substring(prefix.length);
        await _copyBundledMobileVlmAsset(
          assetPath,
          temporaryPath,
          relativePath,
        );
        await updateProgress();
      }

      for (final entry in chunkedAssets.entries) {
        final chunkPaths = entry.value..sort();
        await _rebuildBundledMobileVlmAsset(
          chunkPaths,
          temporaryPath,
          entry.key,
          onChunkCopied: updateProgress,
        );
      }

      final copyIsComplete = await _directoryMatchesBundledMobileVlmSpec(
        temporaryPath,
        localName,
      );
      if (!copyIsComplete) {
        throw StateError(
          'La copia incluida de la descripcion visual quedo incompleta al prepararla.',
        );
      }

      if (await targetDirectory.exists()) {
        await targetDirectory.delete(recursive: true);
      }
      await temporaryDirectory.rename(targetPath);
    } catch (_) {
      if (await temporaryDirectory.exists()) {
        await temporaryDirectory.delete(recursive: true);
      }
      rethrow;
    }
  }

  Future<void> installModel({
    void Function(double? progress, String statusMessage, bool isError)?
    onProgress,
  }) async {
    if (hasRemoteEndpoint) {
      onProgress?.call(1, 'LFM 2.5 esta configurado por endpoint', false);
      return;
    }
    if (!_isCactusPlatformSupported) {
      throw StateError(
        'La descripcion visual local no esta soportada en esta plataforma.',
      );
    }
    final localName = activeModelName;
    final directUrl = mobileVlmDirectDownloadUrls[localName];
    if (directUrl != null) {
      final filename = directUrl.split('?').first.split('/').last;
      final success = await DownloadService.downloadAndExtractModels([
        DownloadTask(url: directUrl, filename: filename, folder: localName),
      ], onProgress);
      if (!success) {
        throw StateError(
          'No pude descargar y extraer ${mobileVlmReadableModelName(localName)}.',
        );
      }
      return;
    }

    final installer = CactusLM(enableToolFiltering: false);
    try {
      await installer.downloadModel(
        model: localName,
        downloadProcessCallback: onProgress,
      );
    } catch (error) {
      throw StateError(
        '${mobileVlmReadableModelName(localName)} no esta disponible para descarga local. '
        'Puedes configurar MAGICVISION_VLM_ENDPOINT para LiquidAI/LFM2.5-VL-450M o copiar el modelo a la carpeta de la app. '
        'Detalle: $error',
      );
    } finally {
      installer.unload();
    }
  }

  Future<String?> externalModelPath() async {
    if (!Platform.isAndroid) return null;
    final searchedPaths = await _candidateModelPaths(activeModelName);
    for (final path in searchedPaths) {
      if (!_isPrivateModelPath(path) &&
          await _directoryLooksLikeMobileVlmModel(path)) {
        return path;
      }
    }
    return null;
  }

  Future<void> migrateExternalModel({
    void Function(double? progress, String statusMessage)? onProgress,
  }) async {
    final sourcePath = await externalModelPath();
    if (sourcePath == null) {
      throw StateError(
        'No encontre una copia externa valida de la descripcion visual.',
      );
    }

    final searchedPaths = await _candidateModelPaths(activeModelName);
    final targetPath = searchedPaths.firstWhere(
      _isPrivateModelPath,
      orElse: () => searchedPaths.first,
    );
    if (await _directoryLooksLikeMobileVlmModel(targetPath)) {
      onProgress?.call(
        1,
        'Descripcion visual ya esta en almacenamiento interno',
      );
      return;
    }

    onProgress?.call(0, 'Moviendo descripcion visual a almacenamiento interno');
    await _copyDirectoryTree(
      sourcePath,
      targetPath,
      onProgress: onProgress,
      statusMessage: 'Moviendo descripcion visual a almacenamiento interno',
    );
    if (!await _directoryLooksLikeMobileVlmModel(targetPath)) {
      throw StateError(
        'Copie la descripcion visual al almacenamiento interno, pero faltan archivos.',
      );
    }
    onProgress?.call(
      1,
      'Descripcion visual quedo lista en almacenamiento interno',
    );
  }

  Future<void> _ensureInitialized(String modelPath) {
    final currentInit = _initializing;
    if (currentInit != null) return currentInit;

    final init = () async {
      if (_handle != null && _loadedModelPath == modelPath) return;
      close();
      _lastInitializationError = null;
      final errors = <String>[];
      for (final candidateContextSize in _contextSizeCandidates(contextSize)) {
        debugPrint(
          'Probando MobileVLM en $modelPath con contexto $candidateContextSize',
        );
        final result = await CactusContext.initContext(
          modelPath,
          candidateContextSize,
        );
        final handle = result.$1;
        if (handle != null) {
          _handle = handle;
          _loadedModelPath = modelPath;
          return;
        }
        errors.add('contexto $candidateContextSize: ${result.$2}');
      }
      _lastInitializationError = errors.join(' | ');
      throw StateError(
        'No pude cargar la descripcion visual local desde $modelPath. $_lastInitializationError',
      );
    }();
    _initializing = init;
    return init.whenComplete(() => _initializing = null);
  }

  Future<void> _ensureInitializedWithBundledRepair(String modelPath) async {
    try {
      await _ensureInitialized(modelPath);
      return;
    } catch (error, stackTrace) {
      if (!await hasBundledModel()) rethrow;
      debugPrint(
        'Visual description init failed, rebuilding bundled copy: $error',
      );
      debugPrintStack(stackTrace: stackTrace);
    }

    await installBundledModel(force: true);
    final repairedStatus = await status();
    final repairedPath = repairedStatus.modelPath;
    if (repairedPath == null) {
      throw StateError(repairedStatus.speechMessage);
    }
    await _ensureInitialized(repairedPath);
  }

  void close() {
    final handle = _handle;
    if (handle != null) {
      CactusContext.freeContext(handle);
      _handle = null;
      _loadedModelPath = null;
    }
  }

  void dispose() {
    close();
    if (_ownsHttpClient) {
      _httpClient.close();
    }
  }
}

List<int> _contextSizeCandidates(int preferredContextSize) {
  final candidates = <int>[preferredContextSize, 1536, 1024, 768, 512];
  final seen = <int>{};
  return [
    for (final value in candidates)
      if (value > 0 && seen.add(value)) value,
  ];
}

enum MobileVlmBackend { local, remote }

class MobileVlmModelStatus {
  const MobileVlmModelStatus({
    required this.isPlatformSupported,
    required this.modelName,
    required this.modelId,
    required this.modelPath,
    required this.searchedPaths,
    this.backend = MobileVlmBackend.local,
    this.endpoint,
  });

  final bool isPlatformSupported;
  final String modelName;
  final String modelId;
  final String? modelPath;
  final List<String> searchedPaths;
  final MobileVlmBackend backend;
  final String? endpoint;

  bool get isRemote => backend == MobileVlmBackend.remote;
  bool get isReady => isPlatformSupported && (isRemote || modelPath != null);
  String get displayName => mobileVlmReadableModelName(modelName);

  String get speechMessage {
    if (isRemote) {
      return '$displayName esta configurado por endpoint. '
          'La descripcion completa usara $modelId.';
    }
    if (!isPlatformSupported) {
      return 'La descripcion visual local esta implementada para Android, iOS y macOS. '
          'En este dispositivo no puedo cargar el motor local.';
    }
    if (isReady) {
      return 'Encontre los archivos de ${mobileVlmReadableModelName(modelName)} en el telefono. '
          'Falta probar que el motor local pueda cargarlos antes de decir que esta lista para usar.';
    }
    return '${mobileVlmReadableModelName(modelName)} esta configurado en la app, pero aun no encuentro todos sus archivos. '
        'Tambien puedes decir: descargar modelo visual. Luego podre describir imagenes sin internet.';
  }
}

bool get _isCactusPlatformSupported =>
    Platform.isAndroid || Platform.isIOS || Platform.isMacOS;

Future<List<String>> _candidateModelPaths(String modelName) async {
  final paths = <String>[];
  final documentsDir = await getApplicationDocumentsDirectory();
  paths.add(_joinPath([documentsDir.path, 'models', modelName]));

  final supportDir = await getApplicationSupportDirectory();
  paths.add(_joinPath([supportDir.path, 'models', modelName]));

  if (Platform.isAndroid) {
    final externalDir = await getExternalStorageDirectory();
    if (externalDir != null) {
      paths.add(_joinPath([externalDir.path, 'models', modelName]));
    }
  }

  final seen = <String>{};
  return [
    for (final path in paths)
      if (seen.add(path)) path,
  ];
}

bool _isPrivateModelPath(String path) {
  if (!Platform.isAndroid) return true;
  return !path.contains('/storage/emulated/') && !path.contains('/sdcard/');
}

String _joinPath(List<String> parts) {
  return parts.join(Platform.pathSeparator);
}

String? mobileVlmChunkBasePath(String relativePath) {
  final match = RegExp(r'^(.*)\.part\d{3,}$').firstMatch(relativePath);
  return match?.group(1);
}

Future<void> _copyBundledMobileVlmAsset(
  String assetPath,
  String targetPath,
  String relativePath,
) async {
  final destinationPath = _joinPath([targetPath, ...relativePath.split('/')]);
  final destination = File(destinationPath);
  await destination.parent.create(recursive: true);
  final data = await rootBundle.load(assetPath);
  final bytes = data.buffer.asUint8List(data.offsetInBytes, data.lengthInBytes);
  await destination.writeAsBytes(bytes, flush: false);
}

Future<void> _rebuildBundledMobileVlmAsset(
  List<String> chunkAssetPaths,
  String targetPath,
  String relativePath, {
  required Future<void> Function() onChunkCopied,
}) async {
  final destinationPath = _joinPath([targetPath, ...relativePath.split('/')]);
  final destination = File(destinationPath);
  final temporary = File('$destinationPath.tmp');
  await temporary.parent.create(recursive: true);
  if (await temporary.exists()) await temporary.delete();

  final sink = temporary.openWrite(mode: FileMode.write);
  try {
    for (final chunkAssetPath in chunkAssetPaths) {
      final data = await rootBundle.load(chunkAssetPath);
      final bytes = data.buffer.asUint8List(
        data.offsetInBytes,
        data.lengthInBytes,
      );
      sink.add(bytes);
      await onChunkCopied();
    }
  } finally {
    await sink.close();
  }

  if (await destination.exists()) await destination.delete();
  await temporary.rename(destinationPath);
}

Future<void> _copyDirectoryTree(
  String sourcePath,
  String targetPath, {
  void Function(double? progress, String statusMessage)? onProgress,
  required String statusMessage,
}) async {
  final source = Directory(sourcePath);
  if (!await source.exists()) {
    throw StateError('La carpeta origen no existe: $sourcePath');
  }

  final target = Directory(targetPath);
  final temporary = Directory('$targetPath.tmp');
  if (await temporary.exists()) await temporary.delete(recursive: true);
  await temporary.create(recursive: true);

  final entities = await source
      .list(recursive: true, followLinks: false)
      .toList();
  final totalFiles = entities.whereType<File>().length;
  var copiedFiles = 0;

  try {
    for (final entity in entities) {
      final relativePath = entity.path.substring(sourcePath.length + 1);
      final destinationPath = _joinPath([
        temporary.path,
        ...relativePath.split(Platform.pathSeparator),
      ]);

      if (entity is Directory) {
        await Directory(destinationPath).create(recursive: true);
        continue;
      }
      if (entity is File) {
        final destination = File(destinationPath);
        await destination.parent.create(recursive: true);
        await entity.copy(destination.path);
        copiedFiles++;
        if (totalFiles > 0 && copiedFiles % 25 == 0) {
          onProgress?.call(copiedFiles / totalFiles, statusMessage);
        }
      }
    }

    if (await target.exists()) await target.delete(recursive: true);
    await temporary.rename(target.path);
  } catch (_) {
    if (await temporary.exists()) await temporary.delete(recursive: true);
    rethrow;
  }
}

Future<bool> _directoryLooksLikeMobileVlmModel(String path) async {
  try {
    final directory = Directory(path);
    if (!await directory.exists()) return false;

    final requiredFiles = [
      'config.txt',
      'tokenizer.json',
      'token_embeddings.weights',
      'vocab.txt',
    ];
    for (final filename in requiredFiles) {
      if (!await File(_joinPath([path, filename])).exists()) return false;
    }

    await for (final entity in directory.list(followLinks: false)) {
      if (entity is! File) continue;
      final filename = entity.uri.pathSegments.last;
      final isLayerWeight =
          filename.endsWith('.weights') &&
          (filename.startsWith('layer_') ||
              filename.startsWith('vision_layer_'));
      if (isLayerWeight) return true;
    }
  } on FileSystemException catch (error) {
    debugPrint('No puedo revisar la carpeta del modelo $path: $error');
    return false;
  }
  return false;
}

Future<bool> _directoryMatchesBundledMobileVlmSpec(
  String path,
  String modelName,
) async {
  if (!await _directoryLooksLikeMobileVlmModel(path)) return false;

  final spec = mobileVlmBundledModelSpecs[modelName];
  if (spec == null) return true;

  final directory = Directory(path);
  var fileCount = 0;
  var totalBytes = 0;
  try {
    await for (final entity in directory.list(
      recursive: true,
      followLinks: false,
    )) {
      if (entity is! File) continue;
      fileCount++;
      totalBytes += await entity.length();
    }
  } on FileSystemException catch (error) {
    debugPrint('No puedo validar la carpeta del modelo $path: $error');
    return false;
  }

  return fileCount == spec.fileCount && totalBytes == spec.totalBytes;
}

Future<void> _ensureModelFolders(List<String> paths) async {
  for (final path in paths) {
    try {
      await Directory(path).create(recursive: true);
    } catch (_) {
      // Some platform directories may be unavailable; the next path may work.
    }
  }
}

Future<List<String>> _bundledMobileVlmAssetPaths(String modelName) async {
  try {
    final manifestText = await rootBundle.loadString('AssetManifest.json');
    final manifest = jsonDecode(manifestText);
    if (manifest is! Map) return const [];
    final prefix = '$mobileVlmBundledAssetRoot/$modelName/';
    final paths =
        manifest.keys
            .whereType<String>()
            .where((path) => path.startsWith(prefix))
            .toList()
          ..sort();
    return paths;
  } catch (_) {
    return const [];
  }
}

String mobileVlmReadableModelName(String modelName) {
  return switch (modelName) {
    'lfm2.5-vl-450m' => 'LFM 2.5 VL 450M',
    'lfm2-5-vl-450m' => 'LFM 2.5 VL 450M',
    'lfm2-vl-450m' => 'LFM 2 VL 450M',
    'lfm2-vl-1.6b' => 'LFM 2 VL 1.6B',
    _ => modelName,
  };
}

String friendlyMobileVlmError(Object error) {
  final raw = error.toString().trim();
  final message = raw
      .replaceFirst(RegExp(r'^(StateError|Exception|FormatException):\s*'), '')
      .trim();
  final lower = message.toLowerCase();
  if (lower.contains('failed to initialize model') ||
      lower.contains('failed to initialize context') ||
      lower.contains('contexto 512') ||
      lower.contains('check weight files')) {
    return 'El motor local encontro la carpeta, pero no pudo inicializar los pesos del modelo. Ya intente cargarlo con contextos mas pequenos y reconstruir la copia incluida desde la app. Reinstala este APK si el error sigue apareciendo.';
  }
  if (lower.contains('out of memory') ||
      lower.contains('cannot allocate') ||
      lower.contains('bad_alloc')) {
    return 'El telefono se quedo sin memoria al cargar el modelo. Cierra otras apps y vuelve a intentarlo.';
  }
  if (lower.contains('magicvision_vlm_endpoint') ||
      lower.contains('endpoint') ||
      lower.contains('connection refused') ||
      lower.contains('failed host lookup') ||
      lower.contains('xmlhttprequest error') ||
      lower.contains('lfm 2.5 respondio')) {
    return 'No pude conectar con el motor LFM 2.5. Verifica que el servidor visual este encendido y que MAGICVISION_VLM_ENDPOINT apunte a la IP correcta.';
  }
  if (lower.contains('no devolvio texto') || lower.contains('empty')) {
    return 'El modelo cargo, pero no devolvio texto para esta foto. Intenta con mas luz o apunta directo al objeto.';
  }
  if (message.isEmpty) {
    return 'El motor local devolvio un error sin detalle.';
  }
  return message.length > 240 ? '${message.substring(0, 240)}.' : message;
}

String _resultText(CactusCompletionResult result) {
  final text = result.response.trim();
  if (text.isEmpty) return '';
  return text;
}

String _mimeTypeForImagePath(String path) {
  final lower = path.toLowerCase();
  if (lower.endsWith('.png')) return 'image/png';
  if (lower.endsWith('.webp')) return 'image/webp';
  return 'image/jpeg';
}

String mobileVlmPromptForDetections(List<Detection> detections) {
  final reliableDetections = detections
      .where(isReliableSceneDetection)
      .toList();
  final hints = reliableDetections.isEmpty
      ? 'El detector rapido no encontro objetos confiables todavia.'
      : 'El detector rapido ve: ${summarizeLabelsForSpeech(reliableDetections)}.';
  return 'Describe esta escena para una persona con baja vision. '
      'Las detecciones YOLO26m son indicios de esta misma captura, no certezas. '
      'Si agregas algo no confirmado por YOLO, dilo como posible y solo si es evidente. '
      'No inventes objetos, personas, vehiculos, huecos, escaleras ni senales. '
      'Responde en espanol latino, natural, seguro y en maximo dos frases. '
      'Prioriza ubicacion espacial y obstaculos. No asegures paso libre ni inventes distancias, movimiento o sentido de escaleras. '
      'Si ves dos personas, empieza con "Veo dos personas" y luego ubica '
      'a cada una con izquierda, derecha o al frente. '
      'No expliques el analisis. '
      '$hints';
}

String extractMobileVlmText(Object? value) {
  if (value == null) return '';
  if (value is String) return value.trim();
  if (value is List) {
    for (final item in value) {
      final text = extractMobileVlmText(item);
      if (text.isNotEmpty) return text;
    }
    return '';
  }
  if (value is Map) {
    for (final key in const [
      'description',
      'answer',
      'response',
      'text',
      'caption',
      'content',
      'message',
    ]) {
      final text = extractMobileVlmText(value[key]);
      if (text.isNotEmpty) return text;
    }

    final choicesText = extractMobileVlmText(value['choices']);
    if (choicesText.isNotEmpty) return choicesText;
  }
  return '';
}

String normalizeMobileVlmDescription(String text) {
  return text
      .replaceAll(
        RegExp('^(respuesta|answer|assistant)\\s*:?\\s*', caseSensitive: false),
        '',
      )
      .replaceAll(RegExp('\\s+'), ' ')
      .trim();
}

String groundedMobileVlmSpeech({
  required String vlmDescription,
  required AnalysisResult? analysis,
  required List<Detection> detections,
}) {
  final clean = normalizeMobileVlmDescription(vlmDescription);
  final yoloGrounding = yoloGroundingForSpeech(
    analysis: analysis,
    detections: detections,
  );
  final unsupportedClaim = RegExp(
    r'\b(metros?|cent[ií]metros?|kil[oó]metros?|se acerca|acerc[aá]ndose|camino (est[aá] )?libre|ruta (est[aá] )?libre|puerta[^.]{0,35}(abierta|cerrada))\b',
    caseSensitive: false,
  ).hasMatch(clean);
  if (clean.isEmpty || looksUnhelpfulMobileVlmText(clean) || unsupportedClaim) {
    return yoloGrounding.isEmpty
        ? 'No pude obtener una descripcion visual confiable. Usa que ves para una lectura rapida.'
        : yoloGrounding;
  }

  if (yoloGrounding.isEmpty) return clean;
  return '$yoloGrounding Contexto visual: $clean';
}

String yoloGroundingForSpeech({
  required AnalysisResult? analysis,
  required List<Detection> detections,
}) {
  if (analysis != null) {
    final objects = analysis.detectedObjects.isEmpty
        ? 'sin objetos peligrosos confirmados'
        : analysis.detectedObjects.take(4).join(', ');
    return 'Detectado por YOLO: ${analysis.description} '
        'Objetos: $objects. Recomendacion: ${analysis.recommendation}';
  }

  final reliable = detections.where(isReliableSceneDetection).toList();
  if (reliable.isEmpty) return '';
  return 'Detectado por YOLO: ${summarizeSceneForSpeech(reliable)}';
}

bool looksUnhelpfulMobileVlmText(String text) {
  final lower = text.toLowerCase();
  if (lower.length < 18) return true;
  const badFragments = [
    'no puedo ver',
    'no puedo analizar',
    'no tengo acceso',
    'lo siento',
    'como modelo',
    'no se muestra',
    'imagen no disponible',
    'no hay suficiente informacion',
    'no puedo determinar',
  ];
  return badFragments.any(lower.contains);
}

Position? chooseBestPosition({
  required Position? fresh,
  required Position? cached,
  required DateTime now,
  bool navigation = false,
}) {
  if (cached == null) return fresh;

  final maxCachedAge = navigation
      ? const Duration(minutes: 2)
      : const Duration(minutes: 5);
  var cachedAge = now.difference(cached.timestamp);
  if (cachedAge.isNegative) cachedAge = -cachedAge;
  final cachedIsRecent = cachedAge <= maxCachedAge;
  if (fresh == null) return cachedIsRecent ? cached : null;
  if (!cachedIsRecent) return fresh;

  final freshAccuracy = normalizedAccuracyMeters(fresh);
  final cachedAccuracy = normalizedAccuracyMeters(cached);
  if (freshAccuracy > 60 && cachedAccuracy + 10 < freshAccuracy) {
    return cached;
  }
  return fresh;
}

double normalizedAccuracyMeters(Position position) {
  final accuracy = position.accuracy;
  if (!accuracy.isFinite || accuracy <= 0) return 9999;
  return accuracy;
}

bool isReliableHeading(Position position) {
  final heading = position.heading;
  if (!heading.isFinite || heading < 0) return false;
  final headingAccuracy = position.headingAccuracy;
  if (headingAccuracy.isFinite &&
      headingAccuracy > 0 &&
      headingAccuracy <= 50) {
    return true;
  }
  return position.speed.isFinite && position.speed >= 0.5;
}

String currentLocationSpeech({
  required Position position,
  required String address,
}) {
  final coordinates =
      '${position.latitude.toStringAsFixed(5)}, ${position.longitude.toStringAsFixed(5)}';
  final accuracy = gpsAccuracyForSpeech(normalizedAccuracyMeters(position));
  final base = address.trim().isEmpty
      ? 'No tengo una direccion exacta, pero te ubico en coordenadas $coordinates.'
      : 'Te tengo cerca de $address.';
  final coordinatesText = address.trim().isEmpty
      ? ''
      : 'Coordenadas $coordinates.';
  return [
    base,
    accuracy,
    coordinatesText,
  ].where((part) => part.trim().isNotEmpty).join(' ').trim();
}

String gpsAccuracyForSpeech(double accuracyMeters) {
  if (!accuracyMeters.isFinite || accuracyMeters >= 9000) {
    return 'La precision GPS no esta disponible.';
  }
  final meters = accuracyMeters.round();
  if (accuracyMeters <= 10) {
    return 'Precision muy buena, alrededor de $meters metros.';
  }
  if (accuracyMeters <= navigationGoodAccuracyMeters) {
    return 'Precision GPS de unos $meters metros.';
  }
  if (accuracyMeters <= 80) {
    return 'Precision media, alrededor de $meters metros; tomalo como aproximado.';
  }
  return 'La senal GPS esta baja, alrededor de $meters metros de margen.';
}

String cleanNavigationDestination(String value) {
  return value
      .replaceAll(
        RegExp(
          r'\b(por favor|porfa|caminando|a pie|andando|en carro|en auto|en taxi|ahora)\b',
        ),
        ' ',
      )
      .replaceAll(RegExp(r'\s+'), ' ')
      .trim();
}

List<String> navigationSearchQueries(String destination, {Placemark? near}) {
  final clean = cleanNavigationDestination(destination);
  if (clean.isEmpty) return const [];

  final context = _uniqueTextParts([
    near?.locality,
    near?.subAdministrativeArea,
    near?.administrativeArea,
    near?.country,
  ]);
  final queries = <String>[];
  if (context.isNotEmpty) {
    queries.add('$clean, ${context.take(3).join(', ')}');
  }
  if (context.length > 1) {
    queries.add('$clean, ${context.take(2).join(', ')}');
  }
  queries.add(clean);
  return _uniqueTextParts(queries);
}

String addressFromPlacemark(Placemark place) {
  final parts = _uniqueTextParts([
    place.street,
    place.subLocality,
    place.locality,
    place.subAdministrativeArea,
    place.administrativeArea,
    place.country,
  ]);
  return parts.join(', ');
}

List<String> _uniqueTextParts(Iterable<String?> values) {
  final seen = <String>{};
  final result = <String>[];
  for (final raw in values) {
    final value = (raw ?? '').replaceAll(RegExp(r'\s+'), ' ').trim();
    if (value.isEmpty) continue;
    final normalized = normalizeSpeechText(value);
    if (seen.add(normalized)) result.add(value);
  }
  return result;
}

String navigationCueForSpeech({
  required String targetName,
  required String direction,
  required double distanceMeters,
  required double accuracyMeters,
  bool movingAway = false,
  bool opening = false,
}) {
  final distance = navigationDistanceForSpeech(distanceMeters);
  final accuracyNote = accuracyMeters > 80
      ? 'La senal GPS esta baja; usa esto como orientacion aproximada.'
      : accuracyMeters > navigationGoodAccuracyMeters
      ? 'La precision todavia es media, asi que ire ajustando la guia.'
      : '';

  final lead = opening
      ? 'Listo. Encontre $targetName.'
      : movingAway
      ? 'Te estas alejando un poco.'
      : 'Vas bien.';
  final action = opening
      ? '$targetName queda $direction, a $distance.'
      : '$targetName queda $direction, a $distance.';
  final controls = opening
      ? 'Te dare avisos de rumbo y distancia. Di abrir Maps para ruta peatonal por calles, o cancelar ruta para detenerme.'
      : '';
  return [
    lead,
    action,
    accuracyNote,
    controls,
  ].where((part) => part.trim().isNotEmpty).join(' ').trim();
}

String navigationDistanceForSpeech(double meters) {
  if (!meters.isFinite || meters < 0) return 'una distancia desconocida';
  if (meters < 15) return 'menos de 15 metros';
  if (meters < 100) return '${((meters / 5).round() * 5).round()} metros';
  if (meters < 1000) return '${((meters / 10).round() * 10).round()} metros';

  final kilometers = meters / 1000;
  if (kilometers < 10) {
    final text = kilometers.toStringAsFixed(1).replaceAll('.0', '');
    return text == '1' ? '1 kilometro' : '$text kilometros';
  }
  return '${kilometers.round()} kilometros';
}

String navigationDistanceCompact(double meters) {
  if (!meters.isFinite || meters < 0) return '--';
  if (meters < 1000) return '${meters.round()} m';
  return '${(meters / 1000).toStringAsFixed(1).replaceAll('.0', '')} km';
}

String normalizeSpeechText(String value) {
  return value
      .toLowerCase()
      .replaceAll('\u00e1', 'a')
      .replaceAll('\u00e9', 'e')
      .replaceAll('\u00ed', 'i')
      .replaceAll('\u00f3', 'o')
      .replaceAll('\u00fa', 'u')
      .replaceAll('\u00fc', 'u')
      .replaceAll('\u00f1', 'n')
      .replaceAll(RegExp('[^a-z0-9 ]'), ' ')
      .replaceAll(RegExp(' +'), ' ')
      .trim();
}

bool hasSpeechWord(String command, String word) {
  return RegExp('\\b${RegExp.escape(word)}\\b').hasMatch(command);
}

bool mentionsMobileVlmForSpeech(String value) {
  final command = normalizeSpeechText(value);
  if (command.isEmpty) return false;

  return command.contains('mobilevlm') ||
      command.contains('mobile vlm') ||
      command.contains('mobile v l m') ||
      command.contains('movil vlm') ||
      command.contains('movil v l m') ||
      command.contains('v l m') ||
      command.contains('b l m') ||
      command.contains('uve ele eme') ||
      command.contains('ve ele eme') ||
      command.contains('be ele eme') ||
      command.contains('vision lenguaje') ||
      command.contains('vision de lenguaje') ||
      command.contains('vision local') ||
      command.contains('modelo visual') ||
      command.contains('modelo de vision') ||
      command.contains('modelo vision') ||
      hasSpeechWord(command, 'vlm') ||
      hasSpeechWord(command, 'blm');
}

bool isMobileVlmAnalyzeCommandForSpeech(String value) {
  final command = normalizeSpeechText(value);
  if (command.isEmpty) return false;

  final asksForAnalysis =
      command.contains('analiza imagen') ||
      command.contains('analiza la imagen') ||
      command.contains('analizar imagen') ||
      command.contains('analizar la imagen') ||
      command.contains('analiza foto') ||
      command.contains('analizar foto') ||
      command.contains('describe imagen') ||
      command.contains('describe la imagen') ||
      command.contains('descripcion inteligente') ||
      command.contains('vision inteligente');

  return mentionsMobileVlmForSpeech(command) || asksForAnalysis;
}

bool isMobileVlmInstallCommandForSpeech(String value) {
  final command = normalizeSpeechText(value);
  final asksInstall =
      command.contains('descarga') ||
      command.contains('descargar') ||
      command.contains('instala') ||
      command.contains('instalar') ||
      command.contains('prepara') ||
      command.contains('preparar') ||
      command.contains('baja el modelo') ||
      command.contains('cargar modelo');
  if (!asksInstall) return false;
  return mentionsMobileVlmForSpeech(command) ||
      command.contains('modelo visual') ||
      command.contains('modelo de vision') ||
      command.contains('modelo pesado');
}

bool isMobileVlmStatusCommandForSpeech(String value) {
  final command = normalizeSpeechText(value);
  final asksStatus =
      command.contains('estado') ||
      command.contains('estatus') ||
      command.contains('carpeta') ||
      command.contains('ruta del modelo') ||
      command.contains('modelo instalado') ||
      command.contains('modelo preparado') ||
      command.contains('modelo listo') ||
      command.contains('que modelo tiene') ||
      command.contains('que modelo tienes') ||
      command.contains('que modelo usa') ||
      command.contains('que modelo usas') ||
      command.contains('cual modelo tiene') ||
      command.contains('cual modelo tienes') ||
      command.contains('cual modelo usa') ||
      command.contains('cual modelo usas') ||
      command.contains('verifica modelo');
  if (!asksStatus) return false;
  return mentionsMobileVlmForSpeech(command) ||
      command.contains('estado mobile') ||
      command.contains('estado movil') ||
      command.contains('estado del modelo') ||
      command.contains('estado modelo') ||
      command.contains('modelo visual') ||
      command.contains('modelo de vision') ||
      command.contains('modelo listo') ||
      command.contains('modelo instalado') ||
      command.contains('que modelo tiene') ||
      command.contains('que modelo tienes') ||
      command.contains('que modelo usa') ||
      command.contains('que modelo usas') ||
      command.contains('cual modelo tiene') ||
      command.contains('cual modelo tienes') ||
      command.contains('cual modelo usa') ||
      command.contains('cual modelo usas');
}

String speechRecognitionStatusMessage(String errorMsg) {
  final normalized = normalizeSpeechText(errorMsg);
  if (normalized.contains('no_match') ||
      normalized.contains('no match') ||
      normalized.contains('no speech') ||
      normalized.contains('speech_timeout') ||
      normalized.contains('speech timeout') ||
      normalized.contains('timeout')) {
    return 'No te escuche claro. Prueba: analiza imagen o estado modelo visual';
  }
  if (normalized.contains('language') || normalized.contains('idioma')) {
    return 'Voz: falta el idioma espanol en el reconocimiento del telefono';
  }
  if (normalized.contains('permission') || normalized.contains('permiso')) {
    return 'Voz: falta permiso de microfono';
  }
  if (normalized.contains('network') || normalized.contains('red')) {
    return 'Voz: el reconocedor necesita conexion o un paquete local de idioma';
  }
  return 'Voz: $errorMsg';
}

bool isCancelAnalysisCommand(String command) =>
    RegExp(r'\b(?:cancelar|cancela) (?:el )?analisis\b').hasMatch(command);

bool isStopSpeakingCommand(String command, {bool whileSpeaking = false}) {
  final normalized = normalizeSpeechText(command);
  if (normalized.isEmpty) return false;

  const exactCommands = {
    'alto',
    'cancelar',
    'cancela',
    'para',
    'parar',
    'detente',
    'silencio',
    'callate',
    'stop',
  };
  if (exactCommands.contains(normalized)) return true;
  if (hasSpeechWord(normalized, 'alto')) return true;

  const phrases = [
    'para de hablar',
    'deja de hablar',
    'deja de sonar',
    'haz silencio',
    'corta la voz',
    'cortar la voz',
    'deten la voz',
    'detener la voz',
    'deten el audio',
    'detener el audio',
    'cancelar voz',
    'cancelar la voz',
    'cancela la voz',
    'cancelar audio',
    'cancela el audio',
  ];
  if (phrases.any(normalized.contains)) return true;

  if (!whileSpeaking) return false;
  return normalized == 'cancelar por favor' ||
      normalized == 'silencio por favor' ||
      normalized == 'alto por favor';
}

Map<String, String> _stringVoiceMap(Map<dynamic, dynamic> rawVoice) {
  return rawVoice.map(
    (key, value) => MapEntry(key.toString(), value?.toString() ?? ''),
  );
}

int _spanishVoiceScore(Map<String, String> voice) {
  final locale = (voice['locale'] ?? '').toLowerCase().replaceAll('_', '-');
  final name = (voice['name'] ?? '').toLowerCase();
  var score = 0;

  if (locale == 'es-pe') {
    score += 60;
  } else if (locale == 'es-mx') {
    score += 56;
  } else if (locale == 'es-419') {
    score += 54;
  } else if (locale == 'es-us') {
    score += 50;
  } else if (locale == 'es-es') {
    score += 45;
  } else if (locale.startsWith('es')) {
    score += 35;
  }

  if (name.contains('neural') ||
      name.contains('natural') ||
      name.contains('wavenet')) {
    score += 18;
  }
  if (name.contains('premium') || name.contains('enhanced')) score += 10;
  if (name.contains('google')) score += 6;
  if (name.contains('pico') || name.contains('espeak')) score -= 40;
  if ((voice['network_required'] ?? '').toLowerCase() == 'true') score += 4;

  final quality = int.tryParse(voice['quality'] ?? '');
  if (quality != null) score += (quality / 100).round();

  return score;
}

String formatLongNumbersForSpeech(String text) {
  return text.replaceAllMapped(RegExp(r'\d(?:[\s.,-]?\d){3,}'), (match) {
    final digitsText = match.group(0)!.replaceAll(RegExp(r'\D'), '');
    final value = int.tryParse(digitsText);
    if (value != null &&
        digitsText.length <= 6 &&
        !digitsText.startsWith('0')) {
      return numberToSpanishWords(value);
    }
    return digitsText.split('').join(' ');
  });
}

String numberToSpanishWords(int value) {
  if (value == 0) return 'cero';
  if (value < 0 || value > 999999) {
    return value.toString().split('').join(' ');
  }
  if (value < 1000) return _numberBelowThousandToSpanish(value);

  final thousands = value ~/ 1000;
  final rest = value % 1000;
  final thousandsText = thousands == 1
      ? 'mil'
      : '${_numberBelowThousandToSpanish(thousands)} mil';
  if (rest == 0) return thousandsText;
  return '$thousandsText ${_numberBelowThousandToSpanish(rest)}';
}

String _numberBelowThousandToSpanish(int value) {
  const units = [
    '',
    'uno',
    'dos',
    'tres',
    'cuatro',
    'cinco',
    'seis',
    'siete',
    'ocho',
    'nueve',
    'diez',
    'once',
    'doce',
    'trece',
    'catorce',
    'quince',
    'dieciseis',
    'diecisiete',
    'dieciocho',
    'diecinueve',
    'veinte',
    'veintiuno',
    'veintidos',
    'veintitres',
    'veinticuatro',
    'veinticinco',
    'veintiseis',
    'veintisiete',
    'veintiocho',
    'veintinueve',
  ];
  const tens = {
    30: 'treinta',
    40: 'cuarenta',
    50: 'cincuenta',
    60: 'sesenta',
    70: 'setenta',
    80: 'ochenta',
    90: 'noventa',
  };
  const hundreds = {
    100: 'cien',
    200: 'doscientos',
    300: 'trescientos',
    400: 'cuatrocientos',
    500: 'quinientos',
    600: 'seiscientos',
    700: 'setecientos',
    800: 'ochocientos',
    900: 'novecientos',
  };

  if (value < 30) return units[value];
  if (value < 100) {
    final ten = (value ~/ 10) * 10;
    final unit = value % 10;
    if (unit == 0) return tens[ten]!;
    return '${tens[ten]} y ${units[unit]}';
  }
  if (value == 100) return 'cien';
  final hundred = (value ~/ 100) * 100;
  final rest = value % 100;
  final hundredText = hundred == 100 ? 'ciento' : hundreds[hundred]!;
  if (rest == 0) return hundredText;
  return '$hundredText ${_numberBelowThousandToSpanish(rest)}';
}

String formatTimeForSpeech(DateTime dateTime) {
  final minute = dateTime.minute.toString().padLeft(2, '0');
  return '${dateTime.hour} y $minute';
}

String formatDateForSpeech(DateTime dateTime) {
  const weekdays = [
    'lunes',
    'martes',
    'miercoles',
    'jueves',
    'viernes',
    'sabado',
    'domingo',
  ];
  const months = [
    'enero',
    'febrero',
    'marzo',
    'abril',
    'mayo',
    'junio',
    'julio',
    'agosto',
    'septiembre',
    'octubre',
    'noviembre',
    'diciembre',
  ];
  final weekday = weekdays[dateTime.weekday - 1];
  final month = months[dateTime.month - 1];
  return '$weekday ${dateTime.day} de $month de ${dateTime.year}';
}

Duration? parseSpanishDuration(String command) {
  final normalized = command.replaceAll(RegExp(r'[,.;]'), ' ');
  final directMatch = RegExp(
    r'(\d+)\s*(segundo|segundos|minuto|minutos|hora|horas)',
  ).firstMatch(normalized);
  if (directMatch != null) {
    return _durationFromAmountAndUnit(
      int.parse(directMatch.group(1)!),
      directMatch.group(2)!,
    );
  }

  final words = normalized.split(RegExp(r'\s+'));
  for (var index = 0; index < words.length; index++) {
    final unit = words[index];
    if (!_isDurationUnit(unit)) continue;
    final amountWords = words
        .sublist(math.max(0, index - 4), index)
        .where((word) => word != 'en' && word != 'de' && word != 'por')
        .toList();
    for (var start = 0; start < amountWords.length; start++) {
      final amount = spanishNumberToInt(amountWords.sublist(start));
      if (amount != null) return _durationFromAmountAndUnit(amount, unit);
    }
  }
  return null;
}

Duration _durationFromAmountAndUnit(int amount, String unit) {
  final safeAmount = amount.clamp(1, 24 * 60).toInt();
  if (unit.startsWith('segundo')) return Duration(seconds: safeAmount);
  if (unit.startsWith('hora')) return Duration(hours: safeAmount);
  return Duration(minutes: safeAmount);
}

bool _isDurationUnit(String word) {
  return word.startsWith('segundo') ||
      word.startsWith('minuto') ||
      word.startsWith('hora');
}

int? spanishNumberToInt(List<String> words) {
  if (words.isEmpty) return null;
  final compact = words.join(' ');
  final direct = int.tryParse(compact);
  if (direct != null) return direct;

  const units = {
    'un': 1,
    'una': 1,
    'uno': 1,
    'dos': 2,
    'tres': 3,
    'cuatro': 4,
    'cinco': 5,
    'seis': 6,
    'siete': 7,
    'ocho': 8,
    'nueve': 9,
    'diez': 10,
    'once': 11,
    'doce': 12,
    'trece': 13,
    'catorce': 14,
    'quince': 15,
    'dieciseis': 16,
    'diecisiete': 17,
    'dieciocho': 18,
    'diecinueve': 19,
    'veinte': 20,
    'veintiuno': 21,
    'veintidos': 22,
    'veintitres': 23,
    'veinticuatro': 24,
    'veinticinco': 25,
    'veintiseis': 26,
    'veintisiete': 27,
    'veintiocho': 28,
    'veintinueve': 29,
  };
  const tens = {
    'treinta': 30,
    'cuarenta': 40,
    'cincuenta': 50,
    'sesenta': 60,
    'setenta': 70,
    'ochenta': 80,
    'noventa': 90,
  };

  if (words.length == 1) {
    return units[words.first] ?? tens[words.first];
  }
  final first = tens[words.first];
  final last = units[words.last];
  if (first != null && words.contains('y') && last != null && last < 10) {
    return first + last;
  }
  return null;
}

String extractReminderMessage(String command) {
  var message = command
      .replaceAll('recuerdame', '')
      .replaceAll('recordarme', '')
      .replaceAll('recordatorio', '')
      .trim();
  message = message.replaceFirst(RegExp(r'\ben\b.+$'), '').trim();
  message = message.replaceFirst(RegExp(r'\bpara\b'), '').trim();
  if (message.isEmpty) return 'tienes algo pendiente';
  return message;
}

ParsedVoiceContact? parseVoiceContactCommand(String command) {
  var working = command
      .replaceFirst(
        RegExp(
          r'\b(agrega|agregar|guarda|guardar|registra|registrar)\s+contacto\b',
        ),
        '',
      )
      .replaceAll(RegExp(r'\bcomo\s+sos\b'), ' sos ')
      .replaceAll(RegExp(r'\bpara\s+sos\b'), ' sos ')
      .trim();
  final isEmergency =
      working.contains('sos') ||
      working.contains('emergencia') ||
      working.contains('auxilio');
  working = working
      .replaceAll(RegExp(r'\b(sos|emergencia|auxilio)\b'), ' ')
      .replaceAll(RegExp(r'\s+'), ' ')
      .trim();

  final marker = RegExp(
    r'\b(numero|telefono|celular|fono)\b',
  ).firstMatch(working);
  if (marker == null) return null;
  final name = working.substring(0, marker.start).trim();
  final phoneWords = working.substring(marker.end).trim().split(RegExp(r'\s+'));
  final phone = normalizePeruMobilePhone(phoneFromSpanishWords(phoneWords));
  if (name.isEmpty || phone == null) return null;
  return ParsedVoiceContact(name: name, phone: phone, isEmergency: isEmergency);
}

String contactNameFromWizardSpeech(String spokenText) {
  var name = normalizeSpeechText(spokenText);
  if (name.isEmpty) return '';

  name = name
      .replaceFirst(RegExp(r'^(el )?nombre( del contacto)?( es)?\b'), '')
      .replaceFirst(
        RegExp(r'^(se llama|llamalo|guardar como|guardalo como|es)\b'),
        '',
      )
      .replaceFirst(RegExp(r'^(mi contacto|contacto)\b'), '')
      .replaceAll(RegExp(r'\b(numero|telefono|celular|fono)\b.*$'), '')
      .replaceAll(RegExp(r'\s+'), ' ')
      .trim();
  if (name.isEmpty) return '';
  if ({'si', 'no', 'sos', 'cancelar', 'alto'}.contains(name)) return '';
  return titleCaseContactName(name);
}

String titleCaseContactName(String name) {
  return name
      .split(RegExp(r'\s+'))
      .where((word) => word.isNotEmpty)
      .map((word) {
        if (word.length == 1) return word.toUpperCase();
        return '${word[0].toUpperCase()}${word.substring(1)}';
      })
      .join(' ');
}

String? peruMobilePhoneFromSpeech(String speech) {
  final words = normalizeSpeechText(speech).split(RegExp(r'\s+'));
  return normalizePeruMobilePhone(phoneFromSpanishWords(words));
}

String? normalizePeruMobilePhone(String value) {
  final digits = value.replaceAll(RegExp(r'\D'), '');
  if (digits.length == 9 && digits.startsWith('9')) {
    return '+51$digits';
  }
  if (digits.length == 11 &&
      digits.startsWith('51') &&
      digits.substring(2).startsWith('9')) {
    return '+$digits';
  }
  if (digits.length == 12 &&
      digits.startsWith('051') &&
      digits.substring(3).startsWith('9')) {
    return '+51${digits.substring(3)}';
  }
  return null;
}

bool? yesNoFromSpeech(String command) {
  final normalized = normalizeSpeechText(command);
  if (normalized.isEmpty) return null;
  if (hasSpeechWord(normalized, 'no') ||
      normalized.contains('nop') ||
      normalized.contains('normal') ||
      normalized.contains('negativo') ||
      normalized.contains('no es sos')) {
    return false;
  }
  if (hasSpeechWord(normalized, 'si') ||
      normalized.contains('claro') ||
      normalized.contains('correcto') ||
      normalized.contains('afirmativo') ||
      normalized.contains('como sos') ||
      normalized.contains('emergencia') ||
      hasSpeechWord(normalized, 'sos')) {
    return true;
  }
  return null;
}

String phoneFromSpanishWords(List<String> words) {
  const digits = {
    'cero': '0',
    'uno': '1',
    'un': '1',
    'una': '1',
    'dos': '2',
    'tres': '3',
    'cuatro': '4',
    'cinco': '5',
    'seis': '6',
    'siete': '7',
    'siente': '7',
    'ocho': '8',
    'nueve': '9',
  };
  final buffer = StringBuffer();
  for (final word in words) {
    final clean = word.replaceAll(RegExp(r'[^0-9a-z+]'), '');
    if (clean.isEmpty) continue;
    if (clean.startsWith('+') && buffer.isEmpty) {
      buffer.write('+');
      buffer.write(clean.replaceAll(RegExp(r'[^0-9]'), ''));
      continue;
    }
    final directDigits = clean.replaceAll(RegExp(r'[^0-9]'), '');
    if (directDigits.isNotEmpty) {
      buffer.write(directDigits);
      continue;
    }
    final digit = digits[clean];
    if (digit != null) buffer.write(digit);
  }
  return buffer.toString();
}

String formatDurationForSpeech(Duration duration) {
  final seconds = duration.inSeconds <= 0 ? 1 : duration.inSeconds;
  if (seconds < 60) {
    return seconds == 1 ? 'un segundo' : '$seconds segundos';
  }
  final minutes = (seconds / 60).round();
  if (minutes < 60) {
    return minutes == 1 ? 'un minuto' : '$minutes minutos';
  }
  final hours = (minutes / 60).round();
  return hours == 1 ? 'una hora' : '$hours horas';
}

String formatDateTimeForHistory(DateTime dateTime) {
  final day = dateTime.day.toString().padLeft(2, '0');
  final month = dateTime.month.toString().padLeft(2, '0');
  final hour = dateTime.hour.toString().padLeft(2, '0');
  final minute = dateTime.minute.toString().padLeft(2, '0');
  return '$day/$month/${dateTime.year} $hour:$minute';
}

enum RiskLevel { danger, caution, info }

const Set<String> _dangerLabels = {
  'bache',
  'grieta',
  'hueco',
  'hoyo',
  'alcantarilla',
  'open_manhole',
  'pothole',
  'crack',
};

const Set<String> _cautionLabels = {
  'persona',
  'mascota',
  'auto',
  'triciclo',
  'autobus',
  'camion',
  'motocicleta',
  'furgoneta',
  'semaforo_rojo',
  'semaforo_amarillo',
};

String summarizeDetectionsForSpeech(List<Detection> detections) {
  final dangers = detections.where(isDangerousDetection).toList();
  if (dangers.isEmpty) {
    return detections.isEmpty
        ? 'No detecto objetos urbanos confiables en este momento.'
        : 'No detecto un riesgo alto inmediato. Para una lectura completa, dime: que ves.';
  }

  dangers.sort(
    (a, b) => _dangerClosenessScore(b).compareTo(_dangerClosenessScore(a)),
  );
  final nearest = dangers.first;
  final label = labelForSpeech(nearest.label);
  final side = dangerDirectionForSpeech(nearest);
  final distanceText = distanceForSpeech(nearest);
  final extraCount = dangers.length - 1;
  final extraText = extraCount > 0
      ? ' Tambien detecto $extraCount peligro${extraCount == 1 ? '' : 's'} mas.'
      : '';
  return 'Peligro $side, $distanceText: $label.$extraText';
}

String summarizeSceneForSpeech(List<Detection> detections) {
  final reliableDetections = detections
      .where(isReliableSceneDetection)
      .toList();
  if (reliableDetections.isEmpty) {
    return 'No detecto nada claro por ahora. Mueve un poco la camara y vuelvo a revisar.';
  }

  final sorted = reliableDetections.toList()
    ..sort((a, b) => _scenePriorityScore(b).compareTo(_scenePriorityScore(a)));
  final people = sorted
      .where((detection) => riskCanonicalLabel(detection.label) == 'persona')
      .toList();
  if (people.length >= 2) {
    final others = sorted
        .where((detection) => riskCanonicalLabel(detection.label) != 'persona')
        .take(3)
        .map(sceneDetectionForSpeech)
        .toList();
    final otherText = others.isEmpty
        ? ''
        : ' Tambien veo ${joinForSpeech(others)}.';
    final extraCount = math.max(
      0,
      sorted.length - people.length - others.length,
    );
    final extraText = extraCount > 0
        ? ' Tambien hay $extraCount elemento${extraCount == 1 ? '' : 's'} mas.'
        : '';
    return '${repeatedSceneGroupForSpeech(people, 'persona')}$otherText$extraText';
  }

  final visible = sorted.take(5).map(sceneDetectionForSpeech).toList();
  final extraCount = sorted.length - visible.length;
  final extraText = extraCount > 0
      ? ' Tambien veo $extraCount elemento${extraCount == 1 ? '' : 's'} mas.'
      : '';
  return 'Veo ${joinForSpeech(visible)}.$extraText';
}

String summarizeLabelsForSpeech(List<Detection> detections) {
  final counts = <String, int>{};
  for (final detection in detections.where(isReliableSceneDetection)) {
    final label = labelForSpeech(detection.label);
    counts[label] = (counts[label] ?? 0) + 1;
  }
  final entries = counts.entries.toList()
    ..sort((a, b) => b.value.compareTo(a.value));
  return entries
      .take(11)
      .map((entry) {
        if (entry.value == 1) return entry.key;
        return '${entry.value} ${entry.key}';
      })
      .join(', ');
}

bool isReliableSceneDetection(Detection detection) {
  final canonicalLabel = riskCanonicalLabel(detection.label);
  if (canonicalLabel == 'persona') {
    return detection.confidence >= sceneSpeechPersonMinConfidence;
  }
  if (isDangerousDetection(detection)) {
    return detection.confidence.isFinite && detection.confidence >= 0;
  }
  return detection.confidence >= sceneSpeechMinConfidence;
}

String sceneDetectionForSpeech(Detection detection) {
  final label = labelForSpeech(detection.label);
  final articleLabel = labelWithArticleForSpeech(label);
  final position = detectionDirectionForSpeech(detection);
  final distance = naturalDistanceForSpeech(detection);
  final risk = riskLevelForLabel(detection.label);
  final suffix = switch (risk) {
    RiskLevel.danger => ', peligro',
    RiskLevel.caution => ', precaucion',
    RiskLevel.info => '',
  };
  return '$articleLabel $position, $distance$suffix';
}

String repeatedSceneGroupForSpeech(List<Detection> detections, String label) {
  final sorted = detections.toList()
    ..sort((a, b) => _scenePriorityScore(b).compareTo(_scenePriorityScore(a)));
  final speechLabel = labelForSpeech(label);
  final plural = pluralLabelForSpeech(speechLabel);
  final details = <String>[];

  for (var index = 0; index < sorted.length; index++) {
    final detection = sorted[index];
    final subject = repeatedSubjectForSpeech(index, speechLabel);
    details.add(
      '$subject esta ${detectionDirectionForSpeech(detection)}, '
      '${naturalDistanceForSpeech(detection)}',
    );
  }

  return 'Veo ${countForSpeech(sorted.length)} $plural. ${joinForSpeech(details)}.';
}

String repeatedSubjectForSpeech(int index, String label) {
  final feminine = isFeminineLabelForSpeech(label);
  if (index == 0) return feminine ? 'Una' : 'Uno';
  if (index == 1) return feminine ? 'la otra' : 'el otro';
  return feminine ? 'otra mas' : 'otro mas';
}

String countForSpeech(int count) {
  return switch (count) {
    0 => 'ningun',
    1 => 'un',
    2 => 'dos',
    3 => 'tres',
    4 => 'cuatro',
    5 => 'cinco',
    _ => '$count',
  };
}

String pluralLabelForSpeech(String label) {
  if (label == 'persona') return 'personas';
  if (label.endsWith('z')) return '${label.substring(0, label.length - 1)}ces';
  if (label.endsWith('s')) return label;
  return '${label}s';
}

String labelWithArticleForSpeech(String label) {
  final clean = label.trim();
  if (clean.isEmpty) return clean;
  return '${isFeminineLabelForSpeech(clean) ? 'una' : 'un'} $clean';
}

bool isFeminineLabelForSpeech(String label) {
  final clean = label.trim();
  if (clean.isEmpty) return false;
  final firstWord = clean.split(' ').first;
  return firstWord.endsWith('a') ||
      firstWord.endsWith('ion') ||
      firstWord == 'grieta' ||
      firstWord == 'alcantarilla' ||
      firstWord == 'persona' ||
      firstWord == 'mascota' ||
      firstWord == 'motocicleta' ||
      firstWord == 'furgoneta';
}

String joinForSpeech(List<String> items) {
  if (items.isEmpty) return '';
  if (items.length == 1) return items.first;
  if (items.length == 2) return '${items.first} y ${items.last}';
  return '${items.take(items.length - 1).join('; ')}; y ${items.last}';
}

String statusForDetections(List<Detection> detections) {
  final nearby = nearestNearbyDanger(detections);
  if (nearby != null) {
    return 'Peligro ${dangerDirectionForSpeech(nearby)}, ${distanceForSpeech(nearby)}';
  }

  final dangerCount = detections.where(isDangerousDetection).length;
  if (dangerCount > 0) {
    return '$dangerCount riesgo${dangerCount == 1 ? '' : 's'} detectado${dangerCount == 1 ? '' : 's'}';
  }

  if (detections.isNotEmpty) {
    return 'Entorno estable: ${detections.length} elemento${detections.length == 1 ? '' : 's'}';
  }

  return 'Escaneando peligros del piso';
}

Detection? nearestNearbyDanger(List<Detection> detections) {
  final nearby =
      detections.where(isDangerousDetection).where(isNearbyDanger).toList()
        ..sort(
          (a, b) =>
              _dangerClosenessScore(b).compareTo(_dangerClosenessScore(a)),
        );
  return nearby.firstOrNull;
}

bool isNearbyDanger(Detection detection) {
  final distance = estimateDangerDistanceMeters(detection);
  if (distance != null && distance <= nearbyDangerMeters) return true;

  final bottomRatio = detection.rect.bottom / detection.sourceSize.height;
  final areaRatio =
      (detection.rect.width * detection.rect.height) /
      (detection.sourceSize.width * detection.sourceSize.height);
  return bottomRatio > 0.60 && areaRatio > 0.025;
}

bool isDangerousDetection(Detection detection) {
  return riskLevelForLabel(detection.label) == RiskLevel.danger;
}

RiskLevel riskLevelForLabel(String label) {
  if (campusClassNames.contains(campusName(label))) {
    return switch (campusPriority(label)) {
      'critical' => RiskLevel.danger,
      'high' || 'medium' => RiskLevel.caution,
      _ => RiskLevel.info,
    };
  }
  final canonical = riskCanonicalLabel(label);
  if (_dangerLabels.contains(canonical)) return RiskLevel.danger;
  if (_cautionLabels.contains(canonical)) return RiskLevel.caution;
  return RiskLevel.info;
}

String riskCanonicalLabel(String label) {
  final normalized = _normalizeLabel(label);
  return switch (normalized) {
    'pothole' || 'hole' || 'huecos' || 'holes' => 'bache',
    'crack' || 'cracks' => 'grieta',
    'open_manhole' || 'manhole' || 'alcantarilla_abierta' => 'alcantarilla',
    'car' || 'automovil' || 'coche' => 'auto',
    'bus' => 'autobus',
    'truck' => 'camion',
    'motorcycle' || 'moto' => 'motocicleta',
    'van' => 'furgoneta',
    'pet' => 'mascota',
    'person' => 'persona',
    'trash_bin' => 'tacho_de_basura',
    _ => normalized,
  };
}

String _normalizeLabel(String label) {
  return label
      .toLowerCase()
      .trim()
      .replaceAll('\u00e1', 'a')
      .replaceAll('\u00e9', 'e')
      .replaceAll('\u00ed', 'i')
      .replaceAll('\u00f3', 'o')
      .replaceAll('\u00fa', 'u')
      .replaceAll('\u00fc', 'u')
      .replaceAll('-', '_')
      .replaceAll(' ', '_');
}

String labelForSpeech(String label) {
  if (campusClassNames.contains(campusName(label))) {
    return campusSpanish(label).toLowerCase();
  }
  final canonical = riskCanonicalLabel(label);
  return switch (canonical) {
    'bache' => 'bache',
    'grieta' => 'grieta',
    'hueco' || 'hoyo' => 'hueco',
    'alcantarilla' => 'alcantarilla abierta',
    'semaforo_atras' => 'semaforo atras',
    'semaforo_verde' => 'semaforo verde',
    'semaforo_rojo' => 'semaforo rojo',
    'semaforo_amarillo' => 'semaforo amarillo',
    'cebolla_de_verdeo' => 'cebolla de verdeo',
    'judia_verde' => 'judia verde',
    'melon_cantalupo' => 'melon cantalupo',
    'fresa_2' => 'fresa',
    'tomate_2' => 'tomate',
    'stairs' => 'escalera',
    'bicycle' => 'bicicleta',
    'pole' => 'poste',
    'traffic_sign' => 'senal de transito',
    'obstacle' => 'obstaculo',
    _ => canonical.replaceAll('_', ' '),
  };
}

String dangerDirectionForSpeech(Detection detection) {
  return detectionDirectionForSpeech(detection);
}

String detectionDirectionForSpeech(Detection detection) {
  return switch (dangerZoneForDetection(detection)) {
    'left' => 'a la izquierda',
    'right' => 'a la derecha',
    _ => 'al frente',
  };
}

String dangerZoneForDetection(Detection detection) {
  final centerX = detection.rect.center.dx / detection.sourceSize.width;
  final direction = campusDirection(centerX);
  return direction == 'center' ? 'front' : direction;
}

String distanceForSpeech(Detection detection) {
  final distance = estimateDangerDistanceMeters(detection);
  return distance == null
      ? distanceBandForSpeech(detection)
      : 'aproximadamente $distance metros';
}

String naturalDistanceForSpeech(Detection detection) {
  final distance = estimateDangerDistanceMeters(detection);
  return distance == null
      ? distanceBandForSpeech(detection)
      : 'a unos $distance metros';
}

String compactDangerSummary(Detection detection) {
  final distance = estimateDangerDistanceMeters(detection);
  final distanceText = distance == null
      ? distanceBandForSpeech(detection)
      : '$distance m aprox.';
  final side = switch (dangerZoneForDetection(detection)) {
    'left' => 'izquierda',
    'right' => 'derecha',
    _ => 'frente',
  };
  return '${labelForSpeech(detection.label)} al $side, $distanceText';
}

int? estimateDangerDistanceMeters(Detection detection) {
  // Legacy overlay objects have no calibrated Depth evidence. Size is not metres.
  return null;
}

String distanceBandForSpeech(Detection detection) {
  final areaRatio =
      (detection.rect.width * detection.rect.height) /
      (detection.sourceSize.width * detection.sourceSize.height);
  if (areaRatio > 0.18) return 'muy cerca';
  if (areaRatio > 0.07) return 'cerca';
  return 'mas adelante';
}

double _dangerClosenessScore(Detection detection) {
  final area = detection.rect.width * detection.rect.height;
  final centerY = detection.rect.center.dy / detection.sourceSize.height;
  return area * (1 + centerY.clamp(0, 1));
}

double _scenePriorityScore(Detection detection) {
  final riskWeight = switch (riskLevelForLabel(detection.label)) {
    RiskLevel.danger => 3,
    RiskLevel.caution => 2,
    RiskLevel.info => 1,
  };
  return riskWeight * 1000000000 + _dangerClosenessScore(detection);
}

class VoiceReminder {
  const VoiceReminder({
    required this.message,
    required this.dueAt,
    required this.isTimer,
    required this.timer,
  });

  final String message;
  final DateTime dueAt;
  final bool isTimer;
  final Timer timer;
}

class NavigationTarget {
  const NavigationTarget({
    required this.name,
    required this.latitude,
    required this.longitude,
    this.address,
  });

  final String name;
  final double latitude;
  final double longitude;
  final String? address;
}

class NavigationSnapshot {
  const NavigationSnapshot({
    required this.targetName,
    required this.targetAddress,
    required this.distanceMeters,
    required this.direction,
    required this.bearing,
    required this.accuracyMeters,
    required this.movingAway,
    required this.updatedAt,
  });

  final String targetName;
  final String? targetAddress;
  final double distanceMeters;
  final String direction;
  final double bearing;
  final double accuracyMeters;
  final bool movingAway;
  final DateTime updatedAt;
}

class ParsedVoiceContact {
  const ParsedVoiceContact({
    required this.name,
    required this.phone,
    required this.isEmergency,
  });

  final String name;
  final String phone;
  final bool isEmergency;
}

class CommandHistoryEntry {
  const CommandHistoryEntry({
    required this.spokenText,
    required this.action,
    required this.at,
    this.type = 'command',
    this.description = '',
    this.riskLevel = '',
    this.detectedObjects = const [],
    this.recommendation = '',
    this.source = '',
  });

  final String spokenText;
  final String action;
  final DateTime at;
  final String type;
  final String description;
  final String riskLevel;
  final List<String> detectedObjects;
  final String recommendation;
  final String source;

  Map<String, Object?> toJson() {
    return {
      'spokenText': spokenText,
      'action': action,
      'at': at.toIso8601String(),
      'type': type,
      'description': description,
      'riskLevel': riskLevel,
      'detectedObjects': detectedObjects,
      'recommendation': recommendation,
      'source': source,
    };
  }

  static CommandHistoryEntry fromJson(Map<String, dynamic> json) {
    final rawObjects = json['detectedObjects'];
    return CommandHistoryEntry(
      spokenText: json['spokenText'] as String? ?? '',
      action: json['action'] as String? ?? 'Comando',
      at: DateTime.tryParse(json['at'] as String? ?? '') ?? DateTime.now(),
      type: json['type'] as String? ?? 'command',
      description: json['description'] as String? ?? '',
      riskLevel: json['riskLevel'] as String? ?? '',
      detectedObjects: rawObjects is List
          ? rawObjects.map((item) => item.toString()).toList()
          : const [],
      recommendation: json['recommendation'] as String? ?? '',
      source: json['source'] as String? ?? '',
    );
  }
}

class VoiceContact {
  VoiceContact({
    required this.name,
    required this.phone,
    required this.aliases,
    required this.isEmergency,
  });

  final String name;
  final String phone;
  final List<String> aliases;
  bool isEmergency;

  bool matches(String command) {
    return aliases.any((alias) {
      return RegExp('\\b${RegExp.escape(alias)}\\b').hasMatch(command);
    });
  }

  Map<String, Object?> toJson() {
    return {
      'name': name,
      'phone': phone,
      'aliases': aliases,
      'isEmergency': isEmergency,
    };
  }

  static VoiceContact fromJson(Map<String, dynamic> json) {
    final name = (json['name'] as String? ?? '').trim();
    final aliases = (json['aliases'] as List? ?? const [])
        .whereType<String>()
        .map((alias) => alias.trim())
        .where((alias) => alias.isNotEmpty)
        .toList();
    return VoiceContact(
      name: name,
      phone: (json['phone'] as String? ?? '').trim(),
      aliases: aliases.isEmpty ? [name.toLowerCase()] : aliases,
      isEmergency: json['isEmergency'] == true,
    );
  }
}

class _CommandPanel extends StatelessWidget {
  const _CommandPanel({
    required this.mode,
    required this.lastWords,
    required this.ocrText,
    required this.lastAnalysis,
    required this.lastVisualDescription,
    required this.lastVisualSource,
    required this.isAnalyzingEnvironment,
    required this.isDescribingScene,
    required this.isReading,
    required this.isSpeaking,
    required this.onHelp,
    required this.onTime,
    required this.onDate,
    required this.onDescribe,
    required this.onVisualModel,
    required this.onMobileVlmStatus,
    required this.onLocation,
    required this.onDanger,
    required this.onPrepareRead,
    required this.onReadNow,
    required this.onStopSpeaking,
    required this.onContacts,
    required this.onHistory,
    required this.onSos,
  });

  final AppMode mode;
  final String lastWords;
  final String ocrText;
  final AnalysisResult? lastAnalysis;
  final String? lastVisualDescription;
  final String lastVisualSource;
  final bool isAnalyzingEnvironment;
  final bool isDescribingScene;
  final bool isReading;
  final bool isSpeaking;
  final VoidCallback onHelp;
  final VoidCallback onTime;
  final VoidCallback onDate;
  final VoidCallback onDescribe;
  final VoidCallback onVisualModel;
  final VoidCallback onMobileVlmStatus;
  final VoidCallback onLocation;
  final VoidCallback onDanger;
  final VoidCallback onPrepareRead;
  final VoidCallback onReadNow;
  final VoidCallback onStopSpeaking;
  final VoidCallback onContacts;
  final VoidCallback onHistory;
  final VoidCallback onSos;

  @override
  Widget build(BuildContext context) {
    if (mode != AppMode.reading) {
      return Container(
        decoration: BoxDecoration(
          color: const Color(0xff101722).withValues(alpha: 0.74),
          borderRadius: BorderRadius.circular(8),
          border: Border.all(color: Colors.white.withValues(alpha: 0.12)),
        ),
        padding: const EdgeInsets.all(10),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Row(
              children: [
                Expanded(
                  child: _ActionButton(
                    icon: Icons.warning_amber_outlined,
                    label: 'Peligro',
                    selected: mode == AppMode.danger,
                    onPressed: onDanger,
                  ),
                ),
                const SizedBox(width: 8),
                Expanded(
                  child: _ActionButton(
                    icon: Icons.chrome_reader_mode_outlined,
                    label: 'Leer',
                    selected: false,
                    onPressed: onPrepareRead,
                  ),
                ),
                const SizedBox(width: 8),
                if (isSpeaking) ...[
                  _IconActionButton(
                    icon: Icons.volume_off_outlined,
                    semanticLabel: 'Silencio',
                    onPressed: onStopSpeaking,
                  ),
                  const SizedBox(width: 8),
                ],
                _IconActionButton(
                  icon: Icons.contacts_outlined,
                  semanticLabel: 'Contactos',
                  onPressed: onContacts,
                ),
                const SizedBox(width: 8),
                _IconActionButton(
                  icon: Icons.history,
                  semanticLabel: 'Historial',
                  onPressed: onHistory,
                ),
                const SizedBox(width: 8),
                _IconActionButton(
                  icon: Icons.sos_outlined,
                  semanticLabel: 'Activar SOS',
                  danger: true,
                  onPressed: onSos,
                ),
              ],
            ),
            const SizedBox(height: 8),
            _QuickVisionRow(
              onDescribe: onDescribe,
              onVisualModel: onVisualModel,
              onMobileVlmStatus: onMobileVlmStatus,
              isAnalyzing: isAnalyzingEnvironment,
              isVisualBusy: isDescribingScene,
            ),
            if (isDescribingScene || lastVisualDescription != null) ...[
              const SizedBox(height: 10),
              _VisualIntelligencePanel(
                description: lastVisualDescription,
                source: lastVisualSource,
                busy: isDescribingScene,
              ),
            ],
            if (lastAnalysis != null) ...[
              const SizedBox(height: 10),
              _EnvironmentAnalysisSummary(result: lastAnalysis!),
            ],
            if (lastWords.isNotEmpty) ...[
              const SizedBox(height: 8),
              Text(
                'Ultimo comando: $lastWords',
                maxLines: 1,
                overflow: TextOverflow.ellipsis,
                style: TextStyle(
                  color: Colors.white.withValues(alpha: 0.68),
                  fontSize: 12,
                ),
              ),
            ],
          ],
        ),
      );
    }

    final displayText = ocrText.isNotEmpty
        ? ocrText
        : 'Modo lectura listo. Enfoca el documento y di "leer ahora" para capturar con OCR.';

    return ConstrainedBox(
      constraints: BoxConstraints(
        maxHeight: MediaQuery.sizeOf(context).height * 0.58,
      ),
      child: Container(
        decoration: BoxDecoration(
          color: const Color(0xff101722).withValues(alpha: 0.92),
          borderRadius: BorderRadius.circular(8),
          border: Border.all(color: Colors.white.withValues(alpha: 0.14)),
        ),
        padding: const EdgeInsets.all(12),
        child: SingleChildScrollView(
          child: Column(
            mainAxisSize: MainAxisSize.min,
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Row(
                children: [
                  Expanded(
                    child: _ActionButton(
                      icon: Icons.warning_amber_outlined,
                      label: 'Peligro',
                      selected: mode == AppMode.danger,
                      onPressed: onDanger,
                    ),
                  ),
                  const SizedBox(width: 8),
                  Expanded(
                    child: _ActionButton(
                      icon: mode == AppMode.reading
                          ? Icons.document_scanner_outlined
                          : Icons.chrome_reader_mode_outlined,
                      label: mode == AppMode.reading ? 'Leer ahora' : 'Leer',
                      selected: mode == AppMode.reading,
                      onPressed: isReading
                          ? null
                          : mode == AppMode.reading
                          ? onReadNow
                          : onPrepareRead,
                    ),
                  ),
                  const SizedBox(width: 8),
                  if (isSpeaking) ...[
                    _IconActionButton(
                      icon: Icons.volume_off_outlined,
                      semanticLabel: 'Silencio',
                      onPressed: onStopSpeaking,
                    ),
                    const SizedBox(width: 8),
                  ],
                  _IconActionButton(
                    icon: Icons.contacts_outlined,
                    semanticLabel: 'Contactos',
                    onPressed: onContacts,
                  ),
                  const SizedBox(width: 8),
                  _IconActionButton(
                    icon: Icons.history,
                    semanticLabel: 'Historial',
                    onPressed: onHistory,
                  ),
                  const SizedBox(width: 8),
                  _IconActionButton(
                    icon: Icons.sos_outlined,
                    semanticLabel: 'Activar SOS',
                    danger: true,
                    onPressed: onSos,
                  ),
                ],
              ),
              const SizedBox(height: 10),
              _QuickAssistGrid(
                onHelp: onHelp,
                onTime: onTime,
                onDate: onDate,
                onDescribe: onDescribe,
                onLocation: onLocation,
              ),
              const SizedBox(height: 12),
              _VoiceFlow(
                mode: mode,
                isReading: isReading,
                isSpeaking: isSpeaking,
              ),
              const SizedBox(height: 12),
              if (lastWords.isNotEmpty)
                Text(
                  'Ultimo comando: $lastWords',
                  maxLines: 1,
                  overflow: TextOverflow.ellipsis,
                  style: TextStyle(
                    color: Colors.white.withValues(alpha: 0.68),
                    fontSize: 12,
                  ),
                ),
              if (lastWords.isNotEmpty) const SizedBox(height: 8),
              _CommandReference(),
              const SizedBox(height: 12),
              Text(
                displayText,
                maxLines: 5,
                overflow: TextOverflow.ellipsis,
                style: const TextStyle(
                  height: 1.35,
                  fontSize: 14,
                  fontWeight: FontWeight.w500,
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

class _QuickVisionRow extends StatelessWidget {
  const _QuickVisionRow({
    required this.onDescribe,
    required this.onVisualModel,
    required this.onMobileVlmStatus,
    required this.isAnalyzing,
    required this.isVisualBusy,
  });

  final VoidCallback onDescribe;
  final VoidCallback onVisualModel;
  final VoidCallback onMobileVlmStatus;
  final bool isAnalyzing;
  final bool isVisualBusy;

  @override
  Widget build(BuildContext context) {
    return Row(
      children: [
        Expanded(
          child: _AssistButton(
            icon: isAnalyzing
                ? Icons.cancel_outlined
                : Icons.visibility_outlined,
            label: isAnalyzing ? 'Cancelar' : 'Que ves',
            semanticLabel: isAnalyzing
                ? 'Cancelar analisis'
                : 'Describir escena',
            onPressed: onDescribe,
          ),
        ),
        const SizedBox(width: 8),
        Expanded(
          child: _AssistButton(
            icon: Icons.auto_awesome_outlined,
            label: 'LFM 2.5',
            semanticLabel: 'Analizar imagen con LFM 2.5',
            onPressed: isVisualBusy ? null : onVisualModel,
            busy: isVisualBusy,
          ),
        ),
        const SizedBox(width: 8),
        Expanded(
          child: _AssistButton(
            icon: Icons.info_outline,
            label: 'Estado',
            semanticLabel: 'Estado de LFM 2.5',
            onPressed: onMobileVlmStatus,
          ),
        ),
      ],
    );
  }
}

class _VisualIntelligencePanel extends StatelessWidget {
  const _VisualIntelligencePanel({
    required this.description,
    required this.source,
    required this.busy,
  });

  final String? description;
  final String source;
  final bool busy;

  @override
  Widget build(BuildContext context) {
    final clean = description?.trim() ?? '';
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(10),
      decoration: BoxDecoration(
        color: const Color(0xff0c1220).withValues(alpha: 0.82),
        borderRadius: BorderRadius.circular(8),
        border: Border.all(
          color: const Color(0xff7dd3fc).withValues(alpha: 0.20),
        ),
      ),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Container(
            width: 34,
            height: 34,
            decoration: BoxDecoration(
              color: const Color(0xff7dd3fc).withValues(alpha: 0.14),
              borderRadius: BorderRadius.circular(8),
            ),
            child: busy
                ? const Padding(
                    padding: EdgeInsets.all(9),
                    child: CircularProgressIndicator(strokeWidth: 2),
                  )
                : const Icon(
                    Icons.auto_awesome_outlined,
                    color: Color(0xff7dd3fc),
                    size: 19,
                  ),
          ),
          const SizedBox(width: 10),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Row(
                  children: [
                    Expanded(
                      child: Text(
                        busy ? 'Analizando con LFM 2.5' : source,
                        maxLines: 1,
                        overflow: TextOverflow.ellipsis,
                        style: const TextStyle(
                          fontSize: 13,
                          fontWeight: FontWeight.w900,
                        ),
                      ),
                    ),
                    _MiniStatusPill(
                      label: busy ? 'en curso' : 'VLM',
                      color: busy
                          ? const Color(0xffffbf47)
                          : const Color(0xff7dd3fc),
                    ),
                  ],
                ),
                const SizedBox(height: 5),
                Text(
                  clean.isEmpty
                      ? 'Capturando una descripcion visual completa.'
                      : clean,
                  maxLines: 3,
                  overflow: TextOverflow.ellipsis,
                  style: TextStyle(
                    color: Colors.white.withValues(alpha: 0.78),
                    fontSize: 12,
                    height: 1.3,
                    fontWeight: FontWeight.w600,
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}

class _MiniStatusPill extends StatelessWidget {
  const _MiniStatusPill({required this.label, required this.color});

  final String label;
  final Color color;

  @override
  Widget build(BuildContext context) {
    return Container(
      height: 24,
      padding: const EdgeInsets.symmetric(horizontal: 8),
      decoration: BoxDecoration(
        color: color.withValues(alpha: 0.14),
        borderRadius: BorderRadius.circular(8),
        border: Border.all(color: color.withValues(alpha: 0.25)),
      ),
      alignment: Alignment.center,
      child: Text(
        label,
        style: TextStyle(
          color: color,
          fontSize: 10,
          fontWeight: FontWeight.w900,
        ),
      ),
    );
  }
}

class _EnvironmentAnalysisSummary extends StatelessWidget {
  const _EnvironmentAnalysisSummary({required this.result});

  final AnalysisResult result;

  @override
  Widget build(BuildContext context) {
    final riskColor = _riskColor(result.riskLevel);
    final rawConfidence = result.confidence.isFinite ? result.confidence : 0.0;
    final confidenceValue = rawConfidence.clamp(0.0, 1.0).toDouble();
    final confidence = (confidenceValue * 100).toStringAsFixed(0);
    final objects = result.detectedObjects.take(5).toList();
    final extraObjects = result.detectedObjects.length - objects.length;

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Divider(height: 1, color: Colors.white.withValues(alpha: 0.12)),
        const SizedBox(height: 10),
        Row(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Container(
              width: 4,
              height: 58,
              decoration: BoxDecoration(
                color: riskColor,
                borderRadius: BorderRadius.circular(8),
              ),
            ),
            const SizedBox(width: 10),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Wrap(
                    spacing: 6,
                    runSpacing: 6,
                    crossAxisAlignment: WrapCrossAlignment.center,
                    children: [
                      _AnalysisBadge(
                        icon: _riskIcon(result.riskLevel),
                        label: 'Riesgo ${result.riskLevel}',
                        color: riskColor,
                      ),
                      _AnalysisBadge(
                        icon: result.isMock
                            ? Icons.science_outlined
                            : Icons.memory_outlined,
                        label: result.sourceText,
                        color: result.isMock
                            ? const Color(0xffc4b5fd)
                            : const Color(0xff60a5fa),
                      ),
                    ],
                  ),
                  const SizedBox(height: 8),
                  Text(
                    result.suggestedAction,
                    maxLines: 1,
                    overflow: TextOverflow.ellipsis,
                    style: TextStyle(
                      color: riskColor,
                      fontSize: 15,
                      fontWeight: FontWeight.w900,
                    ),
                  ),
                ],
              ),
            ),
            const SizedBox(width: 8),
            Column(
              crossAxisAlignment: CrossAxisAlignment.end,
              children: [
                Text(
                  '$confidence%',
                  style: const TextStyle(
                    fontSize: 18,
                    fontWeight: FontWeight.w900,
                  ),
                ),
                Text(
                  'confianza',
                  style: TextStyle(
                    color: Colors.white.withValues(alpha: 0.56),
                    fontSize: 10,
                    fontWeight: FontWeight.w800,
                  ),
                ),
              ],
            ),
          ],
        ),
        const SizedBox(height: 10),
        ClipRRect(
          borderRadius: BorderRadius.circular(8),
          child: LinearProgressIndicator(
            value: confidenceValue,
            minHeight: 4,
            backgroundColor: Colors.white.withValues(alpha: 0.10),
            valueColor: AlwaysStoppedAnimation<Color>(riskColor),
          ),
        ),
        const SizedBox(height: 10),
        _AnalysisLine(
          icon: Icons.visibility_outlined,
          label: result.description,
          color: Colors.white,
          strong: true,
        ),
        const SizedBox(height: 6),
        _AnalysisLine(
          icon: Icons.route_outlined,
          label: result.recommendation,
          color: Colors.white.withValues(alpha: 0.76),
        ),
        if (objects.isNotEmpty) ...[
          const SizedBox(height: 10),
          Wrap(
            spacing: 6,
            runSpacing: 6,
            children: [
              for (final object in objects) _ObjectChip(label: object),
              if (extraObjects > 0) _ObjectChip(label: '+$extraObjects'),
            ],
          ),
        ],
      ],
    );
  }

  Color _riskColor(String riskLevel) {
    return switch (riskLevel) {
      'alto' => const Color(0xffff3b5f),
      'medio' => const Color(0xffffbf47),
      _ => const Color(0xff5eead4),
    };
  }

  IconData _riskIcon(String riskLevel) {
    return switch (riskLevel) {
      'alto' => Icons.report_problem_outlined,
      'medio' => Icons.warning_amber_outlined,
      _ => Icons.verified_outlined,
    };
  }
}

class _AnalysisBadge extends StatelessWidget {
  const _AnalysisBadge({
    required this.icon,
    required this.label,
    required this.color,
  });

  final IconData icon;
  final String label;
  final Color color;

  @override
  Widget build(BuildContext context) {
    return Container(
      height: 28,
      padding: const EdgeInsets.symmetric(horizontal: 9),
      decoration: BoxDecoration(
        color: color.withValues(alpha: 0.14),
        borderRadius: BorderRadius.circular(8),
        border: Border.all(color: color.withValues(alpha: 0.32)),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(icon, color: color, size: 14),
          const SizedBox(width: 5),
          Text(
            label,
            maxLines: 1,
            overflow: TextOverflow.ellipsis,
            style: const TextStyle(fontSize: 11, fontWeight: FontWeight.w900),
          ),
        ],
      ),
    );
  }
}

class _AnalysisLine extends StatelessWidget {
  const _AnalysisLine({
    required this.icon,
    required this.label,
    required this.color,
    this.strong = false,
  });

  final IconData icon;
  final String label;
  final Color color;
  final bool strong;

  @override
  Widget build(BuildContext context) {
    return Row(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Padding(
          padding: const EdgeInsets.only(top: 1),
          child: Icon(icon, size: 15, color: color.withValues(alpha: 0.82)),
        ),
        const SizedBox(width: 8),
        Expanded(
          child: Text(
            label,
            maxLines: 2,
            overflow: TextOverflow.ellipsis,
            style: TextStyle(
              color: color,
              fontSize: strong ? 13 : 12,
              height: 1.25,
              fontWeight: strong ? FontWeight.w800 : FontWeight.w600,
            ),
          ),
        ),
      ],
    );
  }
}

class _ObjectChip extends StatelessWidget {
  const _ObjectChip({required this.label});

  final String label;

  @override
  Widget build(BuildContext context) {
    return Container(
      height: 26,
      padding: const EdgeInsets.symmetric(horizontal: 8),
      decoration: BoxDecoration(
        color: Colors.white.withValues(alpha: 0.075),
        borderRadius: BorderRadius.circular(8),
        border: Border.all(color: Colors.white.withValues(alpha: 0.10)),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(
            Icons.center_focus_strong_outlined,
            color: Colors.white.withValues(alpha: 0.62),
            size: 13,
          ),
          const SizedBox(width: 5),
          Text(
            label,
            maxLines: 1,
            overflow: TextOverflow.ellipsis,
            style: TextStyle(
              color: Colors.white.withValues(alpha: 0.78),
              fontSize: 11,
              fontWeight: FontWeight.w800,
            ),
          ),
        ],
      ),
    );
  }
}

class _QuickAssistGrid extends StatelessWidget {
  const _QuickAssistGrid({
    required this.onHelp,
    required this.onTime,
    required this.onDate,
    required this.onDescribe,
    required this.onLocation,
  });

  final VoidCallback onHelp;
  final VoidCallback onTime;
  final VoidCallback onDate;
  final VoidCallback onDescribe;
  final VoidCallback onLocation;

  @override
  Widget build(BuildContext context) {
    return GridView.count(
      crossAxisCount: 2,
      crossAxisSpacing: 8,
      mainAxisSpacing: 8,
      childAspectRatio: 3.6,
      shrinkWrap: true,
      physics: const NeverScrollableScrollPhysics(),
      children: [
        _AssistButton(
          icon: Icons.help_outline,
          label: 'Ayuda',
          semanticLabel: 'Escuchar ayuda de comandos',
          onPressed: onHelp,
        ),
        _AssistButton(
          icon: Icons.schedule,
          label: 'Hora',
          semanticLabel: 'Decir la hora actual',
          onPressed: onTime,
        ),
        _AssistButton(
          icon: Icons.my_location_outlined,
          label: 'Ubicacion',
          semanticLabel: 'Decir ubicacion actual',
          onPressed: onLocation,
        ),
        _AssistButton(
          icon: Icons.visibility_outlined,
          label: 'Que ves',
          semanticLabel: 'Describir lo que ve la camara',
          onPressed: onDescribe,
        ),
        _AssistButton(
          icon: Icons.today_outlined,
          label: 'Fecha',
          semanticLabel: 'Decir la fecha actual',
          onPressed: onDate,
        ),
      ],
    );
  }
}

class _AssistButton extends StatelessWidget {
  const _AssistButton({
    required this.icon,
    required this.label,
    required this.semanticLabel,
    required this.onPressed,
    this.busy = false,
  });

  final IconData icon;
  final String label;
  final String semanticLabel;
  final VoidCallback? onPressed;
  final bool busy;

  @override
  Widget build(BuildContext context) {
    return Semantics(
      button: true,
      label: semanticLabel,
      child: FilledButton.icon(
        onPressed: onPressed,
        icon: busy
            ? const SizedBox(
                width: 18,
                height: 18,
                child: CircularProgressIndicator(
                  strokeWidth: 2,
                  valueColor: AlwaysStoppedAnimation<Color>(Colors.white),
                ),
              )
            : Icon(icon, size: 18),
        label: FittedBox(fit: BoxFit.scaleDown, child: Text(label)),
        style: FilledButton.styleFrom(
          backgroundColor: Colors.white.withValues(alpha: 0.10),
          foregroundColor: Colors.white,
          disabledBackgroundColor: Colors.white.withValues(alpha: 0.08),
          disabledForegroundColor: Colors.white70,
          shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(8)),
          padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 10),
        ),
      ),
    );
  }
}

class _IconActionButton extends StatelessWidget {
  const _IconActionButton({
    required this.icon,
    required this.semanticLabel,
    required this.onPressed,
    this.danger = false,
  });

  final IconData icon;
  final String semanticLabel;
  final VoidCallback onPressed;
  final bool danger;

  @override
  Widget build(BuildContext context) {
    final background = danger
        ? const Color(0xffef4444)
        : Colors.white.withValues(alpha: 0.10);
    return Semantics(
      button: true,
      label: semanticLabel,
      child: SizedBox(
        width: 46,
        height: 46,
        child: FilledButton(
          onPressed: onPressed,
          style: FilledButton.styleFrom(
            backgroundColor: background,
            foregroundColor: Colors.white,
            shape: RoundedRectangleBorder(
              borderRadius: BorderRadius.circular(8),
            ),
            padding: EdgeInsets.zero,
          ),
          child: Icon(icon, size: 20),
        ),
      ),
    );
  }
}

class _ContactsDialog extends StatefulWidget {
  const _ContactsDialog({
    required this.contacts,
    required this.onAdd,
    required this.onDelete,
    required this.onEmergency,
    required this.onCall,
    required this.onWhatsApp,
    required this.onSos,
  });

  final List<VoiceContact> contacts;
  final Future<void> Function({
    required String name,
    required String phone,
    required bool emergency,
  })
  onAdd;
  final Future<void> Function(VoiceContact contact) onDelete;
  final Future<void> Function(VoiceContact contact) onEmergency;
  final Future<void> Function(VoiceContact contact) onCall;
  final Future<void> Function(VoiceContact contact) onWhatsApp;
  final VoidCallback onSos;

  @override
  State<_ContactsDialog> createState() => _ContactsDialogState();
}

class _ContactsDialogState extends State<_ContactsDialog> {
  final TextEditingController _nameController = TextEditingController();
  final TextEditingController _phoneController = TextEditingController();
  bool _emergency = true;

  @override
  void dispose() {
    _nameController.dispose();
    _phoneController.dispose();
    super.dispose();
  }

  Future<void> _add() async {
    if (_nameController.text.trim().isEmpty ||
        _phoneController.text.trim().isEmpty) {
      return;
    }
    await widget.onAdd(
      name: _nameController.text,
      phone: _phoneController.text,
      emergency: _emergency,
    );
    _nameController.clear();
    _phoneController.clear();
    if (mounted) setState(() => _emergency = false);
  }

  @override
  Widget build(BuildContext context) {
    final emergencyContact = widget.contacts
        .where((contact) => contact.isEmergency)
        .firstOrNull;

    return Dialog(
      backgroundColor: const Color(0xff0c111b),
      insetPadding: const EdgeInsets.symmetric(horizontal: 16, vertical: 22),
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(10)),
      child: ConstrainedBox(
        constraints: const BoxConstraints(maxWidth: 560, maxHeight: 720),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Padding(
              padding: const EdgeInsets.fromLTRB(18, 16, 10, 10),
              child: Row(
                children: [
                  Container(
                    width: 42,
                    height: 42,
                    decoration: BoxDecoration(
                      color: const Color(0xff22d3a6).withValues(alpha: 0.16),
                      borderRadius: BorderRadius.circular(8),
                    ),
                    child: const Icon(
                      Icons.contacts_outlined,
                      color: Color(0xff5eead4),
                    ),
                  ),
                  const SizedBox(width: 12),
                  Expanded(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        const Text(
                          'Contactos',
                          style: TextStyle(
                            fontSize: 18,
                            fontWeight: FontWeight.w900,
                          ),
                        ),
                        Text(
                          '${widget.contacts.length} guardado${widget.contacts.length == 1 ? '' : 's'}'
                          '${emergencyContact == null ? '' : ' · SOS: ${emergencyContact.name}'}',
                          maxLines: 1,
                          overflow: TextOverflow.ellipsis,
                          style: TextStyle(
                            color: Colors.white.withValues(alpha: 0.65),
                            fontSize: 12,
                            fontWeight: FontWeight.w700,
                          ),
                        ),
                      ],
                    ),
                  ),
                  IconButton(
                    tooltip: 'Cerrar',
                    onPressed: () => Navigator.of(context).pop(),
                    icon: const Icon(Icons.close),
                  ),
                ],
              ),
            ),
            Flexible(
              child: SingleChildScrollView(
                padding: const EdgeInsets.fromLTRB(18, 4, 18, 12),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    _ContactSection(
                      title: 'Nuevo contacto',
                      child: Column(
                        children: [
                          TextField(
                            controller: _nameController,
                            decoration: const InputDecoration(
                              prefixIcon: Icon(Icons.badge_outlined),
                              labelText: 'Nombre',
                              hintText: 'Mama',
                              helperText:
                                  'Este nombre se usa en comandos de voz.',
                            ),
                            textCapitalization: TextCapitalization.words,
                          ),
                          const SizedBox(height: 10),
                          TextField(
                            controller: _phoneController,
                            decoration: const InputDecoration(
                              prefixIcon: Icon(Icons.phone_outlined),
                              labelText: 'Numero de telefono',
                              hintText: '+51999999999',
                              helperText:
                                  'Incluye codigo de pais para WhatsApp.',
                            ),
                            keyboardType: TextInputType.phone,
                          ),
                          const SizedBox(height: 6),
                          SwitchListTile(
                            value: _emergency,
                            onChanged: (value) {
                              setState(() => _emergency = value);
                            },
                            contentPadding: EdgeInsets.zero,
                            title: const Text('Usar como contacto SOS'),
                            subtitle: const Text(
                              'Recibe emergencia y ubicacion.',
                            ),
                            secondary: const Icon(Icons.sos_outlined),
                          ),
                          SizedBox(
                            width: double.infinity,
                            child: FilledButton.icon(
                              onPressed: _add,
                              icon: const Icon(Icons.person_add_alt_1),
                              label: const Text('Guardar contacto'),
                              style: FilledButton.styleFrom(
                                padding: const EdgeInsets.symmetric(
                                  vertical: 13,
                                ),
                                shape: RoundedRectangleBorder(
                                  borderRadius: BorderRadius.circular(8),
                                ),
                              ),
                            ),
                          ),
                        ],
                      ),
                    ),
                    const SizedBox(height: 12),
                    _ContactSection(
                      title: 'Guardados',
                      trailing: TextButton.icon(
                        onPressed: widget.onSos,
                        icon: const Icon(Icons.sos_outlined, size: 18),
                        label: const Text('Probar SOS'),
                      ),
                      child: widget.contacts.isEmpty
                          ? const _EmptyContacts()
                          : Column(
                              children: widget.contacts
                                  .map(
                                    (contact) => _ContactTile(
                                      contact: contact,
                                      onDelete: () async {
                                        await widget.onDelete(contact);
                                        if (mounted) setState(() {});
                                      },
                                      onEmergency: () async {
                                        await widget.onEmergency(contact);
                                        if (mounted) setState(() {});
                                      },
                                      onCall: () => widget.onCall(contact),
                                      onWhatsApp: () =>
                                          widget.onWhatsApp(contact),
                                    ),
                                  )
                                  .toList(),
                            ),
                    ),
                  ],
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class _ContactSection extends StatelessWidget {
  const _ContactSection({
    required this.title,
    required this.child,
    this.trailing,
  });

  final String title;
  final Widget child;
  final Widget? trailing;

  @override
  Widget build(BuildContext context) {
    return Container(
      width: double.infinity,
      decoration: BoxDecoration(
        color: Colors.white.withValues(alpha: 0.055),
        borderRadius: BorderRadius.circular(8),
        border: Border.all(color: Colors.white.withValues(alpha: 0.10)),
      ),
      padding: const EdgeInsets.all(12),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Expanded(
                child: Text(
                  title,
                  style: const TextStyle(
                    fontSize: 14,
                    fontWeight: FontWeight.w900,
                  ),
                ),
              ),
              if (trailing != null) trailing!,
            ],
          ),
          const SizedBox(height: 10),
          child,
        ],
      ),
    );
  }
}

class _EmptyContacts extends StatelessWidget {
  const _EmptyContacts();

  @override
  Widget build(BuildContext context) {
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(14),
      decoration: BoxDecoration(
        color: Colors.black.withValues(alpha: 0.14),
        borderRadius: BorderRadius.circular(8),
      ),
      child: Column(
        children: [
          Icon(
            Icons.person_search_outlined,
            color: Colors.white.withValues(alpha: 0.55),
            size: 30,
          ),
          const SizedBox(height: 8),
          const Text(
            'Aun no hay contactos',
            style: TextStyle(fontWeight: FontWeight.w900),
          ),
          const SizedBox(height: 4),
          Text(
            'Guarda uno para decir "llamar a mama" o activar SOS.',
            textAlign: TextAlign.center,
            style: TextStyle(
              color: Colors.white.withValues(alpha: 0.64),
              fontSize: 12,
            ),
          ),
        ],
      ),
    );
  }
}

class _ContactTile extends StatelessWidget {
  const _ContactTile({
    required this.contact,
    required this.onDelete,
    required this.onEmergency,
    required this.onCall,
    required this.onWhatsApp,
  });

  final VoiceContact contact;
  final VoidCallback onDelete;
  final VoidCallback onEmergency;
  final VoidCallback onCall;
  final VoidCallback onWhatsApp;

  @override
  Widget build(BuildContext context) {
    return Container(
      margin: const EdgeInsets.only(bottom: 10),
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: Colors.black.withValues(alpha: 0.14),
        borderRadius: BorderRadius.circular(8),
        border: Border.all(
          color: contact.isEmergency
              ? const Color(0xfff87171).withValues(alpha: 0.55)
              : Colors.white.withValues(alpha: 0.10),
        ),
      ),
      child: Column(
        children: [
          Row(
            children: [
              Container(
                width: 42,
                height: 42,
                decoration: BoxDecoration(
                  color: contact.isEmergency
                      ? const Color(0xffef4444).withValues(alpha: 0.16)
                      : const Color(0xff60a5fa).withValues(alpha: 0.14),
                  borderRadius: BorderRadius.circular(8),
                ),
                child: Icon(
                  contact.isEmergency
                      ? Icons.emergency_share_outlined
                      : Icons.person_outline,
                  color: contact.isEmergency
                      ? const Color(0xfffca5a5)
                      : const Color(0xff93c5fd),
                ),
              ),
              const SizedBox(width: 10),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Row(
                      children: [
                        Expanded(
                          child: Text(
                            contact.name,
                            maxLines: 1,
                            overflow: TextOverflow.ellipsis,
                            style: const TextStyle(
                              fontSize: 15,
                              fontWeight: FontWeight.w900,
                            ),
                          ),
                        ),
                        if (contact.isEmergency)
                          const _ContactBadge(
                            icon: Icons.sos_outlined,
                            label: 'SOS',
                            color: Color(0xfff87171),
                          ),
                      ],
                    ),
                    const SizedBox(height: 4),
                    Row(
                      children: [
                        Icon(
                          Icons.phone_iphone_outlined,
                          size: 14,
                          color: Colors.white.withValues(alpha: 0.58),
                        ),
                        const SizedBox(width: 5),
                        Expanded(
                          child: Text(
                            contact.phone,
                            maxLines: 1,
                            overflow: TextOverflow.ellipsis,
                            style: TextStyle(
                              color: Colors.white.withValues(alpha: 0.72),
                              fontSize: 13,
                              fontWeight: FontWeight.w700,
                            ),
                          ),
                        ),
                      ],
                    ),
                  ],
                ),
              ),
            ],
          ),
          const SizedBox(height: 10),
          Row(
            children: [Expanded(child: _VoiceCommandHint(name: contact.name))],
          ),
          const SizedBox(height: 8),
          Row(
            children: [
              Expanded(
                child: _ContactActionButton(
                  icon: Icons.call_outlined,
                  label: 'Llamar',
                  onPressed: onCall,
                ),
              ),
              const SizedBox(width: 8),
              Expanded(
                child: _ContactActionButton(
                  icon: Icons.chat_outlined,
                  label: 'WhatsApp',
                  onPressed: onWhatsApp,
                ),
              ),
              const SizedBox(width: 8),
              _ContactIconButton(
                tooltip: contact.isEmergency ? 'Contacto SOS' : 'Marcar SOS',
                icon: contact.isEmergency
                    ? Icons.star
                    : Icons.star_border_outlined,
                color: contact.isEmergency
                    ? const Color(0xfffbbf24)
                    : Colors.white70,
                onPressed: onEmergency,
              ),
              const SizedBox(width: 6),
              _ContactIconButton(
                tooltip: 'Eliminar',
                icon: Icons.delete_outline,
                color: const Color(0xfffca5a5),
                onPressed: onDelete,
              ),
            ],
          ),
        ],
      ),
    );
  }
}

class _ContactBadge extends StatelessWidget {
  const _ContactBadge({
    required this.icon,
    required this.label,
    required this.color,
  });

  final IconData icon;
  final String label;
  final Color color;

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 7, vertical: 4),
      decoration: BoxDecoration(
        color: color.withValues(alpha: 0.16),
        borderRadius: BorderRadius.circular(8),
        border: Border.all(color: color.withValues(alpha: 0.32)),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(icon, size: 13, color: color),
          const SizedBox(width: 4),
          Text(
            label,
            style: TextStyle(
              color: color,
              fontSize: 11,
              fontWeight: FontWeight.w900,
            ),
          ),
        ],
      ),
    );
  }
}

class _VoiceCommandHint extends StatelessWidget {
  const _VoiceCommandHint({required this.name});

  final String name;

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 9, vertical: 8),
      decoration: BoxDecoration(
        color: Colors.white.withValues(alpha: 0.055),
        borderRadius: BorderRadius.circular(8),
      ),
      child: Row(
        children: [
          Icon(
            Icons.record_voice_over_outlined,
            size: 15,
            color: Colors.white.withValues(alpha: 0.62),
          ),
          const SizedBox(width: 7),
          Expanded(
            child: Text(
              'Voz: "llamar a $name" · "mensaje a $name"',
              maxLines: 1,
              overflow: TextOverflow.ellipsis,
              style: TextStyle(
                color: Colors.white.withValues(alpha: 0.68),
                fontSize: 12,
                fontWeight: FontWeight.w700,
              ),
            ),
          ),
        ],
      ),
    );
  }
}

class _ContactActionButton extends StatelessWidget {
  const _ContactActionButton({
    required this.icon,
    required this.label,
    required this.onPressed,
  });

  final IconData icon;
  final String label;
  final VoidCallback onPressed;

  @override
  Widget build(BuildContext context) {
    return OutlinedButton.icon(
      onPressed: onPressed,
      icon: Icon(icon, size: 17),
      label: FittedBox(fit: BoxFit.scaleDown, child: Text(label)),
      style: OutlinedButton.styleFrom(
        foregroundColor: Colors.white,
        side: BorderSide(color: Colors.white.withValues(alpha: 0.16)),
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(8)),
        padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 10),
      ),
    );
  }
}

class _ContactIconButton extends StatelessWidget {
  const _ContactIconButton({
    required this.tooltip,
    required this.icon,
    required this.color,
    required this.onPressed,
  });

  final String tooltip;
  final IconData icon;
  final Color color;
  final VoidCallback onPressed;

  @override
  Widget build(BuildContext context) {
    return Tooltip(
      message: tooltip,
      child: SizedBox(
        width: 42,
        height: 42,
        child: IconButton(
          onPressed: onPressed,
          icon: Icon(icon, color: color),
          style: IconButton.styleFrom(
            backgroundColor: Colors.white.withValues(alpha: 0.06),
            shape: RoundedRectangleBorder(
              borderRadius: BorderRadius.circular(8),
            ),
          ),
        ),
      ),
    );
  }
}

class _HistoryDialog extends StatelessWidget {
  const _HistoryDialog({
    required this.entries,
    required this.onExport,
    required this.onClear,
  });

  final List<CommandHistoryEntry> entries;
  final Future<void> Function() onExport;
  final Future<void> Function() onClear;

  @override
  Widget build(BuildContext context) {
    final latest = entries.take(80).toList();
    return Dialog(
      backgroundColor: const Color(0xff0c111b),
      insetPadding: const EdgeInsets.symmetric(horizontal: 16, vertical: 22),
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(10)),
      child: ConstrainedBox(
        constraints: const BoxConstraints(maxWidth: 560, maxHeight: 720),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Padding(
              padding: const EdgeInsets.fromLTRB(18, 16, 10, 10),
              child: Row(
                children: [
                  Container(
                    width: 42,
                    height: 42,
                    decoration: BoxDecoration(
                      color: const Color(0xff60a5fa).withValues(alpha: 0.16),
                      borderRadius: BorderRadius.circular(8),
                    ),
                    child: const Icon(Icons.history, color: Color(0xff93c5fd)),
                  ),
                  const SizedBox(width: 12),
                  Expanded(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        const Text(
                          'Historial de uso',
                          style: TextStyle(
                            fontSize: 18,
                            fontWeight: FontWeight.w900,
                          ),
                        ),
                        Text(
                          '${entries.length} comando${entries.length == 1 ? '' : 's'} guardado${entries.length == 1 ? '' : 's'}',
                          style: TextStyle(
                            color: Colors.white.withValues(alpha: 0.65),
                            fontSize: 12,
                            fontWeight: FontWeight.w700,
                          ),
                        ),
                      ],
                    ),
                  ),
                  IconButton(
                    tooltip: 'Cerrar',
                    onPressed: () => Navigator.of(context).pop(),
                    icon: const Icon(Icons.close),
                  ),
                ],
              ),
            ),
            Padding(
              padding: const EdgeInsets.fromLTRB(18, 0, 18, 10),
              child: Row(
                children: [
                  Expanded(
                    child: FilledButton.icon(
                      onPressed: entries.isEmpty ? null : onExport,
                      icon: const Icon(Icons.download_outlined),
                      label: const Text('Descargar'),
                    ),
                  ),
                  const SizedBox(width: 8),
                  OutlinedButton.icon(
                    onPressed: entries.isEmpty
                        ? null
                        : () async {
                            await onClear();
                            if (context.mounted) Navigator.of(context).pop();
                          },
                    icon: const Icon(Icons.delete_sweep_outlined),
                    label: const Text('Limpiar'),
                  ),
                ],
              ),
            ),
            Flexible(
              child: entries.isEmpty
                  ? const _EmptyHistory()
                  : ListView.separated(
                      padding: const EdgeInsets.fromLTRB(18, 0, 18, 16),
                      itemBuilder: (context, index) {
                        return _HistoryTile(entry: latest[index]);
                      },
                      separatorBuilder: (_, _) => const SizedBox(height: 8),
                      itemCount: latest.length,
                    ),
            ),
          ],
        ),
      ),
    );
  }
}

class _EmptyHistory extends StatelessWidget {
  const _EmptyHistory();

  @override
  Widget build(BuildContext context) {
    return Center(
      child: Padding(
        padding: const EdgeInsets.all(24),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(
              Icons.manage_search_outlined,
              size: 36,
              color: Colors.white.withValues(alpha: 0.52),
            ),
            const SizedBox(height: 10),
            const Text(
              'Sin historial todavia',
              style: TextStyle(fontWeight: FontWeight.w900),
            ),
            const SizedBox(height: 4),
            Text(
              'Los comandos de voz apareceran aqui automaticamente.',
              textAlign: TextAlign.center,
              style: TextStyle(
                color: Colors.white.withValues(alpha: 0.64),
                fontSize: 12,
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class _HistoryTile extends StatelessWidget {
  const _HistoryTile({required this.entry});

  final CommandHistoryEntry entry;

  @override
  Widget build(BuildContext context) {
    final isEnvironment = entry.type == 'environment';
    final subtitle = isEnvironment
        ? [
            entry.description,
            if (entry.riskLevel.isNotEmpty) 'Riesgo: ${entry.riskLevel}',
            if (entry.detectedObjects.isNotEmpty)
              'Objetos: ${entry.detectedObjects.join(', ')}',
            if (entry.recommendation.isNotEmpty)
              'Recomendacion: ${entry.recommendation}',
            if (entry.source.isNotEmpty)
              'Fuente: ${_analysisSourceLabel(entry.source)}',
          ].join('. ')
        : entry.spokenText;
    return Container(
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: Colors.white.withValues(alpha: 0.055),
        borderRadius: BorderRadius.circular(8),
        border: Border.all(color: Colors.white.withValues(alpha: 0.10)),
      ),
      child: Row(
        children: [
          Container(
            width: 38,
            height: 38,
            decoration: BoxDecoration(
              color: const Color(0xff22d3a6).withValues(alpha: 0.14),
              borderRadius: BorderRadius.circular(8),
            ),
            child: Icon(
              isEnvironment
                  ? Icons.visibility_outlined
                  : Icons.record_voice_over_outlined,
              color: const Color(0xff5eead4),
              size: 20,
            ),
          ),
          const SizedBox(width: 10),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Row(
                  children: [
                    Expanded(
                      child: Text(
                        entry.action,
                        maxLines: 1,
                        overflow: TextOverflow.ellipsis,
                        style: const TextStyle(fontWeight: FontWeight.w900),
                      ),
                    ),
                    Text(
                      formatDateTimeForHistory(entry.at),
                      style: TextStyle(
                        color: Colors.white.withValues(alpha: 0.55),
                        fontSize: 11,
                        fontWeight: FontWeight.w700,
                      ),
                    ),
                  ],
                ),
                const SizedBox(height: 4),
                Text(
                  subtitle,
                  maxLines: 2,
                  overflow: TextOverflow.ellipsis,
                  style: TextStyle(
                    color: Colors.white.withValues(alpha: 0.72),
                    fontSize: 12,
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}

String _analysisSourceLabel(String source) {
  return switch (source) {
    'mock' => 'Demo mock',
    'aws_sagemaker' => 'YOLO26m AWS SageMaker',
    'tflite_float32' => 'YOLO26m local (historico)',
    'tflite' => 'TFLite',
    _ => source,
  };
}

class _CommandReference extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    final groups = [
      const _CommandGroup(
        title: 'Lectura',
        commands: ['leer', 'leer ahora', 'repite'],
      ),
      const _CommandGroup(
        title: 'Asistente',
        commands: [
          'que hora es',
          'que fecha es',
          'dime donde estoy',
          'que ves',
          'que hay',
        ],
      ),
      const _CommandGroup(
        title: 'Vision',
        commands: [
          'analiza imagen',
          'estado modelo visual',
          'modelo visual',
          'ver peligro',
        ],
      ),
      const _CommandGroup(
        title: 'Rutas',
        commands: [
          'quiero ir a la farmacia',
          'guiame a Plaza de Armas',
          'como llegar al hospital',
          'abrir Maps',
          'cancelar ruta',
        ],
      ),
      const _CommandGroup(
        title: 'Contactos',
        commands: [
          'quiero guardar contacto',
          'nombre: mama',
          'numero: nueve ocho siete seis cinco cuatro tres dos uno',
          'si es SOS / no es SOS',
          'agrega contacto mama numero nueve ocho siete seis cinco cuatro tres dos uno',
          'mis contactos',
          'contacto SOS',
          'llamar a mama',
          'llamar a mama por WhatsApp',
          'mensaje a mama',
          'SOS',
        ],
      ),
      const _CommandGroup(
        title: 'Recordatorios',
        commands: [
          'recuerdame tomar agua en diez minutos',
          'temporizador en cinco minutos',
          'mis recordatorios',
          'que tengo guardado',
          'cancelar recordatorios',
        ],
      ),
      const _CommandGroup(
        title: 'Control',
        commands: ['historial', 'comandos', 'silencio', 'alto', 'cancelar'],
      ),
    ];

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        const Row(
          children: [
            Icon(Icons.record_voice_over_outlined, size: 18),
            SizedBox(width: 8),
            Text(
              'Comandos de voz',
              style: TextStyle(fontSize: 15, fontWeight: FontWeight.w900),
            ),
          ],
        ),
        const SizedBox(height: 8),
        ...groups.map(
          (group) =>
              Padding(padding: const EdgeInsets.only(bottom: 8), child: group),
        ),
      ],
    );
  }
}

class _CommandGroup extends StatelessWidget {
  const _CommandGroup({required this.title, required this.commands});

  final String title;
  final List<String> commands;

  @override
  Widget build(BuildContext context) {
    return Semantics(
      label: '$title. ${commands.join(". ")}',
      child: DecoratedBox(
        decoration: BoxDecoration(
          color: Colors.white.withValues(alpha: 0.06),
          borderRadius: BorderRadius.circular(8),
          border: Border.all(color: Colors.white.withValues(alpha: 0.08)),
        ),
        child: Padding(
          padding: const EdgeInsets.all(10),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(
                title,
                style: TextStyle(
                  color: Colors.white.withValues(alpha: 0.72),
                  fontSize: 12,
                  fontWeight: FontWeight.w800,
                ),
              ),
              const SizedBox(height: 8),
              Wrap(
                spacing: 6,
                runSpacing: 6,
                children: commands
                    .map((command) => _CommandChip(command: command))
                    .toList(),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

class _CommandChip extends StatelessWidget {
  const _CommandChip({required this.command});

  final String command;

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 9, vertical: 7),
      decoration: BoxDecoration(
        color: const Color(0xff60a5fa).withValues(alpha: 0.15),
        borderRadius: BorderRadius.circular(8),
        border: Border.all(
          color: const Color(0xff60a5fa).withValues(alpha: 0.28),
        ),
      ),
      child: Text(
        command,
        style: const TextStyle(fontSize: 12, fontWeight: FontWeight.w700),
      ),
    );
  }
}

class _ActionButton extends StatelessWidget {
  const _ActionButton({
    required this.icon,
    required this.label,
    required this.selected,
    required this.onPressed,
  });

  final IconData icon;
  final String label;
  final bool selected;
  final VoidCallback? onPressed;

  @override
  Widget build(BuildContext context) {
    return FilledButton.icon(
      onPressed: onPressed,
      icon: Icon(icon, size: 18),
      label: FittedBox(fit: BoxFit.scaleDown, child: Text(label)),
      style: FilledButton.styleFrom(
        backgroundColor: selected
            ? const Color(0xff22d3a6)
            : Colors.white.withValues(alpha: 0.10),
        foregroundColor: selected ? const Color(0xff05231d) : Colors.white,
        disabledBackgroundColor: Colors.white.withValues(alpha: 0.08),
        disabledForegroundColor: Colors.white54,
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(8)),
        padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 12),
      ),
    );
  }
}

class _VoiceFlow extends StatelessWidget {
  const _VoiceFlow({
    required this.mode,
    required this.isReading,
    required this.isSpeaking,
  });

  final AppMode mode;
  final bool isReading;
  final bool isSpeaking;

  @override
  Widget build(BuildContext context) {
    final activeStep = isReading
        ? 2
        : isSpeaking
        ? 3
        : mode == AppMode.reading
        ? 1
        : 0;

    return Row(
      children: [
        Expanded(
          child: _FlowChip(
            icon: Icons.record_voice_over_outlined,
            label: 'Leer',
            active: activeStep == 1,
          ),
        ),
        const SizedBox(width: 8),
        Expanded(
          child: _FlowChip(
            icon: Icons.camera_enhance_outlined,
            label: 'Leer ahora',
            active: activeStep == 2,
          ),
        ),
        const SizedBox(width: 8),
        Expanded(
          child: _FlowChip(
            icon: Icons.volume_up_outlined,
            label: 'Voz',
            active: activeStep == 3,
          ),
        ),
      ],
    );
  }
}

class _FlowChip extends StatelessWidget {
  const _FlowChip({
    required this.icon,
    required this.label,
    required this.active,
  });

  final IconData icon;
  final String label;
  final bool active;

  @override
  Widget build(BuildContext context) {
    return AnimatedContainer(
      duration: const Duration(milliseconds: 180),
      height: 36,
      padding: const EdgeInsets.symmetric(horizontal: 8),
      decoration: BoxDecoration(
        color: active
            ? const Color(0xff60a5fa).withValues(alpha: 0.22)
            : Colors.white.withValues(alpha: 0.07),
        borderRadius: BorderRadius.circular(8),
        border: Border.all(
          color: active
              ? const Color(0xff60a5fa).withValues(alpha: 0.55)
              : Colors.white.withValues(alpha: 0.08),
        ),
      ),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Icon(
            icon,
            size: 15,
            color: active ? const Color(0xff93c5fd) : Colors.white54,
          ),
          const SizedBox(width: 5),
          Flexible(
            child: Text(
              label,
              maxLines: 1,
              overflow: TextOverflow.ellipsis,
              style: TextStyle(
                fontSize: 11,
                fontWeight: FontWeight.w800,
                color: active ? Colors.white : Colors.white60,
              ),
            ),
          ),
        ],
      ),
    );
  }
}

class _MetricPill extends StatelessWidget {
  const _MetricPill({required this.icon, required this.label});

  final IconData icon;
  final String label;

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 8),
      decoration: BoxDecoration(
        color: Colors.white.withValues(alpha: 0.08),
        borderRadius: BorderRadius.circular(8),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(icon, size: 15, color: Colors.white70),
          const SizedBox(width: 6),
          Text(label, style: const TextStyle(fontWeight: FontWeight.w800)),
        ],
      ),
    );
  }
}

class Detection {
  const Detection({
    required this.label,
    required this.confidence,
    required this.sourceSize,
    required this.rect,
  });

  final String label;
  final double confidence;
  final Size sourceSize;
  final Rect rect;
}

class DetectionPainter extends CustomPainter {
  DetectionPainter(this.detections);

  final List<Detection> detections;

  @override
  void paint(Canvas canvas, Size size) {
    final lanePaint = Paint()
      ..style = PaintingStyle.stroke
      ..strokeWidth = 1
      ..color = Colors.white.withValues(alpha: 0.10);
    canvas.drawLine(
      Offset(size.width * 0.38, 0),
      Offset(size.width * 0.38, size.height),
      lanePaint,
    );
    canvas.drawLine(
      Offset(size.width * 0.62, 0),
      Offset(size.width * 0.62, size.height),
      lanePaint,
    );

    final visibleDetections = detections.toList()
      ..sort((a, b) {
        final riskCompare = _riskWeight(
          riskLevelForLabel(b.label),
        ).compareTo(_riskWeight(riskLevelForLabel(a.label)));
        if (riskCompare != 0) return riskCompare;
        return b.confidence.compareTo(a.confidence);
      });

    for (final detection in visibleDetections.take(35)) {
      final risk = riskLevelForLabel(detection.label);
      final color = _riskColor(risk);
      final scale = math.max(
        size.width / detection.sourceSize.width,
        size.height / detection.sourceSize.height,
      );
      final dx = (size.width - detection.sourceSize.width * scale) / 2;
      final dy = (size.height - detection.sourceSize.height * scale) / 2;
      final rect = Rect.fromLTRB(
        detection.rect.left * scale + dx,
        detection.rect.top * scale + dy,
        detection.rect.right * scale + dx,
        detection.rect.bottom * scale + dy,
      );
      final roundedRect = RRect.fromRectAndRadius(
        rect,
        const Radius.circular(8),
      );
      final fillPaint = Paint()
        ..style = PaintingStyle.fill
        ..color = color.withValues(
          alpha: risk == RiskLevel.danger ? 0.12 : 0.07,
        );
      final boxPaint = Paint()
        ..style = PaintingStyle.stroke
        ..strokeWidth = risk == RiskLevel.danger ? 3.4 : 2.2
        ..color = color;

      canvas.drawRRect(roundedRect, fillPaint);
      canvas.drawRRect(roundedRect, boxPaint);
      _drawCorners(canvas, rect, color, risk == RiskLevel.danger ? 5 : 3);

      final riskText = switch (risk) {
        RiskLevel.danger => 'PELIGRO',
        RiskLevel.caution => 'ATENCION',
        RiskLevel.info => '',
      };
      final label = _visualLabelFor(detection, riskText);
      final textPainter = TextPainter(
        text: TextSpan(
          text: label,
          style: TextStyle(
            color: Colors.white,
            fontSize: risk == RiskLevel.danger ? 13 : 12,
            fontWeight: FontWeight.w900,
          ),
        ),
        textDirection: TextDirection.ltr,
        maxLines: 1,
        ellipsis: '...',
      )..layout(maxWidth: math.min(size.width - 24, 280));

      final labelWidth = textPainter.width + 14;
      final labelHeight = textPainter.height + 8;
      final labelLeft = rect.left
          .clamp(8.0, math.max(8.0, size.width - labelWidth - 8))
          .toDouble();
      final preferredTop = rect.top - labelHeight - 7;
      final labelTop = preferredTop >= 8
          ? preferredTop
          : math.min(size.height - labelHeight - 8, rect.bottom + 7);
      final labelRect = Rect.fromLTWH(
        labelLeft,
        labelTop,
        labelWidth,
        labelHeight,
      );
      final labelPaint = Paint()
        ..style = PaintingStyle.fill
        ..color = const Color(0xff05070b).withValues(alpha: 0.82);
      canvas.drawRRect(
        RRect.fromRectAndRadius(labelRect, const Radius.circular(6)),
        labelPaint,
      );
      canvas.drawRRect(
        RRect.fromRectAndRadius(labelRect, const Radius.circular(6)),
        Paint()
          ..style = PaintingStyle.stroke
          ..strokeWidth = 1
          ..color = color.withValues(alpha: 0.65),
      );
      textPainter.paint(canvas, labelRect.topLeft + const Offset(7, 4));
    }
  }

  static int _riskWeight(RiskLevel risk) {
    return switch (risk) {
      RiskLevel.danger => 3,
      RiskLevel.caution => 2,
      RiskLevel.info => 1,
    };
  }

  static Color _riskColor(RiskLevel risk) {
    return switch (risk) {
      RiskLevel.danger => const Color(0xffff3b5f),
      RiskLevel.caution => const Color(0xffffbf47),
      RiskLevel.info => const Color(0xff2dd4bf),
    };
  }

  static String _visualLabelFor(Detection detection, String riskText) {
    final confidence = (detection.confidence * 100).toStringAsFixed(0);
    final label = labelForSpeech(detection.label);
    final base = riskText.isEmpty
        ? '$label | $confidence%'
        : '$riskText | $label | $confidence%';
    if (!isDangerousDetection(detection)) return base;
    final distance = estimateDangerDistanceMeters(detection);
    final distanceText = distance == null
        ? distanceBandForSpeech(detection)
        : '$distance m';
    final side = switch (dangerZoneForDetection(detection)) {
      'left' => 'izq.',
      'right' => 'der.',
      _ => 'frente',
    };
    return '$base | $side | $distanceText';
  }

  static void _drawCorners(
    Canvas canvas,
    Rect rect,
    Color color,
    double strokeWidth,
  ) {
    final length = math.min(24.0, math.min(rect.width, rect.height) * 0.34);
    final paint = Paint()
      ..style = PaintingStyle.stroke
      ..strokeWidth = strokeWidth
      ..strokeCap = StrokeCap.round
      ..color = color.withValues(alpha: 0.92);
    final path = Path()
      ..moveTo(rect.left, rect.top + length)
      ..lineTo(rect.left, rect.top)
      ..lineTo(rect.left + length, rect.top)
      ..moveTo(rect.right - length, rect.top)
      ..lineTo(rect.right, rect.top)
      ..lineTo(rect.right, rect.top + length)
      ..moveTo(rect.right, rect.bottom - length)
      ..lineTo(rect.right, rect.bottom)
      ..lineTo(rect.right - length, rect.bottom)
      ..moveTo(rect.left + length, rect.bottom)
      ..lineTo(rect.left, rect.bottom)
      ..lineTo(rect.left, rect.bottom - length);
    canvas.drawPath(path, paint);
  }

  @override
  bool shouldRepaint(covariant DetectionPainter oldDelegate) {
    return oldDelegate.detections != detections;
  }
}
