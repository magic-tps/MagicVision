package com.example.mi_app

import io.flutter.embedding.android.FlutterActivity
import io.flutter.embedding.engine.FlutterEngine
import android.content.Context
import android.media.AudioDeviceInfo
import android.media.AudioManager
import io.flutter.plugin.common.MethodChannel

class MainActivity : FlutterActivity() {
    private var usbCamera: UsbCameraBridge? = null
    private var audioChannel: MethodChannel? = null

    override fun configureFlutterEngine(flutterEngine: FlutterEngine) {
        super.configureFlutterEngine(flutterEngine)
        usbCamera = UsbCameraBridge(this, flutterEngine.dartExecutor.binaryMessenger, flutterEngine.renderer)
        audioChannel = MethodChannel(flutterEngine.dartExecutor.binaryMessenger, "magicvision/audio_input").also { channel ->
            channel.setMethodCallHandler { call, result ->
                if (call.method != "inputDevices") {
                    result.notImplemented()
                } else {
                    val manager = getSystemService(Context.AUDIO_SERVICE) as AudioManager
                    val inputs = manager.getDevices(AudioManager.GET_DEVICES_INPUTS)
                    val usb = inputs.filter { it.type == AudioDeviceInfo.TYPE_USB_DEVICE || it.type == AudioDeviceInfo.TYPE_USB_HEADSET }
                    result.success(mapOf("usb_available" to usb.isNotEmpty(), "explicit_selection" to false,
                        "devices" to inputs.map { mapOf("id" to it.id, "type" to it.type, "name" to it.productName.toString()) }))
                }
            }
        }
    }

    override fun onStop() {
        usbCamera?.suspendCamera()
        super.onStop()
    }

    override fun onStart() {
        super.onStart()
        usbCamera?.resumeCamera()
    }

    override fun cleanUpFlutterEngine(flutterEngine: FlutterEngine) {
        usbCamera?.destroy()
        usbCamera = null
        audioChannel?.setMethodCallHandler(null)
        audioChannel = null
        super.cleanUpFlutterEngine(flutterEngine)
    }
}
