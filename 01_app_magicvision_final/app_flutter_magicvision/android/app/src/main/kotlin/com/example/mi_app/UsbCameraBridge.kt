package com.example.mi_app

import android.app.Activity
import android.content.Context
import android.graphics.ImageFormat
import android.graphics.Rect
import android.graphics.YuvImage
import android.hardware.usb.UsbConstants
import android.hardware.usb.UsbDevice
import android.hardware.usb.UsbManager
import android.os.Handler
import android.os.Looper
import android.view.Surface
import com.herohan.uvcapp.CameraException
import com.herohan.uvcapp.CameraHelper
import com.herohan.uvcapp.ICameraHelper
import com.serenegiant.usb.UVCCamera
import com.serenegiant.utils.UVCUtils
import io.flutter.plugin.common.BinaryMessenger
import io.flutter.plugin.common.EventChannel
import io.flutter.plugin.common.MethodChannel
import io.flutter.view.TextureRegistry
import java.io.ByteArrayOutputStream
import java.util.concurrent.Executors
import java.util.concurrent.RejectedExecutionException
import java.util.concurrent.atomic.AtomicBoolean

/** USB Host permission/attach handling is provided by CameraHelper's USBMonitor.
 * CameraX is never used for UVC. A frame is copied only for an explicit capture.
 */
class UsbCameraBridge(private val activity: Activity, messenger: BinaryMessenger,
                      private val textures: TextureRegistry) : EventChannel.StreamHandler {
    private val methods = MethodChannel(messenger, "magicvision/usb_camera")
    private val events = EventChannel(messenger, "magicvision/usb_camera_events")
    private val main = Handler(Looper.getMainLooper())
    private val encoder = Executors.newSingleThreadExecutor()
    private val wantsFrame = AtomicBoolean(false)
    private var helper: CameraHelper? = null
    private var texture: TextureRegistry.SurfaceTextureEntry? = null
    private var surface: Surface? = null
    private var sink: EventChannel.EventSink? = null
    private var device: UsbDevice? = null
    private var pending: MethodChannel.Result? = null
    private var state = "cameraDisconnected"
    private var message: String? = null
    private var width = 640
    private var height = 480
    @Volatile private var generation = 0
    @Volatile private var captureId = 0
    private var initialized = false
    private var suspended = false
    private val captureTimeout = Runnable { failCapture("capture_timeout", "La camara no entrego una imagen") }
    private val openTimeout = Runnable {
        if (state == "cameraConnecting") {
            helper?.closeCamera()
            emit("error", "No se pudo abrir la camara USB. Usa Reconectar camara")
        }
    }

    init {
        UVCUtils.init(activity.applicationContext)
        events.setStreamHandler(this)
        methods.setMethodCallHandler { call, result ->
            try {
                when (call.method) {
                    "initialize" -> { initialized = true; start(); result.success(snapshot()) }
                    "captureFrame" -> capture(result)
                    "dispose" -> { initialized = false; release(); result.success(null) }
                    else -> result.notImplemented()
                }
            } catch (_: Exception) {
                result.error("usb_error", "No se pudo usar la camara USB", null)
            }
        }
    }

    private fun isUvc(d: UsbDevice): Boolean = d.deviceClass == UsbConstants.USB_CLASS_VIDEO ||
        (0 until d.interfaceCount).any { d.getInterface(it).interfaceClass == UsbConstants.USB_CLASS_VIDEO }

    private fun start() {
        if (suspended) return
        release()
        val current = ++generation
        val camera = CameraHelper()
        helper = camera
        texture = textures.createSurfaceTexture()
        surface = Surface(texture!!.surfaceTexture())
        camera.setStateCallback(object : ICameraHelper.StateCallback {
            override fun onAttach(d: UsbDevice) { if (current == generation && device == null && isUvc(d)) select(d) }
            override fun onDeviceOpen(d: UsbDevice, first: Boolean) {
                if (current == generation && device?.deviceId == d.deviceId && !suspended) camera.openCamera()
            }
            override fun onCameraOpen(d: UsbDevice) {
                if (current != generation || suspended || device?.deviceId != d.deviceId) return
                val size = camera.previewSize
                width = size?.width ?: 640
                height = size?.height ?: 480
                if (width > 1280 || height > 720 || width <= 0 || height <= 0) {
                    camera.closeCamera()
                    emit("error", "Formato UVC no compatible con captura reducida")
                    return
                }
                texture?.surfaceTexture()?.setDefaultBufferSize(width, height)
                surface?.let { camera.addSurface(it, false) }
                val frameWidth = width
                val frameHeight = height
                camera.setFrameCallback({ buffer ->
                    if (current == generation && wantsFrame.compareAndSet(true, false)) {
                        val ticket = captureId
                        val required = frameWidth * frameHeight * 3 / 2
                        val frame = buffer.duplicate().apply { rewind() }
                        if (frame.remaining() < required) {
                            main.post { if (ticket == captureId) failCapture("invalid_frame", "Frame UVC invalido") }
                        } else {
                            val bytes = ByteArray(required)
                            frame.get(bytes)
                            try { encoder.execute {
                                try {
                                    val output = ByteArrayOutputStream()
                                    check(YuvImage(bytes, ImageFormat.NV21, frameWidth, frameHeight, null)
                                        .compressToJpeg(Rect(0, 0, frameWidth, frameHeight), 95, output))
                                    val jpeg = output.toByteArray()
                                    main.post {
                                        if (current == generation && ticket == captureId && state == "cameraReady") {
                                            val reply = pending
                                            pending = null
                                            main.removeCallbacks(captureTimeout)
                                            reply?.success(jpeg)
                                        }
                                    }
                                } catch (_: Exception) {
                                    main.post { if (current == generation && ticket == captureId) failCapture("invalid_frame", "No se pudo codificar el frame") }
                                }
                            } } catch (_: RejectedExecutionException) {
                                // Engine disposal may race the native frame callback.
                                main.post { if (current == generation && ticket == captureId) failCapture("camera_disconnected", "Camara desconectada") }
                            }
                        }
                    }
                }, UVCCamera.PIXEL_FORMAT_NV21)
                camera.startPreview()
                main.removeCallbacks(openTimeout)
                emit("cameraReady")
            }
            override fun onCameraClose(d: UsbDevice) { disconnected(d, current) }
            override fun onDeviceClose(d: UsbDevice) { disconnected(d, current) }
            override fun onDetach(d: UsbDevice) {
                if (current != generation) return
                val wasSelected = device?.deviceId == d.deviceId
                disconnected(d, current)
                if (wasSelected) main.post { if (initialized && !suspended && current == generation) start() }
            }
            override fun onCancel(d: UsbDevice) {
                if (current != generation || device?.deviceId != d.deviceId) return
                device = null
                main.removeCallbacks(openTimeout)
                emit("error", "Permiso USB rechazado. Usa Reconectar camara para reintentar")
            }
            override fun onError(d: UsbDevice, error: CameraException) {
                if (current != generation) return
                main.removeCallbacks(openTimeout)
                failCapture("usb_error", "No se pudo abrir la camara USB")
                device = null
                emit("error", "No se pudo abrir la camara USB")
            }
        })
        val manager = activity.getSystemService(Context.USB_SERVICE) as UsbManager
        manager.deviceList.values.firstOrNull { isUvc(it) }?.let { select(it) }
        if (device == null) emit("cameraDisconnected")
    }

    private fun select(d: UsbDevice) {
        if (device != null || suspended) return
        device = d
        emit("cameraConnecting")
        // USBMonitor requests UsbManager permission and reports denial in onCancel.
        helper?.selectDevice(d)
        main.postDelayed(openTimeout, 30000)
    }

    private fun disconnected(d: UsbDevice, current: Int) {
        if (current != generation || device?.deviceId != d.deviceId) return
        main.removeCallbacks(openTimeout)
        failCapture("camera_disconnected", "Camara desconectada")
        device = null
        emit("cameraDisconnected")
    }

    private fun capture(result: MethodChannel.Result) {
        if (state != "cameraReady" || suspended) {
            result.error("camera_disconnected", "Camara desconectada", null)
            return
        }
        if (pending != null) { result.error("camera_busy", "Captura en curso", null); return }
        pending = result
        captureId++
        wantsFrame.set(true)
        main.postDelayed(captureTimeout, 4000)
    }

    private fun failCapture(code: String, text: String) {
        captureId++
        main.removeCallbacks(captureTimeout)
        wantsFrame.set(false)
        val reply = pending
        pending = null
        reply?.error(code, text, null)
    }

    private fun snapshot(): Map<String, Any?> = mapOf("state" to state, "message" to message,
        "texture_id" to texture?.id(), "width" to width, "height" to height)
    private fun emit(value: String, text: String? = null) {
        state = value
        message = text
        sink?.success(snapshot())
    }

    private fun release() {
        generation++
        main.removeCallbacks(openTimeout)
        failCapture("camera_disconnected", "Camara desconectada")
        helper?.setStateCallback(null)
        helper?.releaseAll()
        helper = null
        surface?.release()
        surface = null
        texture?.release()
        texture = null
        device = null
        emit("cameraDisconnected")
    }

    fun suspendCamera() { suspended = true; if (initialized) release() }
    fun resumeCamera() { val wasSuspended = suspended; suspended = false; if (initialized && wasSuspended) start() }
    fun destroy() { initialized = false; release(); encoder.shutdown(); methods.setMethodCallHandler(null); events.setStreamHandler(null); sink = null }
    override fun onListen(arguments: Any?, events: EventChannel.EventSink?) { sink = events; sink?.success(snapshot()) }
    override fun onCancel(arguments: Any?) { sink = null }
}
