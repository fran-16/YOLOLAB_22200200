package com.RezzaValencia.yololab_22200200.ui

import android.content.Context
import android.graphics.Paint
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import androidx.camera.view.PreviewView
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.drawscope.rotate
import androidx.compose.ui.graphics.nativeCanvas
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalLifecycleOwner
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import com.RezzaValencia.yololab_22200200.camera.CameraController
import com.RezzaValencia.yololab_22200200.detector.BoundingBox
import com.RezzaValencia.yololab_22200200.detector.ImageUtils
import com.RezzaValencia.yololab_22200200.detector.ObjectDetectorHelper
import kotlin.math.atan2
import kotlin.math.max
import kotlin.math.roundToInt

@Composable
fun CameraScreen() {
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current
    val detectorHelper = remember { ObjectDetectorHelper(context.applicationContext) }
    var detections by remember { mutableStateOf<List<BoundingBox>>(emptyList()) }
    var frameWidth by remember { mutableStateOf(0) }
    var frameHeight by remember { mutableStateOf(0) }
    var tiltAngle by remember { mutableStateOf(0f) }
    var cameraController by remember { mutableStateOf<CameraController?>(null) }

    DisposableEffect(Unit) {
        val sensorManager = context.getSystemService(Context.SENSOR_SERVICE) as SensorManager
        val accelerometer = sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)
        val listener = object : SensorEventListener {
            override fun onSensorChanged(event: SensorEvent) {
                tiltAngle = Math.toDegrees(atan2(-event.values[0].toDouble(), event.values[1].toDouble())).toFloat()
            }

            override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) = Unit
        }
        accelerometer?.let { sensorManager.registerListener(listener, it, SensorManager.SENSOR_DELAY_UI) }
        onDispose {
            sensorManager.unregisterListener(listener)
            detectorHelper.close()
        }
    }

    Box(modifier = Modifier.fillMaxSize()) {
        AndroidView(
            modifier = Modifier.fillMaxSize(),
            factory = { ctx ->
                PreviewView(ctx).also { previewView ->
                    previewView.scaleType = PreviewView.ScaleType.FILL_CENTER
                    val controller = CameraController(ctx, lifecycleOwner, previewView) { imageProxy ->
                        try {
                            val bitmap = ImageUtils.imageProxyToBitmap(imageProxy)
                            detections = detectorHelper.detect(bitmap)
                            frameWidth = bitmap.width
                            frameHeight = bitmap.height
                            bitmap.recycle()
                        } catch (_: Exception) {
                            detections = emptyList()
                        } finally {
                            imageProxy.close()
                        }
                    }
                    controller.start()
                    cameraController = controller
                }
            },
            onRelease = {
                cameraController?.shutdown()
                cameraController = null
            }
        )

        DetectionOverlay(
            detections = detections,
            frameWidth = frameWidth,
            frameHeight = frameHeight,
            angleDegrees = tiltAngle,
            modifier = Modifier.fillMaxSize()
        )

        Text(
            text = "Personas detectadas: ${detections.size}",
            color = Color.White,
            style = MaterialTheme.typography.bodyMedium,
            modifier = Modifier
                .align(Alignment.TopStart)
                .padding(16.dp)
                .background(Color.Black.copy(alpha = 0.55f))
                .padding(horizontal = 12.dp, vertical = 8.dp)
        )

        CompassOverlay(
            angleDegrees = tiltAngle,
            modifier = Modifier
                .align(Alignment.BottomEnd)
                .padding(16.dp)
        )
    }
}

@Composable
private fun DetectionOverlay(
    detections: List<BoundingBox>,
    frameWidth: Int,
    frameHeight: Int,
    angleDegrees: Float,
    modifier: Modifier = Modifier
) {
    Canvas(modifier = modifier) {
        if (frameWidth == 0 || frameHeight == 0) return@Canvas
        val scale = max(size.width / frameWidth, size.height / frameHeight)
        val offsetX = (frameWidth * scale - size.width) / 2f
        val offsetY = (frameHeight * scale - size.height) / 2f
        val textPaint = Paint().apply {
            color = android.graphics.Color.WHITE
            textSize = 36f
            isAntiAlias = true
            isFakeBoldText = true
        }

        rotate(angleDegrees, pivot = center) {
            drawLine(
                color = Color(0xFF39FF88),
                start = Offset(-size.width * 0.1f, size.height * 0.18f),
                end = Offset(size.width * 1.1f, size.height * 0.18f),
                strokeWidth = 8f
            )
        }

        detections.forEach { box ->
            val left = box.x1 * scale - offsetX
            val top = box.y1 * scale - offsetY
            val right = box.x2 * scale - offsetX
            val bottom = box.y2 * scale - offsetY
            drawRect(
                color = Color(0xFF39FF88),
                topLeft = Offset(left, top),
                size = Size(right - left, bottom - top),
                style = Stroke(width = 8f)
            )
            drawContext.canvas.nativeCanvas.drawText(
                "${box.label} ${(box.score * 100).toInt()}%",
                left + 8f,
                (top - 12f).coerceAtLeast(36f),
                textPaint
            )
        }
    }
}

@Composable
private fun CompassOverlay(angleDegrees: Float, modifier: Modifier = Modifier) {
    Box(modifier = modifier, contentAlignment = Alignment.Center) {
        Canvas(modifier = Modifier.size(72.dp)) {
            drawCircle(color = Color.Black.copy(alpha = 0.45f), radius = size.minDimension / 2f)
            drawCircle(color = Color.White, radius = size.minDimension / 2f, style = Stroke(width = 3f))
            rotate(angleDegrees) {
                val centerX = size.width / 2f
                val centerY = size.height / 2f
                val length = size.minDimension / 2f - 8f
                drawLine(Color.Red, Offset(centerX, centerY), Offset(centerX, centerY - length), strokeWidth = 5f)
                drawLine(Color.White, Offset(centerX, centerY), Offset(centerX, centerY + length), strokeWidth = 5f)
            }
        }
        Text(
            text = "${angleDegrees.roundToInt()}°",
            color = Color.White,
            style = MaterialTheme.typography.labelSmall
        )
    }
}
