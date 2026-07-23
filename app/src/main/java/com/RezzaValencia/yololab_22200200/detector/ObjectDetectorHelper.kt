package com.RezzaValencia.yololab_22200200.detector

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.CompatibilityList
import org.tensorflow.lite.gpu.GpuDelegate
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel
import kotlin.math.max
import kotlin.math.min

class ObjectDetectorHelper(
    private val context: Context,
    private val modelName: String = "yolov8n_person_fp16.tflite",
    private val inputSize: Int = 320,
    private val confThreshold: Float = 0.62f,
    private val iouThreshold: Float = 0.35f
) {
    private var gpuDelegate: GpuDelegate? = null
    private val interpreter: Interpreter by lazy { setupInterpreter() }

    fun detect(bitmap: Bitmap): List<BoundingBox> {
        val letterbox = ImageUtils.letterbox(bitmap, inputSize)
        val inputBuffer = bitmapToNchwBuffer(letterbox.bitmap)
        val outputShape = interpreter.getOutputTensor(0).shape()
        val channels = outputShape[1]
        val numBoxes = outputShape[2]
        val output = Array(1) { Array(channels) { FloatArray(numBoxes) } }
        interpreter.run(inputBuffer, output)
        letterbox.bitmap.recycle()
        return nonMaxSuppression(decodeOutput(output[0], numBoxes, letterbox, bitmap.width, bitmap.height))
    }

    fun close() {
        interpreter.close()
        gpuDelegate?.close()
    }

    private fun setupInterpreter(): Interpreter {
        val options = Interpreter.Options()
        val compatList = CompatibilityList()
        if (compatList.isDelegateSupportedOnThisDevice) {
            gpuDelegate = GpuDelegate(compatList.bestOptionsForThisDevice)
            options.addDelegate(gpuDelegate)
        } else {
            options.setNumThreads(4)
        }
        return Interpreter(loadModel(), options).also {
            Log.d(TAG, "Input shape: ${it.getInputTensor(0).shape().contentToString()}")
            Log.d(TAG, "Output shape: ${it.getOutputTensor(0).shape().contentToString()}")
        }
    }

    private fun loadModel(): ByteBuffer {
        val fileDescriptor = context.assets.openFd(modelName)
        FileInputStream(fileDescriptor.fileDescriptor).use { input ->
            return input.channel.map(
                FileChannel.MapMode.READ_ONLY,
                fileDescriptor.startOffset,
                fileDescriptor.declaredLength
            )
        }
    }

    private fun bitmapToNchwBuffer(bitmap: Bitmap): ByteBuffer {
        val buffer = ByteBuffer.allocateDirect(1 * 3 * inputSize * inputSize * 4)
        buffer.order(ByteOrder.nativeOrder())
        val pixels = IntArray(inputSize * inputSize)
        bitmap.getPixels(pixels, 0, inputSize, 0, 0, inputSize, inputSize)
        for (channel in 0 until 3) {
            for (pixel in pixels) {
                val value = when (channel) {
                    0 -> (pixel shr 16) and 0xFF
                    1 -> (pixel shr 8) and 0xFF
                    else -> pixel and 0xFF
                }
                buffer.putFloat(value / 255f)
            }
        }
        buffer.rewind()
        return buffer
    }

    private fun decodeOutput(
        raw: Array<FloatArray>,
        numBoxes: Int,
        letterbox: ImageUtils.LetterboxResult,
        originalWidth: Int,
        originalHeight: Int
    ): List<BoundingBox> {
        val boxes = mutableListOf<BoundingBox>()
        val minArea = originalWidth * originalHeight * MIN_BOX_AREA_RATIO
        for (i in 0 until numBoxes) {
            val personScore = raw[PERSON_CLASS_ID + 4][i]
            if (personScore < confThreshold) continue

            val cx = raw[0][i]
            val cy = raw[1][i]
            val w = raw[2][i]
            val h = raw[3][i]
            val left = ((cx - w / 2f) - letterbox.dx) / letterbox.scale
            val top = ((cy - h / 2f) - letterbox.dy) / letterbox.scale
            val right = ((cx + w / 2f) - letterbox.dx) / letterbox.scale
            val bottom = ((cy + h / 2f) - letterbox.dy) / letterbox.scale
            val boxWidth = right - left
            val boxHeight = bottom - top
            val area = boxWidth * boxHeight
            val aspectRatio = boxWidth / (boxHeight + 0.00001f)

            if (area < minArea) continue
            if (aspectRatio !in MIN_PERSON_ASPECT_RATIO..MAX_PERSON_ASPECT_RATIO) continue

            boxes += BoundingBox(
                x1 = left.coerceIn(0f, originalWidth.toFloat()),
                y1 = top.coerceIn(0f, originalHeight.toFloat()),
                x2 = right.coerceIn(0f, originalWidth.toFloat()),
                y2 = bottom.coerceIn(0f, originalHeight.toFloat()),
                score = personScore
            )
        }
        return boxes
    }

    private fun nonMaxSuppression(boxes: List<BoundingBox>): List<BoundingBox> {
        val selected = mutableListOf<BoundingBox>()
        val candidates = boxes.sortedByDescending { it.score }.toMutableList()
        while (candidates.isNotEmpty() && selected.size < MAX_DETECTIONS) {
            val current = candidates.removeAt(0)
            selected += current
            candidates.removeAll { isDuplicate(current, it) }
        }
        return selected
    }

    private fun isDuplicate(a: BoundingBox, b: BoundingBox): Boolean {
        if (iou(a, b) > iouThreshold) return true
        val centerAx = (a.x1 + a.x2) / 2f
        val centerAy = (a.y1 + a.y2) / 2f
        val centerBx = (b.x1 + b.x2) / 2f
        val centerBy = (b.y1 + b.y2) / 2f
        val distanceX = kotlin.math.abs(centerAx - centerBx)
        val distanceY = kotlin.math.abs(centerAy - centerBy)
        val averageWidth = ((a.x2 - a.x1) + (b.x2 - b.x1)) / 2f
        val averageHeight = ((a.y2 - a.y1) + (b.y2 - b.y1)) / 2f
        return distanceX < averageWidth * 0.35f && distanceY < averageHeight * 0.35f
    }

    private fun iou(a: BoundingBox, b: BoundingBox): Float {
        val x1 = max(a.x1, b.x1)
        val y1 = max(a.y1, b.y1)
        val x2 = min(a.x2, b.x2)
        val y2 = min(a.y2, b.y2)
        val intersection = max(0f, x2 - x1) * max(0f, y2 - y1)
        val areaA = max(0f, a.x2 - a.x1) * max(0f, a.y2 - a.y1)
        val areaB = max(0f, b.x2 - b.x1) * max(0f, b.y2 - b.y1)
        return intersection / (areaA + areaB - intersection + 0.00001f)
    }

    private companion object {
        private const val TAG = "ObjectDetectorHelper"
        private const val PERSON_CLASS_ID = 0
        private const val MIN_BOX_AREA_RATIO = 0.015f
        private const val MIN_PERSON_ASPECT_RATIO = 0.12f
        private const val MAX_PERSON_ASPECT_RATIO = 1.8f
        private const val MAX_DETECTIONS = 12
    }
}
