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
import java.nio.FloatBuffer
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import kotlin.math.max
import kotlin.math.min

class ObjectDetectorHelper(
    private val context: Context,
    private val modelName: String = "yolov8n_person_fp16.tflite",
    private val inputSize: Int = 320,
    private val confThreshold: Float = 0.5f,
    private val iouThreshold: Float = 0.45f
) {
    private var interpreter: Interpreter? = null
    private var gpuDelegate: GpuDelegate? = null
    private lateinit var inputBuffer: ByteBuffer
    private lateinit var outputBuffer: FloatBuffer
    private var numBoxes: Int = 0

    init {
        setupInterpreter()
    }

    fun detect(originalBitmap: Bitmap): List<BoundingBox> {
        val interp = interpreter ?: return emptyList()

        return try {
            val resized = ImageUtils.letterbox(originalBitmap, inputSize).bitmap

            inputBuffer.rewind()
            val pixels = IntArray(inputSize * inputSize)
            resized.getPixels(pixels, 0, inputSize, 0, 0, inputSize, inputSize)

            for (channel in 0 until 3) {
                for (pixel in pixels) {
                    val value = when (channel) {
                        0 -> ((pixel shr 16) and 0xFF) / 255f
                        1 -> ((pixel shr 8) and 0xFF) / 255f
                        else -> (pixel and 0xFF) / 255f
                    }
                    inputBuffer.putFloat(value)
                }
            }

            outputBuffer.rewind()
            interp.run(inputBuffer, outputBuffer)
            resized.recycle()

            decodeOutput(originalBitmap.width, originalBitmap.height)
        } catch (e: Exception) {
            Log.e(TAG, "Detect error: ${e.message}", e)
            emptyList()
        }
    }

    fun close() {
        interpreter?.close()
        interpreter = null
        gpuDelegate?.close()
        gpuDelegate = null
    }

    private fun setupInterpreter() {
        try {
            val options = Interpreter.Options()
            val compatList = CompatibilityList()

            if (compatList.isDelegateSupportedOnThisDevice) {
                gpuDelegate = GpuDelegate(compatList.bestOptionsForThisDevice)
                options.addDelegate(gpuDelegate)
            } else {
                options.setNumThreads(4)
            }

            interpreter = Interpreter(loadModelFile(), options)
            val outputShape = interpreter?.getOutputTensor(0)?.shape() ?: intArrayOf(1, 84, 2100)
            numBoxes = outputShape[2]

            inputBuffer = ByteBuffer.allocateDirect(1 * 3 * inputSize * inputSize * 4).apply {
                order(ByteOrder.nativeOrder())
            }

            outputBuffer = ByteBuffer.allocateDirect(outputShape[1] * outputShape[2] * 4).apply {
                order(ByteOrder.nativeOrder())
            }.asFloatBuffer()

            Log.d(TAG, "Model initialized: $modelName. Output shape: ${outputShape.contentToString()}")
        } catch (e: Exception) {
            Log.e(TAG, "Setup error: ${e.message}", e)
        }
    }

    private fun loadModelFile(): MappedByteBuffer {
        val fileDescriptor = context.assets.openFd(modelName)
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        return fileChannel.map(
            FileChannel.MapMode.READ_ONLY,
            fileDescriptor.startOffset,
            fileDescriptor.declaredLength
        )
    }

    private fun decodeOutput(origWidth: Int, origHeight: Int): List<BoundingBox> {
        val result = mutableListOf<BoundingBox>()

        outputBuffer.rewind()
        val raw = FloatArray(outputBuffer.capacity())
        outputBuffer.get(raw)

        val scale = min(inputSize.toFloat() / origWidth, inputSize.toFloat() / origHeight)
        val padX = (inputSize - origWidth * scale) / 2f
        val padY = (inputSize - origHeight * scale) / 2f

        for (index in 0 until numBoxes) {
            val personScore = raw[(PERSON_CLASS_ID + 4) * numBoxes + index]
            if (personScore <= confThreshold) continue

            var centerX = raw[index]
            var centerY = raw[numBoxes + index]
            var width = raw[2 * numBoxes + index]
            var height = raw[3 * numBoxes + index]

            if (centerX <= 1f && centerY <= 1f && width <= 1f && height <= 1f) {
                centerX *= inputSize
                centerY *= inputSize
                width *= inputSize
                height *= inputSize
            }

            val x1 = (centerX - width / 2f - padX) / scale
            val y1 = (centerY - height / 2f - padY) / scale
            val x2 = (centerX + width / 2f - padX) / scale
            val y2 = (centerY + height / 2f - padY) / scale

            if (x2 > x1 && y2 > y1) {
                result.add(
                    BoundingBox(
                        x1 = x1.coerceIn(0f, origWidth.toFloat()),
                        y1 = y1.coerceIn(0f, origHeight.toFloat()),
                        x2 = x2.coerceIn(0f, origWidth.toFloat()),
                        y2 = y2.coerceIn(0f, origHeight.toFloat()),
                        score = personScore,
                        label = "person"
                    )
                )
            }
        }

        return nonMaxSuppression(result)
    }

    private fun nonMaxSuppression(boxes: List<BoundingBox>): List<BoundingBox> {
        val sorted = boxes.sortedByDescending { it.score }.toMutableList()
        val selected = mutableListOf<BoundingBox>()

        while (sorted.isNotEmpty()) {
            val best = sorted.removeAt(0)
            selected.add(best)
            sorted.removeAll { iou(best, it) > iouThreshold }
        }

        return selected
    }

    private fun iou(first: BoundingBox, second: BoundingBox): Float {
        val intersectionX1 = max(first.x1, second.x1)
        val intersectionY1 = max(first.y1, second.y1)
        val intersectionX2 = min(first.x2, second.x2)
        val intersectionY2 = min(first.y2, second.y2)

        val intersectionWidth = max(0f, intersectionX2 - intersectionX1)
        val intersectionHeight = max(0f, intersectionY2 - intersectionY1)
        val intersectionArea = intersectionWidth * intersectionHeight

        val firstArea = max(0f, first.x2 - first.x1) * max(0f, first.y2 - first.y1)
        val secondArea = max(0f, second.x2 - second.x1) * max(0f, second.y2 - second.y1)
        val union = firstArea + secondArea - intersectionArea

        return if (union <= 0f) 0f else intersectionArea / union
    }

    private companion object {
        private const val TAG = "ObjectDetectorHelper"
        private const val PERSON_CLASS_ID = 0
    }
}
