package com.illareklab.abconlyone.detector

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.ImageFormat
import android.graphics.Matrix
import android.graphics.Rect
import android.graphics.YuvImage
import androidx.camera.core.ImageProxy
import java.io.ByteArrayOutputStream

object ImageUtils {
    data class LetterboxResult(
        val bitmap: Bitmap,
        val scale: Float,
        val dx: Float,
        val dy: Float
    )

    fun imageProxyToBitmap(imageProxy: ImageProxy): Bitmap {
        val image = imageProxy.image ?: error("ImageProxy image is null")
        val nv21 = yuv420ToNv21(imageProxy)
        val yuvImage = YuvImage(nv21, ImageFormat.NV21, image.width, image.height, null)
        val stream = ByteArrayOutputStream()
        yuvImage.compressToJpeg(Rect(0, 0, image.width, image.height), 95, stream)
        val bitmap = BitmapFactory.decodeByteArray(stream.toByteArray(), 0, stream.size())
        return bitmap.rotate(imageProxy.imageInfo.rotationDegrees.toFloat())
    }

    fun letterbox(source: Bitmap, size: Int): LetterboxResult {
        val scale = minOf(size / source.width.toFloat(), size / source.height.toFloat())
        val scaledWidth = (source.width * scale).toInt()
        val scaledHeight = (source.height * scale).toInt()
        val resized = Bitmap.createScaledBitmap(source, scaledWidth, scaledHeight, true)
        val output = Bitmap.createBitmap(size, size, Bitmap.Config.ARGB_8888)
        val canvas = android.graphics.Canvas(output)
        val dx = (size - scaledWidth) / 2f
        val dy = (size - scaledHeight) / 2f
        canvas.drawColor(android.graphics.Color.BLACK)
        canvas.drawBitmap(resized, dx, dy, null)
        if (resized != source) resized.recycle()
        return LetterboxResult(output, scale, dx, dy)
    }

    private fun Bitmap.rotate(degrees: Float): Bitmap {
        if (degrees == 0f) return this
        val matrix = Matrix().apply { postRotate(degrees) }
        val rotated = Bitmap.createBitmap(this, 0, 0, width, height, matrix, true)
        recycle()
        return rotated
    }

    private fun yuv420ToNv21(imageProxy: ImageProxy): ByteArray {
        val yBuffer = imageProxy.planes[0].buffer
        val uBuffer = imageProxy.planes[1].buffer
        val vBuffer = imageProxy.planes[2].buffer
        val ySize = yBuffer.remaining()
        val uSize = uBuffer.remaining()
        val vSize = vBuffer.remaining()
        val nv21 = ByteArray(ySize + uSize + vSize)
        yBuffer.get(nv21, 0, ySize)
        vBuffer.get(nv21, ySize, vSize)
        uBuffer.get(nv21, ySize + vSize, uSize)
        return nv21
    }
}
