# ABCOnlyOne

Lab 12 - deteccion de personas en tiempo real con CameraX, YOLOv8 y LiteRT.

## Modelo requerido

Coloca el modelo exportado en:

```text
app/src/main/assets/yolov8n_person_fp16.tflite
```

El codigo filtra la clase `person` de COCO (`PERSON_CLASS_ID = 0`), aplica `confThreshold = 0.5f` y usa NMS para evitar contar duplicados.

## Verificacion

```bash
./gradlew assembleDebug
```

La app no declara permiso de internet; la inferencia ocurre 100% en el dispositivo.

## Exportar el modelo en Google Colab

Abre `colab/export_yolo_litert.ipynb` en Google Colab, ejecuta todas las celdas y descarga:

```text
yolov8n_person_fp16.tflite
```

Luego copia ese archivo a `app/src/main/assets/`.
