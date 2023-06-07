#base

import cv2
import numpy as np

# Загрузка конфигурационного файла и предобученной модели YOLO
net = cv2.dnn.readNet("cfg", "yolo")

# Загрузка классов объектов
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Генерация случайных цветов для каждого класса
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Захват видео из источника (камеры или видеозаписи)
cap = cv2.VideoCapture(0)  # Используйте 0 для захвата с камеры, либо укажите путь к видеозаписи

while True:
    ret, frame = cap.read()

    # Создание блоба из кадра для входа в нейронную сеть
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)

    # Установка входного блоба в сеть
    net.setInput(blob)

    # Получение результатов обнаружения объектов
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    outs = net.forward(output_layers)

    # Инициализация списков для обнаруженных объектов
    class_ids = []
    confidences = []
    boxes = []

    # Обработка результатов обнаружения
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5:  # Пороговое значение для фильтрации низких уверенностей
                # Вычисление координат ограничивающей рамки объекта
                center_x = int(detection[0] * frame.shape[1])
                center_y = int(detection[1] * frame.shape[0])
                width = int(detection[2] * frame.shape[1])
                height = int(detection[3] * frame.shape[0])

                # Вычисление координат верхнего левого угла ограничивающей рамки
                x = int(center_x - width / 2)
                y = int(center_y - height / 2)

                # Добавление информации об объекте в соответствующие списки
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, width, height])

    # Применение подавления немаксимумов для удаления дубликатов
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Отображение результатов на кадре
    font = cv2.FONT_HERSHEY_SIMPLEX
    for i in range(len(boxes)):
        if i in indices:
            x, y, width, height = boxes[i]
            label = classes[class_ids[i]]
            confidence = confidences[i]
            color = colors[class_ids[i]]

            cv2.rectangle(frame, (x, y), (x + width, y + height), color, 2)
            cv2.putText(frame, f"{label}: {confidence:.2f}", (x, y - 10), font, 0.5, color, 2)

    # Отображение кадра с обнаруженными объектами
    cv2.imshow("YOLO Object Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Освобождение ресурсов
cap.release()
cv2.destroyAllWindows()

