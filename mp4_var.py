import cv2
import numpy as np

net = cv2.dnn.readNet("yolov3.cfg", "yolov3.weights")

classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Генерация случайных цветов для каждого класса
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Загрузка видео
video_path = "video.mp4"
cap = cv2.VideoCapture(video_path)

# Получение информации о размере видео
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Создание объекта для записи видео с обнаруженными объектами
output_path = "output.mp4"
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

while True:
    ret, frame = cap.read()
    if not ret:
        break

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
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                box_width = int(detection[2] * width)
                box_height = int(detection[3] * height)

                # Вычисление координат верхнего левого угла ограничивающей рамки
                x = int(center_x - box_width / 2)
                y = int(center_y - box_height / 2)

                # Добавление информации об объекте в соответствующие списки
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, box_width, box_height])

    # Применение подавления немаксимумов для удаления дубликатов
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Отображение результатов на кадре
    font = cv2.FONT_HERSHEY_SIMPLEX
    for i in range(len(boxes)):
        if i in indices:
            x, y, box_width, box_height = boxes[i]
            label = classes[class_ids[i]]
            confidence = confidences[i]
            color = colors[class_ids[i]]

            cv2.rectangle(frame, (x, y), (x + box_width, y + box_height), color, 2)
            cv2.putText(frame, f"{label}: {confidence:.2f}", (x, y - 10), font, 0.5, color, 2)

    # Запись кадра с обнаруженными объектами в видеофайл
    out.write(frame)

    # Отображение кадра с обнаруженными объектами
    cv2.imshow("YOLO Object Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Освобождение ресурсов
cap.release()
out.release()
cv2.destroyAllWindows()
