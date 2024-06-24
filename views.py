from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
import os
import cv2
import numpy as np

def run_object_detection(image_path):
    # Load YOLO
    net = cv2.dnn.readNet("detection\model\yolov3.weights", "detection\model\yolov3.cfg")
    layer_names = net.getLayerNames()
    
    try:
        output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    except IndexError:
        output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    
    with open("detection\model\coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]

    # Load image
    img = cv2.imread(image_path)
    if img is None:
        print("Error: Image not loaded correctly")
        return None

    height, width, channels = img.shape
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = (0, 255, 0)
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, label, (x, y - 10), font, 1, color, 2)

    detected_image_path = image_path.replace('.jpg', '_detected.jpg')
    cv2.imwrite(detected_image_path, img)
    return detected_image_path

def upload_image(request):
    if request.method == 'POST' and request.FILES['image']:
        image = request.FILES['image']
        fs = FileSystemStorage()
        filename = fs.save(image.name, image)
        uploaded_file_url = fs.url(filename)
        uploaded_image_path = fs.path(filename)

        if os.path.exists(uploaded_image_path):
            print(f"Uploaded image exists at: {uploaded_image_path}")

        detected_image_path = run_object_detection(uploaded_image_path)
        if detected_image_path:
            detected_image_url = fs.url(detected_image_path)
            if os.path.exists(detected_image_path):
                print(f"Detected image exists at: {detected_image_path}")
        else:
            detected_image_url = None

        context = {
            'uploaded_file_url': uploaded_file_url,
            'detected_image_url': detected_image_url
        }
        return render(request, 'detection/result.html', context)
    return render(request, 'detection/upload.html')
