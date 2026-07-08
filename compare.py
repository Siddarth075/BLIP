from ultralytics import YOLO
import cv2
import os
import uuid

model = YOLO("models/yolov8n.pt")


def detect_objects(image_path):
    image = cv2.imread(image_path)

    results = model(image)

    detections = []

    for result in results:

        boxes = result.boxes

        for box in boxes:

            cls = int(box.cls[0])

            name = model.names[cls]

            conf = float(box.conf[0])

            if conf < 0.40:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            detections.append({
                "class": name,
                "confidence": conf,
                "box": (x1, y1, x2, y2)
            })

    return image, detections


def compare_images(image1_path, image2_path):

    img1, det1 = detect_objects(image1_path)
    img2, det2 = detect_objects(image2_path)

    classes1 = [d["class"] for d in det1]
    classes2 = [d["class"] for d in det2]

    common = []
    removed = []
    added = []

    for d in det1:

        if d["class"] in classes2:
            common.append(d)
        else:
            removed.append(d)

    for d in det2:

        if d["class"] not in classes1:
            added.append(d)

    image1 = img1.copy()
    image2 = img2.copy()

    green = (0, 255, 0)
    red = (0, 0, 255)
    blue = (255, 0, 0)

    for obj in common:

        x1, y1, x2, y2 = obj["box"]

        cv2.rectangle(image1, (x1, y1), (x2, y2), green, 2)

        cv2.putText(
            image1,
            obj["class"],
            (x1, y1 - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            green,
            2
        )

    for obj in removed:

        x1, y1, x2, y2 = obj["box"]

        cv2.rectangle(image1, (x1, y1), (x2, y2), red, 2)

        cv2.putText(
            image1,
            "Missing : " + obj["class"],
            (x1, y1 - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            red,
            2
        )

    for obj in added:

        x1, y1, x2, y2 = obj["box"]

        cv2.rectangle(image2, (x1, y1), (x2, y2), blue, 2)

        cv2.putText(
            image2,
            "New : " + obj["class"],
            (x1, y1 - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            blue,
            2
        )

    output_folder = "static/results"

    os.makedirs(output_folder, exist_ok=True)

    id1 = str(uuid.uuid4()) + ".jpg"
    id2 = str(uuid.uuid4()) + ".jpg"

    out1 = os.path.join(output_folder, id1)
    out2 = os.path.join(output_folder, id2)

    cv2.imwrite(out1, image1)
    cv2.imwrite(out2, image2)

    report = {
        "common": sorted(list(set([x["class"] for x in common]))),
        "removed": sorted(list(set([x["class"] for x in removed]))),
        "added": sorted(list(set([x["class"] for x in added]))),
        "total_image1": len(det1),
        "total_image2": len(det2)
    }

    return {
        "image1": out1.replace("\\", "/"),
        "image2": out2.replace("\\", "/"),
        "report": report
    }