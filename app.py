from collections import deque
from flask import Flask, render_template, request, redirect, url_for, send_file, session
from PIL import Image
from transformers import (
    BlipProcessor,
    BlipForConditionalGeneration,
    BlipForQuestionAnswering
)

from ultralytics import YOLO
from collections import Counter

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


from compare import compare_images
import os
import uuid
import re
import cv2
import easyocr
import torch
import os
from pymongo import MongoClient
from datetime import datetime
from reportlab.pdfgen import canvas

app = Flask(__name__)

app.secret_key = "visioniq_secret_key"

CHAT_LIMIT = 20


embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

UPLOAD_FOLDER = "static/uploads"
REPORT_FOLDER = "static/reports"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(REPORT_FOLDER, exist_ok=True)

RESULT_FOLDER = "static/results"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading Caption Model...")

caption_processor = BlipProcessor.from_pretrained(
    "Salesforce/blip-image-captioning-base"
)

caption_model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base"
).to(device)

print("Loading VQA Model...")

vqa_processor = BlipProcessor.from_pretrained(
    "Salesforce/blip-vqa-base"
)

vqa_model = BlipForQuestionAnswering.from_pretrained(
    "Salesforce/blip-vqa-base"
).to(device)

reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())
yolo_model = YOLO("yolov8n.pt")

print("Models Loaded Successfully")


# ---------------- MongoDB ----------------

client = MongoClient("mongodb://localhost:27017/")

db = client["visioniq_ai"]

history_collection = db["history"]

print("MongoDB Connected Successfully")



def save_history(image, caption, question, answer, confidence):

    history_collection.insert_one({

        "image": image,

        "caption": caption,

        "question": question,

        "answer": answer,

        "confidence": confidence,

        "date": datetime.now()

    })


def create_pdf(image, caption, question, answer):

    pdf_name = os.path.join(
        REPORT_FOLDER,
        "report.pdf"
    )

    c = canvas.Canvas(pdf_name)

    c.setFont("Helvetica-Bold",18)

    c.drawString(
        180,
        800,
        "VisionIQ Report"
    )

    c.setFont("Helvetica",12)

    c.drawString(50,740,"Image : "+image)

    c.drawString(50,700,"Caption :")

    c.drawString(80,680,caption)

    c.drawString(50,640,"Question :")

    c.drawString(80,620,question)

    c.drawString(50,580,"Answer :")

    c.drawString(80,560,answer)

    c.drawString(
        50,
        520,
        datetime.now().strftime("%d-%m-%Y %H:%M")
    )

    c.save()

    return pdf_name

def answer_from_ocr(question, ocr_text):

    lines = [
        line.strip()
        for line in ocr_text.split("\n")
        if line.strip()
    ]

    if not lines:
        return None

    question_embedding = embedding_model.encode([question])

    line_embeddings = embedding_model.encode(lines)

    similarity = cosine_similarity(
        question_embedding,
        line_embeddings
    )[0]

    best_index = similarity.argmax()

    if similarity[best_index] > 0.30:
        return lines[best_index]

    return None


@app.route("/")
def home():

    return render_template(

    "index.html",

    image_path=session.get("image_path"),

    caption=session.get("caption"),

    ocr_text=session.get("ocr_text"),
    confidence=session.get("confidence",0),
    objects=session.get("objects"),

    chat=session.get("chat",[])

)


@app.route("/history")
def history():

    records = list(
        history_collection.find().sort("date", -1)
    )

    data = []

    for record in records:

        data.append((
            str(record.get("_id")),
            record.get("image", ""),
            record.get("caption", ""),
            record.get("question", ""),
            record.get("answer", ""),
            record.get("confidence", 0),
            record.get("date").strftime("%d-%m-%Y %H:%M")
            if record.get("date") else ""
        ))

    return render_template(
        "history.html",
        data=data
    )


@app.route("/upload", methods=["POST"])
def upload():

    file=request.files.get("image")

    if file is None or file.filename=="":
        return redirect("/")

    filename=file.filename

    image_path=os.path.join(
        app.config["UPLOAD_FOLDER"],
        filename
    )

    file.save(image_path)

    image=Image.open(image_path).convert("RGB")

    results = yolo_model(image_path)

    objects = []

    for result in results:

        for box in result.boxes:

            class_id = int(box.cls[0])

            class_name = yolo_model.names[class_id]

            objects.append(class_name)

    object_counts = dict(Counter(objects))
    # OCR

    img = cv2.imread(image_path)

    ocr_result = reader.readtext(img)

    detected_text = ""

    for item in ocr_result:
        detected_text += item[1] + "\n"

    inputs=caption_processor(
        images=image,
        return_tensors="pt"
    ).to(device)

    output=caption_model.generate(**inputs)

    caption = caption_processor.decode(
        output[0],
        skip_special_tokens=True
    ).strip()

    print("Generated Caption:", repr(caption))
    print("Caption:", caption)
    print("OCR:", detected_text)
    print("Objects:", object_counts)

    session["image_path"]=image_path
    session["caption"]=caption
    print("Session Caption:", session["caption"])
    session["ocr_text"] = detected_text
    session["objects"] = object_counts
    session["chat"]=[]
    session["answer"]=""
    session["confidence"]=0

    return redirect("/")


@app.route("/ask", methods=["POST"])
def ask():

    image_path = session.get("image_path")

    if image_path is None:
        return redirect("/")

    question = request.form.get("question")
    question_lower = question.lower()

    ocr_text = session.get("ocr_text", "")

    image = Image.open(image_path).convert("RGB")

    answer = None

    # ---------------- OCR ----------------

    answer = None

    if ocr_text:

        answer = answer_from_ocr(question, ocr_text)

    # ---------------- BLIP VQA ----------------

    if answer is None:

        inputs = vqa_processor(
            image,
            question,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            outputs = vqa_model.generate(**inputs)

        answer = vqa_processor.decode(
            outputs[0],
            skip_special_tokens=True
        )

    confidence = 100

    save_history(
        os.path.basename(image_path),
        session.get("caption"),
        question,
        answer,
        confidence
    )

    chat = session.get("chat", [])

    chat.append({
        "question": question,
        "answer": answer
    })

    print("================================")
    print("Question:", question)
    print("Answer:", repr(answer))
    print("Chat:", chat)
    print("================================")

    print("Answer:", answer)
    print("Chat:", chat)

    if len(chat) > CHAT_LIMIT:
        chat.pop(0)

    session["chat"] = chat
    session["confidence"] = confidence

    return redirect("/")



@app.route("/crop_region", methods=["POST"])
def crop_region():

    image_path = session.get("image_path")

    if image_path is None:
        return redirect("/")

    print(request.form)
    x = int(request.form["x"])
    y = int(request.form["y"])
    w = int(request.form["w"])
    h = int(request.form["h"])

    image = cv2.imread(image_path)

    cropped = image[y:y+h, x:x+w]

    filename = str(uuid.uuid4()) + ".jpg"

    save_path = os.path.join(
        "static",
        "results",
        filename
    )

    cv2.imwrite(save_path, cropped)

    return render_template(
    "region_vqa.html",
    image="results/" + filename,
    answer=None
)

@app.route("/region_ask", methods=["POST"])
def region_ask():

    image = request.form["image"]
    question = request.form["question"]

    image_path = os.path.join("static", image)

    raw_image = Image.open(image_path).convert("RGB")

    inputs = vqa_processor(
        raw_image,
        question,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        output = vqa_model.generate(**inputs)

    answer = vqa_processor.decode(
        output[0],
        skip_special_tokens=True
    )

    return render_template(
        "region_vqa.html",
        image=image,
        answer=answer
    )


@app.route("/annotate")
def annotate():

    image_path = session.get("image_path")

    if image_path is None:
        return redirect("/")

    return render_template(
        "annotate.html",
        image_path="/" + image_path.replace("\\","/")
    )

@app.route("/download")
def download():

    row = history_collection.find_one(
        sort=[("date", -1)]
    )

    if row is None:
        return "No Report"

    pdf = create_pdf(
        row.get("image", ""),
        row.get("caption", ""),
        row.get("question", ""),
        row.get("answer", "")
    )

    return send_file(
        pdf,
        as_attachment=True
    )

@app.route("/compare", methods=["GET", "POST"])
def compare():

    if request.method == "POST":

        if "image1" not in request.files or "image2" not in request.files:
            return "Please upload two images."

        img1 = request.files["image1"]
        img2 = request.files["image2"]

        if img1.filename == "" or img2.filename == "":
            return "Please choose both images."

        filename1 = str(uuid.uuid4()) + "_" + img1.filename
        filename2 = str(uuid.uuid4()) + "_" + img2.filename

        path1 = os.path.join(UPLOAD_FOLDER, filename1)
        path2 = os.path.join(UPLOAD_FOLDER, filename2)

        img1.save(path1)
        img2.save(path2)

        result = compare_images(path1, path2)

        return render_template(
            "compare.html",
            image1=result["image1"],
            image2=result["image2"],
            report=result["report"]
        )

    return render_template(
        "compare.html",
        image1=None,
        image2=None,
        report=None
    )



if __name__ == "__main__":

    app.run(
        debug=True
    )