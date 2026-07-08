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
import sqlite3
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

DB = "history.db"

conn = sqlite3.connect(DB)

cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS history(
id INTEGER PRIMARY KEY AUTOINCREMENT,
image TEXT,
caption TEXT,
question TEXT,
answer TEXT,
confidence REAL,
date TEXT
)
""")

conn.commit()
conn.close()




def save_history(image, caption, question, answer, confidence):

    conn = sqlite3.connect(DB)

    cursor = conn.cursor()

    cursor.execute(
        """
        INSERT INTO history
        (image,caption,question,answer,confidence,date)

        VALUES(?,?,?,?,?,?)
        """,
        (
            image,
            caption,
            question,
            answer,
            confidence,
            datetime.now().strftime("%d-%m-%Y %H:%M")
        )
    )

    conn.commit()

    conn.close()


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

    conn = sqlite3.connect(DB)

    cursor = conn.cursor()

    cursor.execute(
        "SELECT * FROM history ORDER BY id DESC"
    )

    data = cursor.fetchall()

    conn.close()

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

    caption=caption_processor.decode(
        output[0],
        skip_special_tokens=True
    )

    session["image_path"]=image_path
    session["caption"]=caption
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

    if len(chat) > CHAT_LIMIT:
        chat.pop(0)

    session["chat"] = chat
    session["confidence"] = confidence

    return redirect("/")


@app.route("/download")
def download():

    conn = sqlite3.connect(DB)

    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT image,
               caption,
               question,
               answer
        FROM history
        ORDER BY id DESC
        LIMIT 1
        """
    )

    row = cursor.fetchone()

    conn.close()

    if row is None:
        return "No Report"

    pdf = create_pdf(
        row[0],
        row[1],
        row[2],
        row[3]
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