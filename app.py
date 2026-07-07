from collections import deque
from flask import Flask, render_template, request, redirect, url_for, send_file, session
from PIL import Image
from transformers import (
    BlipProcessor,
    BlipForConditionalGeneration,
    BlipForQuestionAnswering
)

import torch
import os
import sqlite3
from datetime import datetime
from reportlab.pdfgen import canvas

app = Flask(__name__)

app.secret_key = "visioniq_secret_key"

CHAT_LIMIT = 20

UPLOAD_FOLDER = "static/uploads"
REPORT_FOLDER = "static/reports"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(REPORT_FOLDER, exist_ok=True)

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


@app.route("/")
def home():

    return render_template(

    "index.html",

    image_path=session.get("image_path"),

    caption=session.get("caption"),

    confidence=session.get("confidence",0),

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

    image = Image.open(image_path).convert("RGB")

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


if __name__ == "__main__":

    app.run(
        debug=True
    )