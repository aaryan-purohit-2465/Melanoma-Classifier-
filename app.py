import os

from flask import Flask, render_template, request, send_from_directory
from werkzeug.utils import secure_filename

from data import ensure_dummy_dataset
from predict import predict_image
from train import train_all


app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024


def startup():
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
    ensure_dummy_dataset()
    train_all()


startup()


@app.route("/", methods=["GET", "POST"])
def index():
    a = None
    b = None
    c = None
    d = None
    image_name = None

    if request.method == "POST":
        e = request.form.get("model_type", "binary")
        f = request.files.get("image")
        if f is None or f.filename == "":
            b = "Please upload an image."
        else:
            g = secure_filename(f.filename)
            c = os.path.join(app.config["UPLOAD_FOLDER"], g)
            image_name = g
            f.save(c)
            try:
                h = predict_image(c, e)
                a = h["predicted_class"]
                d = round(h["confidence"] * 100, 2)
            except Exception as x:
                b = str(x)

    return render_template(
        "index.html",
        predicted=a,
        confidence=d,
        error=b,
        image_path=c,
        image_name=image_name,
    )


@app.route("/uploads/<path:a>")
def uploaded_file(a):
    return send_from_directory(app.config["UPLOAD_FOLDER"], a)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
