from flask import Flask, request, jsonify
import whisper
import tempfile
import os

app = Flask(__name__)
model = whisper.load_model("base")

@app.route("/")
def index():
    return "Whisper subtitle extractor is running!"

@app.route("/transcribe", methods=["POST"])
def transcribe():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp:
        file.save(temp.name)
        result = model.transcribe(temp.name, fp16=False)
        os.remove(temp.name)

    # ساخت فایل SRT
    segments = result["segments"]
    srt_output = ""
    for i, seg in enumerate(segments, start=1):
        start = format_timestamp(seg["start"])
        end = format_timestamp(seg["end"])
        text = seg["text"].strip()
        srt_output += f"{i}\n{start} --> {end}\n{text}\n\n"

    return jsonify({"srt": srt_output})


def format_timestamp(seconds):
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    ms = int((seconds - int(seconds)) * 1000)
    return f"{hrs:02}:{mins:02}:{secs:02},{ms:03}"


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
