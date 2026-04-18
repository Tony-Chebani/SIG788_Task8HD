from flask import Flask, render_template, request
import os

from openai import AzureOpenAI
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from msrest.authentication import CognitiveServicesCredentials

app = Flask(__name__, template_folder="templates")

# -----------------------------
# CONFIG
# -----------------------------
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# -----------------------------
# ENV VARIABLES (SAFE)
# -----------------------------
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")

AZURE_VISION_KEY = os.getenv("AZURE_VISION_KEY")
AZURE_VISION_ENDPOINT = os.getenv("AZURE_VISION_ENDPOINT")

# -----------------------------
# CLIENTS (SAFE INIT)
# -----------------------------
client = None
vision_client = None

try:
    client = AzureOpenAI(
        api_key=AZURE_OPENAI_API_KEY,
        api_version="2024-02-15-preview",
        azure_endpoint=AZURE_OPENAI_ENDPOINT
    )
except Exception as e:
    print("OpenAI init error:", e)

try:
    vision_client = ComputerVisionClient(
        AZURE_VISION_ENDPOINT,
        CognitiveServicesCredentials(AZURE_VISION_KEY)
    )
except Exception as e:
    print("Vision init error:", e)

# -----------------------------
# FUNCTIONS
# -----------------------------
def process_text(user_input):
    if not client:
        return "AI service not configured."

    try:
        response = client.chat.completions.create(
            model="gpt-5-chat",
            messages=[
                {"role": "system", "content": "You are a healthcare assistant."},
                {"role": "user", "content": user_input}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        print("Text error:", e)
        return "Error processing text."

def process_image(image_path):
    if not vision_client:
        return "Vision service not configured."

    try:
        with open(image_path, "rb") as img:
            desc = vision_client.describe_image_in_stream(img)

        captions = [c.text for c in desc.captions] if desc.captions else []

        with open(image_path, "rb") as img:
            tags_result = vision_client.tag_image_in_stream(img)

        tags = [t.name for t in tags_result.tags] if tags_result.tags else []

        return process_text(f"Image shows: {captions}. Tags: {tags}")

    except Exception as e:
        print("Image error:", e)
        return "Error processing image."

# -----------------------------
# ROUTES
# -----------------------------
@app.route("/health")
def health():
    return "App is running!"


@app.route("/", methods=["GET", "POST"])
def index():
    response = ""

    try:
        if request.method == "POST":

            user_input = request.form.get("text_input")
            if user_input:
                response = process_text(user_input)

            elif "image" in request.files:
                image = request.files["image"]

                if image and image.filename:
                    filepath = os.path.join(app.config["UPLOAD_FOLDER"], image.filename)
                    image.save(filepath)
                    response = process_image(filepath)

    except Exception as e:
        print("Route error:", e)
        response = "Something went wrong."

    return render_template("index.html", response=response)

# -----------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
