# Flask Backend

from flask import Flask, render_template, request
import os

from openai import AzureOpenAI
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from msrest.authentication import CognitiveServicesCredentials

app = Flask(__name__)

# -----------------------------
# CONFIGURATION
# -----------------------------
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Environment variables (NEVER hardcode secrets)
AZURE_OPENAI_API_KEY = os.getenv("2RfVc87hTQsVac6pgV6ZNuogOZpdOk7DKgLPU6I5pSCcIzrmJOjPJQQJ99CDACR0EKYXJ3w3AAABACOGVSsI")
AZURE_OPENAI_ENDPOINT = os.getenv("https://healthcareopenai.openai.azure.com/")

AZURE_VISION_KEY = os.getenv("FDqjSBTzdzqQ4waQdb4An2bGNGymrN7AAfh7JIUmAUHTIp1qLPolJQQJ99CDACrIdLPXJ3w3AAAFACOGPbyv")
AZURE_VISION_ENDPOINT = os.getenv("https://healthcarevision.cognitiveservices.azure.com/")

# -----------------------------
# CLIENT INITIALIZATION (SAFE)
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
    print(f"OpenAI init error: {e}")

try:
    vision_client = ComputerVisionClient(
        AZURE_VISION_ENDPOINT,
        CognitiveServicesCredentials(AZURE_VISION_KEY)
    )
except Exception as e:
    print(f"Vision init error: {e}")

# -----------------------------
# TEXT PROCESSING
# -----------------------------
def process_text(user_input):
    if not client:
        return "AI service not configured."

    try:
        response = client.chat.completions.create(
            model="gpt-5-chat",
            messages=[
                {"role": "system", "content": "You are a healthcare assistant. Provide safe advice."},
                {"role": "user", "content": user_input}
            ]
        )

        return response.choices[0].message.content

    except Exception as e:
        print(f"Text processing error: {e}")
        return "Error processing your request. Please try again."

# -----------------------------
# IMAGE PROCESSING
# -----------------------------
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
        print(f"Image processing error: {e}")
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

            # TEXT INPUT
            user_input = request.form.get("text_input")
            if user_input:
                response = process_text(user_input)

            # IMAGE INPUT
            elif "image" in request.files:
                image = request.files["image"]

                if image and image.filename:
                    filepath = os.path.join(app.config["UPLOAD_FOLDER"], image.filename)
                    image.save(filepath)
                    response = process_image(filepath)

    except Exception as e:
        print(f"Route error: {e}")
        response = "Something went wrong. Check logs."

    # ✅ ALWAYS return template
    try:
        return render_template("index.html", response=response)
    except Exception as e:
        print(f"Template error: {e}")
        return f"Template error: {e}"

# -----------------------------
# LOCAL RUN (ignored in Azure)
# -----------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
