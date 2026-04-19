# Flask Backend

from flask import Flask, render_template, request
from openai import AzureOpenAI
import os
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from msrest.authentication import CognitiveServicesCredentials

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "static/uploads"

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Azure OpenAI
client = AzureOpenAI(
  api_key="2RfVc87hTQsVac6pgV6ZNuogOZpdOk7DKgLPU6I5pSCcIzrmJOjPJQQJ99CDACR0EKYXJ3w3AAABACOGVSsI",
  api_version="2024-02-15-preview",
  azure_endpoint= "https://healthcareopenai.openai.azure.com/"
)

# Azure Vision
vision_client = ComputerVisionClient(
    "https://healthcarevision.cognitiveservices.azure.com/",
    CognitiveServicesCredentials("FDqjSBTzdzqQ4waQdb4An2bGNGymrN7AAfh7JIUmAUHTIp1qLPolJQQJ99CDACrIdLPXJ3w3AAAFACOGPbyv")
)

# Text processing
def process_text(user_input):
    response = client.chat.completions.create(
        model="gpt-5-chat",
        messages=[
            {"role": "system", "content": "You are a healthcare assistant. Provide safe advice."},
            {"role": "user", "content": user_input}
        ]
    )
    return response.choices[0].message.content


# Image processing
def process_image(image_path):
    with open(image_path, "rb") as img:
        desc = vision_client.describe_image_in_stream(img)

    captions = [c.text for c in desc.captions]

    tags_result = vision_client.tag_image_in_stream(open(image_path, "rb"))
    tags = [t.name for t in tags_result.tags]

    return process_text(f"Image shows: {captions}. Tags: {tags}")


@app.route("/", methods=["GET", "POST"])
def index():
    response = ""

    if request.method == "POST":

        # Text input
        if "text_input" in request.form and request.form["text_input"]:
            user_input = request.form["text_input"]
            response = process_text(user_input)

        # Image input
        elif "image" in request.files:
            image = request.files["image"]

            if image.filename != "":
                filepath = os.path.join(app.config["UPLOAD_FOLDER"], image.filename)
                image.save(filepath)
                response = process_image(filepath)

    return render_template("index.html", response=response)


#if __name__ == "__main__":
#    app.run(debug=True)

if __name__ == "__main__":
    app.run()