from flask import Flask, request, jsonify
import torch
from diffusers import StableDiffusionPipeline
import base64
import io
from PIL import Image

app = Flask(__name__)

# تحميل النموذج
model_id = "CompVis/stable-diffusion-v1-4"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe.to("cuda")  # إذا كنت تستخدم GPU على Railway

@app.route('/generate', methods=['POST'])
def generate_image():
    data = request.get_json()
    prompt = data.get("prompt", "")

    # توليد الصورة بناءً على الوصف
    image = pipe(prompt).images[0]

    # تحويل الصورة إلى base64
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format="PNG")
    img_base64 = base64.b64encode(img_byte_arr.getvalue()).decode("utf-8")

    return jsonify({"image": img_base64})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
