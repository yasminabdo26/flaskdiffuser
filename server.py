from flask import Flask, request
import socket
import datetime
from diffusers import DiffusionPipeline
# from diffusers import StableDiffusionPipeline
# import torch

app = Flask(__name__)

@app.route("/generate", methods=['POST'])
def generate():
    data = request.get_json()
    prompt = data["prompt"]
    
    # Local Only + CPU
    device = 'cpu'    
    model_id = r"C:\Users\1047281\AppData\Local\Diffusion\app\anything"
    pipe = DiffusionPipeline.from_pretrained(model_id, local_files_only=True)

    # Interenet + GPU
    # device = 'cuda'
    # model_id = "windwhinny/chilloutmix"
    # pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)

    host = socket.gethostname()
    ip = socket.gethostbyname(host)
    dt_now = datetime.datetime.now()
    timestamp = str(dt_now.strftime("%Y_%m_%d_%H_%M_%S"))

    pipe = pipe.to(device)
    image = pipe(prompt).images[0]    
    image.save("./img/" + timestamp + ".png")
    
    
    return ip + "/img/" + timestamp + ".png"

app.run(host="0.0.0.0", port=8000)
