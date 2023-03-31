from flask import Flask, request
import socket
import datetime
from diffusers import DiffusionPipeline
import torch

app = Flask(__name__)

@app.route("/generate", methods=['POST'])
def generate():
    data = request.get_json()
    prompt = data["prompt"]
    model_id = "Model_path_or_Model_Id"
    
    device = 'cuda'
    # device = 'cpu'
    
    # Online
    pipe = DiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    # Local Only
    # pipe = DiffusionPipeline.from_pretrained(model_id, local_files_only=True)
    
    pipe = pipe.to(device)
    image = pipe(prompt).images[0]    
    image.save("./img/" + prompt.replace(",", "_").replace(" ", "_") + ".png")
    
    host = socket.gethostname()
    ip = socket.gethostbyname(host)
    dt_now = datetime.datetime.now()
    timestamp = datetime.strptime(dt_now, "%Y_%m_%d_%H_%M_%S")
    
    return timestamp + ip + "/img/" + prompt.replace(",", "_").replace(" ", "_") + ".png"

app.run(host="0.0.0.0", port=Your_Port_Number)
