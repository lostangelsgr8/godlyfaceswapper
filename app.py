from flask import Flask, render_template, request
import os
import cv2
import numpy as np
from insightface.app import FaceAnalysis
from insightface.model_zoo import model_zoo
import torch
from realesrgan.utils import RealESRGANer
from realesrgan.archs.srvgg_arch import SRVGGNetCompact

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
OUTPUT_FOLDER = 'static/outputs'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load InsightFace model
face_app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
face_app.prepare(ctx_id=0, det_size=(640, 640))
swapper = model_zoo.get_model("inswapper_128.onnx", root=os.path.expanduser("~/.insightface/models"))

# Optional: Load RealESRGAN upsampler
model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64,
                        num_conv=32, upscale=4, act_type='prelu')
upsampler = RealESRGANer(
    scale=4,
    model_path='weights/RealESRGAN/RealESRGAN_x4plus.pth',
    model=model,
    tile=0,
    tile_pad=10,
    pre_pad=0,
    half=False
)

@app.route('/upload', methods=['POST'])
def upload():
    face_file = request.files['face_image']
    target_files = request.files.getlist('target_images')

    face_path = os.path.join(UPLOAD_FOLDER, 'face.jpg')
    face_file.save(face_path)
    face_img = cv2.imread(face_path)

    result_paths = []

    for i, target in enumerate(target_files):
        target_path = os.path.join(UPLOAD_FOLDER, f'target_{i}.jpg')
        target.save(target_path)
        target_img = cv2.imread(target_path)

        faces = face_app.get(target_img)
        if faces:
            target_face = faces[0]
            swapped = swapper.get(target_img, target_face, face_img)

            # Optional: Upscale result
            output, _ = upsampler.enhance(swapped, outscale=1)

            output_path = os.path.join(OUTPUT_FOLDER, f'swapped_{i}.jpg')
            cv2.imwrite(output_path, output)
            result_paths.append(output_path)

    return render_template('result.html', targets=result_paths)
