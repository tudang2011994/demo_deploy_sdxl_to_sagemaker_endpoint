import torch

from diffusers  import StableDiffusionXLInpaintPipeline

from PIL import Image
import base64, io

#Init Pipe
def model_fn(model_dir):
    pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
        model_dir, torch_dtype = torch.float16, variant ='fp16', use_safetensors= True
    ).to('cuda')

    return pipe


def predict_fn(input_data, pipe):
    #Get request data
    prompt = input_data.get('prompt')
    img_b64 = input_data.get('image')
    mask_b64 = input_data.get('mask')

    #Convert image from json text to PIL Image
    init_img = Image.open(io.BytesIO(base64.b64decode(img_b64))).convert('RGB')
    mask_img = Image.open(io.BytesIO(base64.b64decode(mask_b64))).convert('RGB')

    #Inference model inpainting
    output = pipe(
        prompt= prompt,
        image = init_img,
        mask_image = mask_img,
        strength= 0.99,
        num_inference_steps  = 30
    ).images[0]

    #Convert image and sent response back
    buffer = io.BytesIO()
    output.save(buffer, format='PNG')
    
    return base64.b64encode(buffer.getvalue()).decode('utf-8') 