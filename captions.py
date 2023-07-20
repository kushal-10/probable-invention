import requests
from PIL import Image
from transformers import AutoProcessor, Blip2ForConditionalGeneration
import torch
import pandas as pd
import csv

def generate_captions():

    """
    Generate Captions for the selected 100 images and store them in a list
    """
    # GET LIST OF IMAGES
    image_list = []
    df = pd.read_csv('the_hundread.csv')
    for i in range(len(df)):
        num = df['image_id'].loc[i]
        # Pad with zeros to get exact location of image : example 000000214547.jpg
        pad = '%012d' % num
        im = 'images/' + str(pad) + '.jpg'
        # append to a list of images to be used for blip
        image_list.append(im)

    
    # GET CORRESPONDING CAPTIONS
    processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
    # model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16)
    model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b")  
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    image = Image.open(image_list[0])
    # inputs = processor(image, return_tensors="pt").to(device, torch.float16)
    inputs = processor(image, return_tensors="pt").to(device)

    generated_ids = model.generate(**inputs, max_new_tokens=20)
    print(len(generated_ids))
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    print(generated_text)
     
    
generate_captions()

