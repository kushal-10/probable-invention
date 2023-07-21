from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
import torch
from PIL import Image
import pandas as pd
import csv

def generate_image_list():

    """
    Generate image locations for the selected 100 images and store them in a list
    """
    
    image_list = []
    df = pd.read_csv('the_hundread.csv')
    for i in range(len(df)):
        num = df['image_id'].loc[i]
        # Pad with zeros to get exact location of image : example 000000214547.jpg
        pad = '%012d' % num
        im = 'images/' + str(pad) + '.jpg'
        # append to a list of images to be used for blip
        image_list.append(im)

    return image_list

def generate_captions(image_list):
    """
    Takes the list of images as input and generates captions for the selected 100 images and store them in a list
    Return the caption list
    """
    
    model = InstructBlipForConditionalGeneration.from_pretrained("Salesforce/instructblip-vicuna-7b")
    processor = InstructBlipProcessor.from_pretrained("Salesforce/instructblip-vicuna-7b")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    captions = []

    for i in range(len(image_list)):
        raw_image = Image.open(image_list[i]).convert("RGB")
        prompt = "Give a detailed caption for this image"
        inputs = processor(images=raw_image, text=prompt, return_tensors="pt").to(device)
        outputs = model.generate(
            **inputs,
            do_sample=False,
            num_beams=5,
            max_length=256,
            min_length=1,
            top_p=0.9,
            repetition_penalty=1.5,
            length_penalty=1.0,
            temperature=1,
        )
        generated_text = processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
        captions.append(generated_text)

    return captions

def get_prompt_text(captions):
    with open('the_hundread.csv', 'r') as file:
        csv_reader = csv.DictReader(file)
    csv_data = list(csv_reader)

    # Generate the output text
    output_text = ''
    for i in range(len(csv_data)):
        instance_text = f'i_{i+1}:{{\n'
        #instance_text += f'C_{i+1}: "{txt_data[i].strip()}";\n'
        # instance_text += f'C_{i+1}: [placeholder_'+str(i+1)+']\n'
        instance_text += f'C_{i+1}: "{captions[i]}";\n'
        instance_text += f'Q_{i+1}: "{csv_data[i]["question"]}";\n'
        instance_text += f'A_{i+1}:{{\n'
        instance_text += f'A_{i+1}_1: "{csv_data[i]["choices"].split(",")[0].strip()}";\n'
        instance_text += f'A_{i+1}_2: "{csv_data[i]["choices"].split(",")[1].strip()}";\n'
        instance_text += f'A_{i+1}_3: "{csv_data[i]["choices"].split(",")[2].strip()}";\n'
        instance_text += f'A_{i+1}_4: "{csv_data[i]["choices"].split(",")[3].strip()}";\n'
        instance_text += '}};\n\n'
        output_text += instance_text

    # Write the output text to a new TXT file
    with open('prompting_set_vicuna7B.txt', 'w') as file:
        file.write(output_text)