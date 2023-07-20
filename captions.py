# import requests
# from PIL import Image
# from transformers import AutoProcessor, Blip2ForConditionalGeneration
# import torch
# import pandas as pd
# import csv


# FOR LAVIS
import pandas as pd
import torch
from PIL import Image
import requests
from lavis.models import load_model_and_preprocess

# def generate_captions():

#     """
#     Generate Captions for the selected 100 images and store them in a list
#     """
#     # GET LIST OF IMAGES
#     image_list = []
#     df = pd.read_csv('the_hundread.csv')
#     for i in range(len(df)):
#         num = df['image_id'].loc[i]
#         # Pad with zeros to get exact location of image : example 000000214547.jpg
#         pad = '%012d' % num
#         im = 'images/' + str(pad) + '.jpg'
#         # append to a list of images to be used for blip
#         image_list.append(im)

    
#     # GET CORRESPONDING CAPTIONS
#     processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
#     model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16)
#     # model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b")  
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     model.to(device)

#     image = Image.open(image_list[3])
#     inputs = processor(image, return_tensors="pt").to(device, torch.float16)
#     # inputs = processor(image, return_tensors="pt").to(device)

#     generated_ids = model.generate(**inputs, max_new_tokens=200)
#     print(len(generated_ids))
#     generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
#     print(generated_text)
     
    
# # generate_captions()

def salesforce_lavis_gen():

    image_list = []
    df = pd.read_csv('the_hundread.csv')
    for i in range(len(df)):
        num = df['image_id'].loc[i]
        # Pad with zeros to get exact location of image : example 000000214547.jpg
        pad = '%012d' % num
        im = 'images/' + str(pad) + '.jpg'
        # append to a list of images to be used for blip
        image_list.append(im)

    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    raw_image = Image.open(image_list[0]).convert("RGB")
    model, vis_processors, _ = load_model_and_preprocess(
        name="blip2_t5", model_type="pretrain_flant5xl", is_eval=True, device=device, torch_dtype=torch.float16
    )
    image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
    caption_list = model.generate({"image": image}, use_nucleus_sampling=True, num_captions=1, torch_dtype=torch.float16)
    print(caption_list)
    
    return None

salesforce_lavis_gen()
