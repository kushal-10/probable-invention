# probable-invention
Repository for Project 4 of the course Project Module : Models that Explain Themselves organized at University of Potsdam for Summer Semester 2023

## Introduction

The goal here is to pick a multimodal dataset, and use a LLM to solve the task describe in the dataset. For the task [AOK-VQA](https://allenai.org/project/a-okvqa/home) dataset is selected and to solve the task, [InstructBLIP](https://huggingface.co/docs/transformers/main/model_doc/instructblip) and ChatGPT are used.

## Task
An image is given along with a question and 4 possible answers. (More information about the dataset/questions/answers can be found on[AOK-VQA](https://allenai.org/project/a-okvqa/home) website). The task at hand is to prompt any LLM with the 2 modalities : image and question and make the model solve the task. To proceed, a caption of the image is first generated using img-to-text model InstructBLIP. After getting the captions list, ChatGPT is prompted with the following prompt :

```
Below you will find instances i_1 through i_100. Each i_n consists of the following elements: A caption C_n, describing an image, A question Q_n about the image described by C_n, A set A_n of four possible answers to Q_n, named A_n_1, A_n_2, A_n_3, A_n_4. For each instance, select the most likely answer, print its variable name and its content. Additionally provide an explanation "E-n" explaining why the answer was selected. After your prediction also state which of the 4 following knowledge types you used to generate the explanation E_n: 1) Commonsense - Knowledge about the world that humans learn from their everyday experiences (e.g., many donuts being made in a cart implies they are for sale rather than for personal consumption), 2) Visual - Knowledge of concepts represented visually (e.g., muted color pallets are associated with the 1950s), 3) Knowledge bases - Knowledge obtained from textbooks, Wikipedia and other textual sources (e.g., hot dogs were invented in Austria) or 4) Physical - Knowledge about the physics of the world (e.g., shaded areas have a lower temperature than other areas)." Combine all information and Provide the answer directly in a single JSON code format containing: 
{
"instance": "i_n",
"caption": "C_n",
"question": "Q_n",
"possible_answers ": ["..", "..", "..", ".."]
"gpt_answer": "your answer",
"knowledge_type": "Which knowledge type you used (1,2,3, or 4)",
"explanation": "your explanation E_n",
 
} 

```
After prompting this gpt.JSON file is created from the outputs given by ChatGPT. The next part is to measure how good the generated explanations are by 1) Using BLEU metric and 2) Using manual annotations

## Usage

Run main.py file to do the whole process:

```
python3 main.py
```

THe whole process consists of

1) Generate a list of image locations, so PIL.Image can load the images

2) Loads the Model InstructBLIP from huggingface and generates the captions for all the 100 images

3) Create a text file that can be prompted to a LLM to solve the task (prompting_set_vicuna7B.txt)

This code was initially run on Google Colab Pro on A100 GPU. So it is recommended to use A100. Check Captions_Vicuna7B.ipynb for more information. 
