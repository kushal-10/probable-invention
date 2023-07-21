from captions import generate_image_list, generate_captions, get_prompt_text

def main():

    '''
    Running this file will:
        1) Generate a list of image locations, so PIL.Image can load the images
        2) Loads the Model InstructBLIP from huggingface and generates the captions for all the 100 images
        3) Create a text file that can be prompted to a LLM to solve the task
    '''

    im_list = generate_image_list()
    captions = generate_captions(image_list=im_list)
    get_prompt_text(captions)

    return None


if __name__ == '__main__':
    main()
