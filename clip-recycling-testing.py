import pandas as pd
import torch
from PIL import Image
import open_clip
import os

# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Set Seeds
# For pytorch
torch.manual_seed(0)
# For CUDA
if torch.cuda.is_available():
    torch.cuda.manual_seed(0)
    torch.cuda.manual_seed_all(0)  # For multi-GPU setups
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
model = model.to(device)

tokenizer = open_clip.get_tokenizer('ViT-B-32')

results = {}

material_list = []
material_dict = {}
with open("./data/new_material_list.txt", "r") as f1:
    lines = f1.readlines()
    for line in lines: 
        material = line.split(" [")[0]
        material_list.append(material.strip())  # Trim any leading/trailing whitespace
        disposal = line.split("\'")[1]
        material_dict[material] = disposal
    f1.close()

text = tokenizer(material_list).to(device) # sets the parameters to material_list

input_files = ['plastic.txt', 'cardboard_paper.txt', 'metal.txt', 'electronics.txt', 'organic_matter.txt', 'wood.txt', 'glass.txt', 'textiles.txt', 'other.txt']
input_names = ['plastic', 'cardboard_paper', 'metal', 'electronics', 'organic_matter', 'wood', 'glass', 'textiles', 'other']

#####################################################################################################################
names_list=[]
urls_not_found = []
img_load_issue = []
for i in tqdm(range(len(input_files))):
    print(input_names[i])
    direc = "./dataset_final/"+input_names[i]
    if not os.path.exists(direc):
        os.makedirs(direc)

    with open(".data/" + input_files[i], "r") as f2: 
        lines = f2.readlines()
        other_list = []
        data = {'expected' : [], 'output' : [], 'output disposal' : [], 'probability' : [], 'url' : [], 'img_name':[]}
        for line in tqdm(lines):
            splits = line.split("https://")
            material = splits[0].strip()
            urls = splits[1:]
            url_list = []

            for part in urls:
                url = "https://" + part.strip()
                url_list.append(url)
                name = material.lower().replace(" ", "-").replace("(", " ").replace(")", " ") # might need to add the (1) and (2) etc. things later
                if name in names_list:  
                    j=1
                    while name + f"_{j}" in names_list:
                        j+=1
                    name = name+f"_{j}" 
                    names_list.append(name)
                else:
                    names_list.append(name)
                    
                try:
                    torch.hub.download_url_to_file(url, f"./dataset_final/{input_names[i]}/{name}.jpg")
                except:
                    print(url, name)
                    urls_not_found.append((url,name))
                    continue

                img_name = f"./dataset_final/{input_names[i]}/{name}.jpg"
                try:
                    image = preprocess(Image.open(img_name)).unsqueeze(0).to(device)
                    other_list.append(material)
                except:
                    img_load_issue.append(img_name)
                    continue

                with torch.no_grad(), torch.cuda.amp.autocast():
                    image_features = model.encode_image(image)
                    text_features = model.encode_text(text)
                    image_features /= image_features.norm(dim=-1, keepdim=True)
                    text_features /= text_features.norm(dim=-1, keepdim=True)

                    text_probs = (100*image_features @ text_features.T).softmax(dim=-1)

                    probabilities = []
                    counter = 0
                    for row in text_probs:
                        for column in row:
                            results[float(column)] = [material_list[counter], material_dict[material_list[counter]]] #disposal_list[counter]
                            probabilities.append(float(column))
                            counter += 1

                    sorted_probabilities = sorted(probabilities, reverse=True)
                    max_probability = sorted_probabilities[0]

                    data['expected'].append(material_dict[material])
                    data['output'].append(results[max_probability][0])
                    data['output disposal'].append(results[max_probability][1])
                    data['probability'].append(max_probability)
                    data['url'].append(url)
                    data['img_name'].append(img_name)

        f2.close()
    df = pd.DataFrame(data, index= other_list)
    df.to_csv("./results/" + str(input_names[i]) + '.csv')

