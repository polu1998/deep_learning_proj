import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader# Replace with your actual dataset module
from torchvision.models.segmentation import deeplabv3_resnet101
from tqdm import tqdm
import csv
from PIL import Image
import numpy as np
def load_deeplabv3_resnet101_from_checkpoint(checkpoint_path):
    # Load the model from a checkpoint
    model = deeplabv3_resnet101(pretrained=False,num_classes=49)
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint)
    model.cuda()
    model.eval()
    return model

def preprocess_image(image_path):
    # Preprocess the input image
    transform = transforms.Compose([
        transforms.ToTensor()

    ])

    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    return input_tensor

def perform_inference(model, input_tensor):
    # Perform inference
    with torch.no_grad():
        output = model(input_tensor)['out'][0]
    return output

def main():
    # Replace with the path to your checkpoint file
    checkpoint_path = '/content/drive/MyDrive/deeplabv3_model_14.pth'
    base_path='./hidden'

    # Load the model from the checkpoint
    model = load_deeplabv3_resnet101_from_checkpoint(checkpoint_path)

    # Replace with the path to your input image
    op_list=[]
    for i in range(15000,17000):
        image_path = base_path+'/video_'+str(i)+'/image_10.png'
        print(i)

        # Preprocess the input image
        input_tensor = preprocess_image(image_path)

        # Perform inference
        output = perform_inference(model, input_tensor.cuda())
        print(output.shape)
        predicted_class = torch.argmax(output, dim=0)
        op_list.append(predicted_class)

        # Post-process the output (e.g., visualize or save the result)
        # ...`

    op_list=torch.stack(op_list)
    torch.save(op_list.cpu(),'final_result.pth')

    print("Inference completed.")

if __name__ == "__main__":
    main()
