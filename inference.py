mport torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from dataset_custom import CustomDataset # Replace with your actual dataset module
from torchvision.models.segmentation import deeplabv3_resnet101
from tqdm import tqdm
import csv

def load_deeplabv3_resnet101_from_checkpoint(checkpoint_path):
    # Load the model from a checkpoint
    model = deeplabv3_resnet101(pretrained=False)
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
    checkpoint_path = 'checkpoint_folder/deeplabv3_model_15.pth'

    # Load the model from the checkpoint
    model = load_deeplabv3_resnet101_from_checkpoint(checkpoint_path)

    # Replace with the path to your input image
    op_list=[]
    for i in range(15000,17000):
        image_path = 'hidden/video/'+str(i)+'/image_22.png'

        # Preprocess the input image
        input_tensor = preprocess_image(image_path)

        # Perform inference
        output = perform_inference(model, input_tensor)
        op_list.append(output)

        # Post-process the output (e.g., visualize or save the result)
        # ...`
   
    op_list=torch.stack(op_list)
    np.save('final_result.npy',op_list)

    print("Inference completed.")

if __name__ == "__main__":
    main()
    