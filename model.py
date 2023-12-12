import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from dataset_custom import CustomDataset # Replace with your actual dataset module
from torchvision.models.segmentation import deeplabv3_resnet101
from tqdm import tqdm
import csv

# Enable CUDA_LAUNCH_BLOCKING
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.enabled = True
torch.backends.cuda.matmul.allow_tf32 = False



# Set CUDA_LAUNCH_BLOCKING environment variable
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import numpy as np
total_batches=1000

def batch_segmentation_accuracy(predicted_masks, ground_truth_masks):
    """
    Compute the average pixel-wise accuracy of a batch of segmentation masks.

    Parameters:
    - predicted_masks: NumPy array, batch of predicted segmentation masks
    - ground_truth_masks: NumPy array, batch of ground truth segmentation masks

    Returns:
    - average_accuracy: Float, average pixel-wise accuracy across the batch
    """
    # Ensure the shapes of predicted and ground truth masks match
    assert predicted_masks.shape == ground_truth_masks.shape, "Mask shapes do not match"

    # Calculate accuracy for each pair of masks in the batch
    accuracies = []
    for predicted_mask, ground_truth_mask in zip(predicted_masks, ground_truth_masks):
        correct_pixels = np.sum(predicted_mask == ground_truth_mask)
        total_pixels = predicted_mask.size
        accuracy = correct_pixels / total_pixels
        accuracies.append(accuracy)

    # Calculate average accuracy
    average_accuracy = np.mean(accuracies)

    return average_accuracy

# Define the DeepLabv3 model
num_classes=49
model = deeplabv3_resnet101(pretrained=False, num_classes=num_classes)
# Replace num_classes with the number of classes in your segmentation task


# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
crop_size=(80,80)

# Define transformations for your dataset (modify as needed)
transform = transforms.Compose([
    transforms.ToTensor()

   


    # Add more transforms if required
])

# Create your dataset and dataloader
train_dataset = CustomDataset(root_dir='squashfs-root/dataset/train', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
val_dataset = CustomDataset(root_dir='squashfs-root/dataset/val', transform=transform)  # Adjust the path accordingly
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

# Training loop
num_epochs = 50  # Adjust as needed
batch_size=1

csv_filename = 'deeplab.csv'
with open(csv_filename, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(['Epoch', 'Training Loss', 'Validation Loss','Accuracy'])

for epoch in range(num_epochs):
    model.train()
    train_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}', dynamic_ncols=True)
    
    for batch in train_bar:
        images, masks = batch
        print(images.shape,masks.shape)
        images=images.view(batch_size * 22, 3, 160, 240).to(device)
        masks=masks.view(batch_size * 22,160, 240)
        print(images.shape,masks.shape)
    
        # Forward pass
        outputs = model(images)
        masks=masks.long().to(device)
        
        loss= criterion(outputs['out'], masks)
      

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_bar.set_postfix({'Loss': loss.item()}, refresh=True)
   

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}')
    model.eval()
    val_loss = 0.0
    val_bar = tqdm(val_loader, desc=f'Validation - Epoch {epoch+1}/{num_epochs}', dynamic_ncols=True)
    val_accuracy=0

    with torch.no_grad():
        for val_batch in val_bar:
            val_images, val_masks = val_batch
            val_images = val_images.view(batch_size * 22, 3, 160, 240).to(device)
            val_masks = val_masks.view(batch_size * 22,160, 240)
            val_masks=val_masks.long().to(device)

            val_outputs = model(val_images)
            val_loss+= criterion(val_outputs['out'], val_masks).item()
            val_bar.set_postfix({'Loss': val_loss / len(val_bar)}, refresh=True)
            predicted_masks = val_outputs['out'].argmax(dim=1).detach().cpu().numpy()
            ground_truth_masks = val_masks.cpu().numpy()
            accuracy = batch_segmentation_accuracy(predicted_masks, ground_truth_masks)
            val_accuracy += accuracy
            val_bar.set_postfix({'Loss': val_loss}, refresh=True)



    val_loss /= len(val_loader)
    val_accuracy /= total_batches
    print(f'Validation - Epoch [{epoch+1}/{num_epochs}], Segmentation Accuracy: {val_accuracy * 100:.2f}%')
    with open(csv_filename, 'a', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow([epoch + 1, loss.item(), val_loss,val_accuracy])



# Save the trained model
    torch.save(model.state_dict(), 'checkpoint_folder/deeplabv3_model_'+str(epoch)+'.pth')
