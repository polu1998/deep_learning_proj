# Final Competition-Steps for Replication of Results  

## Semantic Segmentation model Training:

The dataset_custom.py script loads the video frames along with their corresponding masks for training the deeplabv3_resnet101 model we've used for semantic segmentation.

To train the segmentation model, execute the following script-
```
python model.py 
```
Make sure the path to the training and validation dataset is changed in the following code snippet-

```
train_dataset = CustomDataset(root_dir='squashfs-root/dataset/train', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
val_dataset = CustomDataset(root_dir='squashfs-root/dataset/val', transform=transform)  # Adjust the path accordingly
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

```
Make sure to change the path to the csv log for the training loss, validation and accuracy is changed in the snippet-

```
csv_filename = 'deeplab.csv'
```

Make sure to change the path where the saved model will be found in the following code snippet-

```
torch.save(model.state_dict(), 'checkpoint_folder/deeplabv3_model_'+str(epoch)+'.pth')

```
## Inference:

Execute the following script to carry out inference on the hidden dataset.

Make sure to change the path where the trained model is saved following code snippet-  

```
checkpoint_path = '/content/drive/MyDrive/deeplabv3_model_14.pth'
```
Make sure the path to the hidden dataset is changed in the following code snippet-

```
base_path='./hidden'

```
Make sure the path to where the final inference result will be saved is changed in the following code snippet-

```
 torch.save(op_list.cpu(),'final_result.pth')

```










