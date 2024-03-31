from basic_fcn import *
import time
from torch.utils.data import DataLoader
import torch
import gc
import voc
import torchvision.transforms as standard_transforms
import util
import numpy as np
import matplotlib.pyplot as plt

class MaskToTensor(object):
    def __call__(self, img):
        return torch.from_numpy(np.array(img, dtype=np.int32)).long()


def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.xavier_uniform_(m.weight.data)
        torch.nn.init.normal_(m.bias.data) #xavier not applicable for biases



#TODO Get class weights
def getClassWeights():
    # TODO for Q4.c || Caculate the weights for the classes
    raise NotImplementedError


mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
input_transform = standard_transforms.Compose([
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(*mean_std)
    ])

target_transform = MaskToTensor()

train_dataset =voc.VOC('train', transform=input_transform, target_transform=target_transform)
val_dataset = voc.VOC('val', transform=input_transform, target_transform=target_transform)
test_dataset = voc.VOC('test', transform=input_transform, target_transform=target_transform)

train_loader = DataLoader(dataset=train_dataset, batch_size= 16, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size= 16, shuffle=False)
test_loader = DataLoader(dataset=test_dataset, batch_size= 16, shuffle=False)

epochs =30

n_class = 21

fcn_model = FCN(n_class=n_class)
fcn_model.apply(init_weights)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
optimizer = torch.optim.Adam(fcn_model.parameters(), lr=0.0005)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=0.0001, last_epoch=-1, verbose=False)
criterion =  nn.CrossEntropyLoss()

fcn_model =  fcn_model.to(device)


# TODO
def train():
    """
    Train a deep learning model using mini-batches.

    - Perform forward propagation in each epoch.
    - Compute loss and conduct backpropagation.
    - Update model weights.
    - Evaluate model on validation set for mIoU score.
    - Save model state if mIoU score improves.
    - Implement early stopping if necessary.

    Returns:
        None.
    """

    best_iou_score = 0.0
    train_losses = []
    val_losses = []
    ave_val_iou = []
    val_acc= []

    for epoch in range(epochs):
        ts = time.time()
        ep_loss = []
        for iter, (inputs, labels) in enumerate(train_loader):
            # TODO  reset optimizer gradients
            optimizer.zero_grad()

            # both inputs and labels have to reside in the same device as the model's
            inputs =  inputs.to(device)# TODO transfer the input to the same device as the model's
            labels =   labels.to(device)# TODO transfer the labels to the same device as the model's

            outputs =  fcn_model.forward(inputs)# TODO  Compute outputs. we will not need to transfer the output, it will be automatically in the same device as the model's!

            loss =  criterion(outputs, labels) #TODO  calculate loss
            loss_clone = torch.clone(loss).item()
            ep_loss.append(loss_clone)

            # TODO  backpropagate
            loss.backward()
            # TODO  update the weights
            optimizer.step()
            


            if iter % 10 == 0:
                print("epoch{}, iter{}, loss: {}".format(epoch, iter, loss.item()))
        scheduler.step()
        train_losses.append(np.mean(ep_loss))
        print("Finish epoch {}, time elapsed {}".format(epoch, time.time() - ts))

        current_miou_score, ep_acc, ep_val_loss = val(epoch)
        val_losses.append(ep_val_loss) 
        ave_val_iou.append(current_miou_score)
        val_acc.append(ep_acc)

        if current_miou_score > best_iou_score:
            best_iou_score = current_miou_score
            torch.save(fcn_model.state_dict(), "./models/4a.pth")

    fcn_model.load_state_dict(torch.load("./models/4a.pth")) 

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    # Plotting train and validation losses on the first subplot
    axs[0].plot(train_losses, label='Train Loss')
    axs[0].plot(val_losses, label='Validation Loss')
    axs[0].set_title('Training / Validation Losses')
    axs[0].set_xlabel('Epochs')
    axs[0].set_ylabel('Loss')
    axs[0].legend()

    # Plotting average validation IoU and validation accuracy on the second subplot
    axs[1].plot(ave_val_iou, label='Average Validation IoU')
    axs[1].plot(val_acc, label='Validation Accuracy')
    axs[1].set_title('Validation Metrics')
    axs[1].set_xlabel('Epochs')
    axs[1].set_ylabel('Metric Value')
    axs[1].legend()

    plt.tight_layout()
    plt.show()
    
 #TODO
def val(epoch):
    """
    Validate the deep learning model on a validation dataset.

    - Set model to evaluation mode.
    - Disable gradient calculations.
    - Iterate over validation data loader:
        - Perform forward pass to get outputs.
        - Compute loss and accumulate it.
        - Calculate and accumulate mean Intersection over Union (IoU) scores and pixel accuracy.
    - Print average loss, IoU, and pixel accuracy for the epoch.
    - Switch model back to training mode.

    Args:
        epoch (int): The current epoch number.

    Returns:
        tuple: Mean IoU score and mean loss for this validation epoch.
    """
    fcn_model.eval() # Put in eval mode (disables batchnorm/dropout) !
    
    losses = []
    mean_iou_scores = []
    accuracy = []

    with torch.no_grad(): # we don't need to calculate the gradient in the validation/testing

        for iter, (input, label) in enumerate(val_loader):
            input = input.to(device)
            label =  label.to(device)
            output = fcn_model.forward(input)
            losses.append(criterion(output, label).cpu())
            mean_iou_scores.append(util.iou(output, label))
            accuracy.append(util.pixel_acc(output, label))
            




    print(f"Loss at epoch: {epoch} is {np.mean(losses)}")
    print(f"IoU at epoch: {epoch} is {np.mean(mean_iou_scores)}")
    print(f"Pixel acc at epoch: {epoch} is {np.mean(accuracy)}")

    fcn_model.train() #TURNING THE TRAIN MODE BACK ON TO ENABLE BATCHNORM/DROPOUT!!

    return np.mean(mean_iou_scores), np.mean(accuracy), np.mean(losses)

 #TODO
def modelTest():
    """
    Test the deep learning model using a test dataset.

    - Load the model with the best weights.
    - Set the model to evaluation mode.
    - Iterate over the test data loader:
        - Perform forward pass and compute loss.
        - Accumulate loss, IoU scores, and pixel accuracy.
    - Print average loss, IoU, and pixel accuracy for the test data.
    - Switch model back to training mode.

    Returns:
        None. Outputs average test metrics to the console.
    """

    fcn_model.eval()  # Put in eval mode (disables batchnorm/dropout) !
    losses = []
    mean_iou_scores = []
    accuracy = []


    with torch.no_grad():  # we don't need to calculate the gradient in the validation/testing

        for iter, (input, label) in enumerate(test_loader):
            input = input.to(device)
            label =  label.to(device)
            output = fcn_model.forward(input)
            losses.append(criterion(output, label).cpu())
            mean_iou_scores.append(util.iou(output, label))
            accuracy.append(util.pixel_acc(output, label))

    print(f"Loss at test set is {np.mean(losses)}")
    print(f"IoU at test set is {np.mean(mean_iou_scores)}")
    print(f"Pixel acc at test set  is {np.mean(accuracy)}")        


    fcn_model.train()  #TURNING THE TRAIN MODE BACK ON TO ENABLE BATCHNORM/DROPOUT!!


def exportModel(inputs):    
    """
    Export the output of the model for given inputs.

    - Set the model to evaluation mode.
    - Load the model with the best saved weights.
    - Perform a forward pass with the model to get output.
    - Switch model back to training mode.

    Args:
        inputs: Input data to the model.

    Returns:
        Output from the model for the given inputs.
    """

    fcn_model.eval() # Put in eval mode (disables batchnorm/dropout) !
    
    saved_model_path = "Fill Path To Best Model"
    # TODO Then Load your best model using saved_model_path
    
    inputs = inputs.to(device)
    
    output_image = fcn_model(inputs)
    
    fcn_model.train()  #TURNING THE TRAIN MODE BACK ON TO ENABLE BATCHNORM/DROPOUT!!
    
    return output_image

if __name__ == "__main__":

    val(0)  # show the accuracy before training
    train()
    modelTest()

    # housekeeping
    gc.collect()
    torch.cuda.empty_cache()