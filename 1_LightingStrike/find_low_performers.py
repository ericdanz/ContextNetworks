import torch
from basic_net import SmallNet
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch.nn.functional as F


def create_vis_of_grads(grads): 
    width = 960
    img = np.zeros((len(grads)*50,width,3),dtype=np.uint8)
    #assumes the weights are conncted to each other
    #visualize the first dimension 
    for j,grad in enumerate(grads):
        print(j,"grad")
        grad_shape0 = np.zeros(grad.shape[0])
        chunk_size = width//grad.shape[0]
        print(chunk_size)
        for i in range(grad.shape[0]):
            grad_shape0[i] = torch.sum(torch.abs(grad[i,:])).data.numpy() 
        print(grad_shape0.max(),'grad 0 max')
        grad_shape0 /= grad_shape0.max()
        grad_shape0 *= 255
        for i in range(grad.shape[0]):
            img[j*50:j*50+25,chunk_size*i:(i+1)*chunk_size,:] =  grad_shape0[i]
        grad_shape1 = np.zeros(grad.shape[1])
        chunk_size = width//grad.shape[1]
        for i in range(grad.shape[1]):
            grad_shape1[i] = torch.sum(torch.abs(grad[:,i])).data.numpy() 
        print(grad_shape1.max(),'grad 1 max')
        grad_shape1 /= grad_shape1.max()
        grad_shape1 *= 255
        for i in range(grad.shape[1]):
            img[j*50+25:(j+1)*50,chunk_size*i:(i+1)*chunk_size,:] =  grad_shape1[i]
    return img




net = SmallNet()
#net.load_state_dict(torch.load("cifar_net.pth"))
#net.load_state_dict(torch.load("cifar_net_l1_replace_low.pth"))
net.load_state_dict(torch.load("cifar_net_l2.pth"))


criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(net.parameters(), lr=1e-3, weight_decay=0.00)

test_transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=test_transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=1,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


dataiter = iter(testloader)
for i in range(50):
    #optimizer.zero_grad()
    images, labels = dataiter.next()
    # go through the data and  find the highly active paths through the network
    outputs = net(images)
    outputs = outputs.unsqueeze(0)
    loss = criterion(outputs, labels)
    loss.backward()
    # print the grads
    # print(net)
    if i % 10 == 1:
        print(i)

#automatically threshold based on gradients
#continue to ratchet up zeroing until output doesn't match 

print(net)
conv_layers = []
for i in range(15,2,-1):
    conv_layers.append(net.layers[i].conv3.weight.grad)
    conv_layers.append(net.layers[i].conv1.weight.grad)
img = create_vis_of_grads(conv_layers)
    # [net.layers[15].conv3.weight.grad,net.layers[15].conv1.weight.grad,
    # net.layers[14].conv3.weight.grad,net.layers[14].conv1.weight.grad,
    # net.layers[13].conv3.weight.grad,net.layers[13].conv1.weight.grad,
    # net.layers[12].conv3.weight.grad,net.layers[12].conv1.weight.grad,
    # net.layers[11].conv3.weight.grad,net.layers[11].conv1.weight.grad])
cv2.imshow("img",img)
cv2.imwrite("l2_trained.jpg",img)
cv2.waitKey(0)
