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
net = SmallNet()
#net.load_state_dict(torch.load("cifar_net.pth"))
net.load_state_dict(torch.load("cifar_net_l2.pth"))

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(net.parameters(), lr=1e-3, weight_decay=0.002)

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
while True:
    optimizer.zero_grad()
    images, labels = dataiter.next()
    # go through the data and  find the highly active paths through the network
    outputs = net(images)
    outputs = outputs.unsqueeze(0)
    loss = criterion(outputs, labels)
    loss.backward()
    # print the grads
    # print(net)
    layer3_grads = torch.abs(net.layers[3].conv1.weight.grad.clone())
    print(layer3_grads.max(), layer3_grads.min())
    print(net.layers[3].conv2.weight.grad.shape)
    print(net.layers[3].conv3.weight.grad.shape)
    # make an image of the gradients
    img = np.zeros((300, 480))
    chunk_width_1 = 480 // net.layers[3].conv1.weight.grad.shape[1]
    for i in range(net.layers[3].conv1.weight.grad.shape[1]):
        img[:100, i*chunk_width_1:i*chunk_width_1 +
            chunk_width_1] = torch.max(torch.abs(net.layers[3].conv1.weight.grad[:, i]))
    chunk_width_2 = 480 // net.layers[3].conv2.weight.grad.shape[0]
    for i in range(net.layers[3].conv2.weight.grad.shape[0]):
        img[100:200, i*chunk_width_2:i*chunk_width_2 +
            chunk_width_2] = torch.max(torch.abs(net.layers[3].conv2.weight.grad[i]))
    chunk_width_3 = 480 // net.layers[3].conv3.weight.grad.shape[0]
    for i in range(net.layers[3].conv3.weight.grad.shape[0]):
        img[200:, i*chunk_width_3:i*chunk_width_3 +
            chunk_width_3] = torch.max(torch.abs(net.layers[3].conv3.weight.grad[i, :]))
    print(img.max())
    old_max = img.max()
    img /= img.max()
    cv2.imshow("naive grads", img)
    cv2.waitKey(0)
    
    #automatically threshold based on gradients
    #continue to ratchet up zeroing until output doesn't match 

    print(outputs.shape,F.softmax(outputs),labels)
    img = np.zeros((300, 480))
    chunk_width_3 = 480 // net.layers[3].conv3.weight.grad.shape[0]
    largest_grads_3 = np.zeros(net.layers[3].conv3.weight.grad.shape[0])
    for i in range(net.layers[3].conv3.weight.grad.shape[0]):
        plt.scatter(range(net.layers[3].conv3.weight.grad.shape[1]), torch.abs(
            net.layers[3].conv3.weight.grad[i, :]),color="grey")
        temp_grads = torch.abs(net.layers[3].conv3.weight.grad[i,:])
        temp_grads[temp_grads<.002] = -1
        print("size ",temp_grads.size(),"num green" ,np.count_nonzero(temp_grads+1))
        plt.scatter(range(net.layers[3].conv3.weight.grad.shape[1]), temp_grads,color="green")
        plt.ylim([0, 0.1])
        plt.title("feature layer {}".format(i))
        plt.show(block=False)
        plt.pause(.3)
        plt.close()


        img[200:, i*chunk_width_3:i*chunk_width_3 +
            chunk_width_3] = torch.max(torch.abs(net.layers[3].conv3.weight.grad[i, :]))
        vals, inds = torch.max(
            torch.abs(net.layers[3].conv3.weight.grad[i, :]), 0)
        largest_grads_3[i] = torch.squeeze(inds)
    unique_largest_grads_3 = np.unique(largest_grads_3)
    continue

    chunk_width_2 = 480 // net.layers[3].conv2.weight.grad.shape[0]
    largest_grads_2 = np.zeros(net.layers[3].conv2.weight.grad.shape[0])
    for i in range(net.layers[3].conv2.weight.grad.shape[0]):
        img[100:200,i*chunk_width_2:i*chunk_width_2+chunk_width_2] = torch.max(torch.abs(net.layers[3].conv2.weight.grad[i]))

    chunk_width_1 = 480 // net.layers[3].conv1.weight.grad.shape[1]
    largest_grads_1 = np.zeros(net.layers[3].conv1.weight.grad.shape[1])
    print(net.layers[3].conv1.weight.grad.shape)
    # change the gradients to mask everything but the largest from previous 
    layer1_grads = net.layers[3].conv1.weight.grad[unique_largest_grads_3, :]
    for i in range(net.layers[3].conv1.weight.grad.shape[1]):
        img[:100,i*chunk_width_1:i*chunk_width_1+chunk_width_1] = torch.max(torch.abs(layer1_grads[:,i]))
        vals, inds = torch.max(torch.abs(layer1_grads[:,i]),0) 
        largest_grads_1[i] = torch.squeeze(inds)
        print(vals,inds)
    print(img.max())
    # img /= img.max() 
    img /= old_max 

    cv2.imshow("max grads",img)
    cv2.waitKey(0) 

    print(net.layers[4].conv1.weight.grad.shape)
    print(net.layers[5].conv1.weight.grad.shape)
    print(net.layers[6].conv1.weight.grad.shape)
    print(net.layers[7].conv1.weight.grad.shape)
    print(net.layers[8].conv1.weight.grad.shape)
    print(net.layers[9].conv1.weight.grad.shape)
    print(net.layers[10].conv1.weight.grad.shape)
    print(net.layers[11].conv1.weight.grad.shape)
    print(net.layers[12].conv1.weight.grad.shape)
    print(net.layers[13].conv1.weight.grad.shape)
    print(net.layers[14].conv1.weight.grad.shape)
    print(net.layers[15].conv1.weight.grad.shape)
    print(net.layers[18].weight.grad.shape)

    break
