import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import basic_net 
from torchsummary import summary 
import torch.nn as nn
from focal_loss import FocalLoss

transform = transforms.Compose(
    [transforms.RandomHorizontalFlip(),
     transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
     transforms.RandomRotation(degrees=10),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

test_transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32,
                                          shuffle=True, num_workers=6)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=test_transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')



net= basic_net.SmallNet()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device,"device")

#criterion = nn.CrossEntropyLoss()
criterion = FocalLoss(gamma=2) 
optimizer = optim.AdamW(net.parameters(), lr=1e-3,weight_decay=0.001) 
net.to(device)
summary(net,(3,32,32))

for epoch in range(250):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data[0].to(device), data[1].to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        #print(i)
        running_loss += loss.item()
        if i % 250 == 249:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 250))
            running_loss = 0.0
    if epoch % 2 == 0: 
        continue
    total = 0
    correct = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images.to(device))
            outputs = outputs.to("cpu")
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * correct / total))
    total = 0
    correct = 0
    with torch.no_grad():
        for i,data in enumerate(trainloader):
            images, labels = data
            outputs = net(images.to(device))
            outputs = outputs.to("cpu")
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            if i > 10000:
                break

    print('Accuracy of the network on 10000 train images: %d %%' % (
        100 * correct / total))




print('Finished Training')

PATH = './cifar_net.pth'
net.to("cpu")
torch.save(net.state_dict(), PATH)


dataiter = iter(testloader)
images, labels = dataiter.next()

# print images
print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))

net= basic_net.SmallNet()
net.load_state_dict(torch.load(PATH))

outputs = net(images)

_, predicted = torch.max(outputs, 1)

print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                              for j in range(4)))

correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))

class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1


for i in range(10):
    print('Accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))

