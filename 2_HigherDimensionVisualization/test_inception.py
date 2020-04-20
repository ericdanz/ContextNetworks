import torchvision.models as models 
import torch
import numpy as np
import os
import cv2
import torchvision
import json

with open("class_list.json","r") as f:
    classes = json.load(f)

print(classes["400"])
googlenet = models.googlenet(pretrained=True)
def crop(image):
    print(image.shape)
    h = image.shape[0]
    w = image.shape[1]
    print(h,w)
    if h> w:
        #make w 224
        w_ratio =  224/w 
        h = w_ratio*h
        w = w_ratio*w
        print(h,w)
        h_diff =int( (h - 224)/2)
        image = cv2.resize(image,(0,0),fx=w_ratio,fy=w_ratio)
        img = image[h_diff:h_diff+224,:,:]

    if w >= h:
        #make h 224 
        h_ratio = 224/h
        h = h_ratio*h
        w = h_ratio*w
        w_diff = int((w - 224)/2)
        image = cv2.resize(image,(0,0),fx=h_ratio,fy=h_ratio)
        img = image[:,w_diff:w_diff+224,:]

    return img


def get_imagenet_images():
    filelist = os.listdir("imagen/imagen/")
    filelist = [os.path.join("imagen/imagen/",x) for x in filelist]
    img_list = []
    for filename in filelist:
        img_list.append([cv2.imread(filename),filename])
    return img_list


print("loaded")
print(googlenet)
for name, W in googlenet.named_parameters():
    print(name)
    if name =="inception5b.branch3.0.conv.weight":
        print(W.shape)

images = get_imagenet_images()
for image,filename in images:
    #squeeze and crop 
    img = crop(image)
    cv2.imshow("img",img)
    cv2.waitKey(100)
    img = torchvision.transforms.ToTensor()(img)

    img = torch.unsqueeze(img,0)
    print(img.shape)
    output = googlenet(img)
    print("class", classes[str(np.argmax(output.detach().numpy()))])
    print(filename)

