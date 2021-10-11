import torch
import torchvision.models
from torchinfo import summary
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
import cv2 as cv
from torchvision import transforms
from PIL import Image



MODEL_PATH = r'C:\Users\Konrad\TCM_scripts\output\model_final.pth'

#model = torch.load(MODEL_PATH)
model.load_state_dict(torch.load(MODEL_PATH), False)
model.eval()

'''
model = MyModel() 
model.load_state_dict(torch.load('model_best.pth.tar')['state_dict'])
'''
transform = transforms.Compose([transforms.Resize(600)])



summary(model)
data = r'H:\Konrad\tcm_scan\20210621_092043\images\007_036.png'
im = Image.open(data)
img = transform(im)


print(model.parameters)

output = model(im)
