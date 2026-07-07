import PIL.Image as Image
import torchvision.models as models

resnet_weights = models.ResNet50_Weights.DEFAULT
categories = resnet_weights.meta["categories"]
transforms = resnet_weights.transforms()

model = models.resnet50(weights=resnet_weights)
img = Image.open("neural_network/image.jpg").convert("RGB")
img = transforms(img).unsqueeze(0)

model.eval()

predict = model(img).squeeze()
response = predict.softmax(dim=0).sort(dim=0, descending=True)

for s, i in zip(response[0][:5], response[1][:5]):
    print(f"{categories[i]} {s:.4f}")
