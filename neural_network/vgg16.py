from PIL import Image
import torchvision.models as models

vgg16_weights = models.VGG16_Weights.DEFAULT

transforms = vgg16_weights.transforms()
categories = vgg16_weights.meta["categories"]

img = Image.open("neural_network/image.jpg").convert("RGB")
img_t = transforms(img).unsqueeze(0)
print(img_t.shape)

model = models.vgg16(weights=vgg16_weights)
model.eval()

predict = model(img_t).squeeze()
response = predict.softmax(dim=0).sort(dim=0, descending=True)

for s, i in zip(response[0][:5], response[1][:5]):
    print(f"{categories[i]} {s:.4f}")
