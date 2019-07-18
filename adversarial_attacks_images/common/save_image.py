import torchvision.transforms as transforms
def save_image(image,filename):
    if image.size(0) == 3:
        temp = transforms.ToPILImage()(image).convert('RGB')
        temp.save('./save/{}.png'.format(filename))
        return
    for i in range(image.size(0)):
        temp = transforms.ToPILImage()(image[i]).convert('RGB')
        temp.save('./save/{}{}.png'.format(filename,i))