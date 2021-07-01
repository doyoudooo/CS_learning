from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

writer=SummaryWriter('logs')
img=Image.open('image/img.png')
print(img)#PIL

trans_totensor=transforms.ToTensor()
tensor_img=trans_totensor(img)#tensor
writer.add_image('to_tensor',tensor_img)
writer.close()

#normalize
print(tensor_img[0][0][0])
trans_norm=transforms.Normalize([1,2,4],[4,5,1])
img_norm=trans_norm(tensor_img)
print(img_norm[0][0][0])
writer.add_image('normalize',tensor_img,1)

writer.close()


#resize
print(img.size)
trans_resize=transforms.Resize((512,512))
#img PIL —>reszie  img_size  PIL
img_resize=trans_resize(img)
#img_resize  PIL —> totensor _>img_resize  tensor
img_resize=trans_totensor(img_resize)
print(img_resize)

writer.add_image('resize',img_resize,0)

writer.close()

# compose ()   resize_2
trans_resize_2=transforms.Resize(512)
trans_compose=transforms.Compose([trans_resize_2,trans_totensor])
img_resize_2=trans_compose(img)
writer.add_image('resize',img_resize_2,1)

#randomCrop
tran_random=transforms.RandomCrop(400)
trans_compose_2=transforms.Compose([tran_random,trans_totensor])
for i in range(10):
    img_crop=trans_compose_2(img)
    writer.add_image('randomcrop',img_crop,i)

writer.close()
