import torch #test new method
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from torchvision import transforms
from tqdm import tqdm
import torch.nn.functional as F
from torchvision.transforms import functional as F2
import torchvision
import matplotlib.pyplot as plt
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Root(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1):
        super(Root, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            stride=1, padding=(kernel_size - 1) // 2, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, xs):
        x = torch.cat(xs, 1)
        out = F.relu(self.bn(self.conv(x)))
        return out


class Tree(nn.Module):
    def __init__(self, block, in_channels, out_channels, level=1, stride=1):
        super(Tree, self).__init__()
        self.level = level
        if level == 1:
            self.root = Root(2*out_channels, out_channels)
            self.left_node = block(in_channels, out_channels, stride=stride)
            self.right_node = block(out_channels, out_channels, stride=1)
        else:
            self.root = Root((level+2)*out_channels, out_channels)
            for i in reversed(range(1, level)):
                subtree = Tree(block, in_channels, out_channels,
                               level=i, stride=stride)
                self.__setattr__('level_%d' % i, subtree)
            self.prev_root = block(in_channels, out_channels, stride=stride)
            self.left_node = block(out_channels, out_channels, stride=1)
            self.right_node = block(out_channels, out_channels, stride=1)

    def forward(self, x):
        xs = [self.prev_root(x)] if self.level > 1 else []
        for i in reversed(range(1, self.level)):
            level_i = self.__getattr__('level_%d' % i)
            x = level_i(x)
            xs.append(x)
        x = self.left_node(x)
        xs.append(x)
        x = self.right_node(x)
        xs.append(x)
        out = self.root(xs)
        return out


class DLA(nn.Module):
    def __init__(self, block=BasicBlock, num_classes=10):
        super(DLA, self).__init__()
        self.base = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(True)
        )

        self.layer1 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(True)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True)
        )

        self.layer3 = Tree(block,  32,  64, level=1, stride=1)
        self.layer4 = Tree(block,  64, 128, level=2, stride=2)
        self.layer5 = Tree(block, 128, 256, level=2, stride=2)
        self.layer6 = Tree(block, 256, 512, level=1, stride=2)
        self.linear = nn.Linear(512, num_classes)

    def forward(self, x):
        out = self.base(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

class FNN(nn.Module):
    def __init__(self):
        super().__init__()
        #self.proc = nn.Sequential(
            #nn.Linear(784, 150), nn.ReLU(),
            #nn.Linear(150, 10), nn.ReLU(),
        #)
        self.proc = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5, padding=2), nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5), nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(576, 120), nn.ReLU(),
            nn.Linear(120, 84), nn.ReLU(),
            nn.Linear(84, 10)
        )
        #self.proc = nn.Sequential(
            #nn.Conv2d(3, 32, kernel_size=3, padding='same'), nn.ReLU(),
            #nn.BatchNorm2d(32),
            #nn.Conv2d(32, 32, kernel_size=3, padding='same'), nn.ReLU(),
            #nn.BatchNorm2d(32),
            #nn.MaxPool2d(kernel_size=2),
            #nn.Dropout(0.2),
            #nn.Conv2d(32, 64, kernel_size=3, padding='same'), nn.ReLU(),
            #nn.BatchNorm2d(64),
            #nn.Conv2d(64, 64, kernel_size=3, padding='same'), nn.ReLU(),
            #nn.BatchNorm2d(64),
            #nn.MaxPool2d(kernel_size=2),
            #nn.Dropout(0.3),
            #nn.Conv2d(64, 128, kernel_size=3, padding='same'), nn.ReLU(),
            #nn.BatchNorm2d(128),
            #nn.Conv2d(128, 128, kernel_size=3, padding='same'), nn.ReLU(),
            #nn.BatchNorm2d(128),
            #nn.MaxPool2d(kernel_size=2),
            #nn.Dropout(0.4),
            #nn.Flatten(),
            #nn.Linear(4*4*128, 120), nn.ReLU(),
            #nn.BatchNorm1d(120),
            #nn.Dropout(0.5),
            #nn.Linear(120, 10)
        #)
    def forward(self,x):
        return self.proc(x)

batch_size=10
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
training_data=datasets.CIFAR10(root="data-cifar10", train=True, download=True, transform=transform_train)
test_data=datasets.CIFAR10(root="data-cifar10", train=False, download=True, transform=transform_test)
train_dataloader=DataLoader(training_data, batch_size=batch_size)
test_dataloader=DataLoader(test_data, batch_size=batch_size)
ns = 0.0
attack=0.0
hs = 0.0
models=[]
signs=[]
hss=[]
loops=11
for i in range(0,loops):
    models.append(torch.load("model_sign"+str(i)+".pt"))
    models[i].to("cuda")
    signs.append(torch.load("sign"+str(i)+".pt").to("cuda"))
    hss.append(torch.load("hs"+str(i)+".pt"))
for i in range(loops,100):
    models.append([])
    signs.append([])
    hss.append([])
models.append(torch.load("model_sign"+"100"+".pt"))
models[100].to("cuda")
signs.append(torch.load("sign"+"100"+".pt").to("cuda"))
hss.append(torch.load("hs"+"100"+".pt"))
loss_fn=nn.CrossEntropyLoss()
optimizer=torch.optim.SGD(models[100].parameters(),lr=0.01,momentum=0.9)
for i, xp in enumerate(models[100].parameters()):
    xp.requires_grad=True
device="cuda"
shuf=torch.randperm(50000)
shuf2=torch.randperm(10000)
bgimages=[]
bgimages.append(torchvision.io.read_image("./background/1.jpeg")/255.0)
bgimages.append(torchvision.io.read_image("./background/2.jpeg")/255.0)
bgimages.append(torchvision.io.read_image("./background/3.jpeg")/255.0)
bgimages.append(torchvision.io.read_image("./background/4.jpeg")/255.0)
bgimages.append(torchvision.io.read_image("./background/5.jpeg")/255.0)
bgimages.append(torchvision.io.read_image("./background/6.jpeg")/255.0)
#bgimages.append(torchvision.io.read_image("./background/7.jpeg")/255.0)
#bgimages.append(torchvision.io.read_image("./background/8.jpeg")/255.0)
#bgimages.append(torchvision.io.read_image("./background/9.jpeg")/255.0)
#bgimages.append(torchvision.io.read_image("./background/10.jpeg")/255.0)
number=6
print(bgimages[0].size())
def test(dataloader, loss_fn):
    size, num_batches = len(dataloader.dataset), len(dataloader)
    test_loss, correct = 0, 0
    correct2=0
    for i in range(0,loops):
        models[i].eval()
    index=0
    for X, y in tqdm(list(dataloader), desc="Testing..."):
        with torch.no_grad():
            #xx=(X+torch.rand_like(X)*ns)/(ns+1)
            xx=torch.clone(X)
            xx=xx.to("cpu")
            y=y.to("cpu")
            x=torch.clone(xx)
            hs=0.0
            xx-=hs
            for j in range(0,batch_size):
                xx[j,:,:,:]=xx[j,:,:,:]*(signs[100].to("cpu"))
        xx=xx.to(device)
        y=y.to(device)
        #xx.requires_grad=True
        #models[100].train()
        #pred=models[100](xx)
        #loss=loss_fn(pred,y)
        #optimizer.zero_grad()
        #loss.backward()
        #with torch.no_grad():
            #xx+=attack/10*torch.sign(xx.grad)
        #gra=attack/10*torch.sign(xx.grad)
        #for j in range(1,10):
            #optimizer.zero_grad()
            #xx.grad = None
            #xx.requires_grad=True
            #models[100].train()
            #pred=models[100](xx)
            #loss=loss_fn(pred,y)
            #optimizer.zero_grad()
            #loss.backward()
            #with torch.no_grad():
                #xx+=attack/10*torch.sign(xx.grad)
            #gra+=attack/10*torch.sign(xx.grad)
        #signs[100]=signs[100].to("cuda")
        #for j in range(0,batch_size):
            #gra[j,:,:,:]=gra[j,:,:,:]*signs[100]
        #optimizer.zero_grad()
        #xx.grad = None
        if index==0:
            with torch.no_grad():
                xx=torch.clone(X)
            for jj in range(0,1):
                print(jj)
                with torch.enable_grad():
                    xx=xx.to("cuda")
                    xx.requires_grad=True
                    models[100].train()
                    pred=models[100](xx)
                    loss=loss_fn(pred,y)
                    optimizer.zero_grad()
                    loss.backward()
                with torch.no_grad():
                    gd=attack/1*torch.sign(xx.grad)[0,:,:,:]
                    if jj==0:
                        gra=torch.zeros((batch_size,3,32,32)).to("cuda")
                        for j in range(0,batch_size):
                            gra[j,:,:,:]=gd
                    else:
                        for j in range(0,batch_size):
                            gra[j,:,:,:]+=gd
                    xx+=attack/1*torch.sign(xx.grad)
                #print(gra)
        xx=torch.clone(X)
        xx=xx.to("cpu")
        y=y.to("cpu")
        gra=gra.to("cuda")
        with torch.no_grad():
            pred=torch.zeros((10,10))
            for i in range(loops-3,loops):
                for jj in range(0,5):
                    hs=hss[i]
                    xx=torch.clone(x)
                    xx=xx.to("cuda")
                    xx=xx+gra
                    ss=0.0
                    r=torch.rand(xx.size())
                    sgn=torch.where(r<0.5,1,-1).to("cuda")
                    xx=xx+ss*sgn
                    mi=0
                    ma=1
                    xx=torch.where(xx<mi,mi,xx)
                    xx=torch.where(xx>ma,ma,xx)
                    if index==10:##
                        image=xx[0,:,:,:].cpu().permute(1,2,0).squeeze().numpy()
                        plt.imshow(image)
                        plt.axis('off')
                        plt.show()
                    xx=F2.normalize(xx,(0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                    xx=(xx+torch.rand_like(xx)*ns)/(ns+1)
                    r=torch.rand(xx.size()).to("cuda")
                    r2=torch.randint(0,2,xx.size()).to("cuda")
                    xx=torch.where(r<0.0,r2,xx)
                    xx=xx-hs
                    xx=xx.to("cpu")
                    y=y.to("cpu")
                    for j in range(0,batch_size):
                        xx[j,:,:,:]=xx[j,:,:,:]*(signs[i].to("cpu"))
                    tar=torch.zeros((1)).int()
                    id=torch.randint(number,(1,)).item()%number
                    img=bgimages[id]
                    tar2=id
                    img=img.reshape((1,3,32,32))
                    tar[0]=tar2
                    for j in range(1,batch_size):
                        tar2=torch.zeros((1)).int()
                        id=torch.randint(number,(1,)).item()%number
                        img2=bgimages[id]
                        tar3=id
                        img2=img2.reshape((1,3,32,32))
                        tar2[0]=tar3
                        img=torch.cat((img,img2),0)
                        tar=torch.cat((tar,tar2),0)
                    img2=torch.zeros((batch_size,3,32,32))
                    img2[:,:,:,:]=img
                    ratio=-0.2
                    xx=xx+ratio*img2
                    #xx=torch.exp(xx)
                    xx=torch.tanh(3*(xx-0.5)/2+0.5)
                    xx=xx.to(device)
                    y=y.to(device)
                    pred2=models[i](xx)
                    pred3=(pred2.argmax(1)).type(torch.int)%10
                    for z in range(0,10):
                        pred[z][pred3[z]]+=1
            y=y.to("cpu")
            correct += (pred.argmax(1) == y).type(torch.float32).sum().item()
            pred2=pred2.to("cpu")
            correct2 += ((pred2.argmax(1)/10).int() == tar).type(torch.float32).sum().item()
            index+=batch_size
    test_loss /= num_batches
    correct /= size
    correct2/=size
    print(f"Test Error: Accuracy = {(100*correct):.4f}%, Avg loss = {test_loss:>8f} \n")
    print(f"Test Error: Accuracy = {(100*correct2):.4f}%, Avg loss = {test_loss:>8f} \n")
epochs = 1
count=0
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    test(test_dataloader, loss_fn)
print("Done!")
