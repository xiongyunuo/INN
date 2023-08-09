import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from torchvision import transforms
from tqdm import tqdm
import torch.nn.functional as F
import torchvision
number=6

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
    def __init__(self, block=BasicBlock, num_classes=10*number):
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

device="cuda"
model=DLA().to(device)
loss_fn=nn.CrossEntropyLoss()
optimizer=torch.optim.SGD(model.parameters(),lr=0.01,momentum=0.9,weight_decay=5e-4)
#optimizer=torch.optim.Adam(model.parameters(),lr=0.002,betas=(0.9,0.99))
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
batch_size=100
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
training_data=datasets.CIFAR10(root="data-cifar10", train=True, download=True, transform=transform_train)
test_data=datasets.CIFAR10(root="data-cifar10", train=False, download=True, transform=transform_test)
train_dataloader=DataLoader(training_data, batch_size=batch_size)
test_dataloader=DataLoader(test_data, batch_size=batch_size)
nt=0.2
ns = 0.0
hs = 0.0
ratio=-0.2
salt=0.2
k=20.4204
sign=torch.ones((3,32,32))
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
print(bgimages[0].size())
#print(bgimages[0])
def train(dataloader, model, loss_fn, optimizer):
    size, num_batches=len(dataloader.dataset), len(dataloader)
    correct=test_loss=0
    model.train()
    index=0
    for X, y in tqdm(list(dataloader), desc="Training..."):
        X,y=X.to("cpu"),y.to("cpu")
        #hs=((torch.rand(1)-0.5)/0.5)*0.3
        X=(X+nt*torch.rand_like(X))/(nt+1)
        r=torch.rand(X.size())
        r2=torch.randint(0,2,X.size())
        X=torch.where(r<salt,r2,X)
        #id=torch.randint(0,10,(1,)).item()%3
        #X[:,id,:,:]=X[:,id,:,:]-hs
        #hs=((torch.rand(1)-0.5)/0.5)*0.4
        #X[:,0,:,:]=X[:,0,:,:]-hs
        #hs=((torch.rand(1)-0.5)/0.5)*0.4
        #X[:,1,:,:]=X[:,1,:,:]-hs
        #hs=((torch.rand(1)-0.5)/0.5)*0.4
        #X[:,2,:,:]=X[:,2,:,:]-hs
        for i in range(0,X.size()[0]):
            X[i,:,:,:]=X[i,:,:,:]*sign
        tar=torch.zeros((1)).int()
        id=torch.randint(number,(1,)).item()%number
        imgg=bgimages[id]
        tar2=id
        img=torch.zeros((1,3,32,32))
        img[0,:,:,:]=imgg
        tar[0]=tar2
        for j in range(1,batch_size):
            tar2=torch.zeros((1)).int()
            id=torch.randint(number,(1,)).item()%number
            img22=bgimages[id]
            tar3=id
            img2=torch.zeros((1,3,32,32))
            img2[0,:,:,:]=img22
            #print(img2.size())
            tar2[0]=tar3
            img=torch.cat((img,img2),0)
            tar=torch.cat((tar,tar2),0)
            #print(img.size())
        img2=torch.zeros((batch_size,3,32,32))
        img2[:,:,:,:]=img
        X=X+ratio*img2
        for j in range(0,batch_size):
            y[j]=10*tar[j]+y[j]
        #X=torch.exp(X)
        X=torch.tanh(3*(X-0.5)/2+0.5)
        X,y=X.to(device),y.to(device)
        pred=model(X)
        loss=loss_fn(pred,y)
        correct += (pred.argmax(1) == y).type(torch.float32).sum().item()
        test_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        index+=batch_size
    test_loss /= num_batches
    correct /= size
    print(f"Train Error: Accuracy = {(100*correct):.4f}%, Avg loss = {test_loss:>8f} \n")
def test(dataloader, model, loss_fn):
    size, num_batches = len(dataloader.dataset), len(dataloader)
    test_loss, correct = 0, 0
    correct2=0
    model.eval()
    with torch.no_grad():
        index=0
        for X, y in tqdm(list(dataloader), desc="Testing..."):
            X,y=X.to("cpu"),y.to("cpu")
            #hs=((torch.rand(1)-0.5)/0.5)*0.0
            X=(X+torch.rand_like(X)*ns)/(ns+1)
            #id=torch.randint(0,10,(1,)).item()%3
            #X[:,id,:,:]=X[:,id,:,:]-hs
            #hs=((torch.rand(1)-0.5)/0.5)*0.0
            #X[:,0,:,:]=X[:,0,:,:]-hs
            #hs=((torch.rand(1)-0.5)/0.5)*0.0
            #X[:,1,:,:]=X[:,1,:,:]-hs
            #hs=((torch.rand(1)-0.5)/0.5)*0.0
            #X[:,2,:,:]=X[:,2,:,:]-hs
            for i in range(0,X.size()[0]):
                X[i,:,:,:]=X[i,:,:,:]*sign
            tar=torch.zeros((1)).int()
            id=torch.randint(number,(1,)).item()%number
            imgg=bgimages[id]
            tar2=id
            img=torch.zeros((1,3,32,32))
            img[0,:,:,:]=imgg
            tar[0]=tar2
            for j in range(1,batch_size):
                tar2=torch.zeros((1)).int()
                id=torch.randint(number,(1,)).item()%number
                img22=bgimages[id]
                tar3=id
                img2=torch.zeros((1,3,32,32))
                img2[0,:,:,:]=img22
                tar2[0]=tar3
                img=torch.cat((img,img2),0)
                tar=torch.cat((tar,tar2),0)
            img2=torch.zeros((batch_size,3,32,32))
            img2[:,:,:,:]=img
            X=X+ratio*img2
            #X=torch.exp(X)
            X=torch.tanh(3*(X-0.5)/2+0.5)
            X,y=X.to(device),y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            tar=tar.to("cuda")
            correct += ((pred.argmax(1)%10) == (y)).type(torch.float32).sum().item()
            correct2+=((pred.argmax(1)/10).int() == (tar)).type(torch.float32).sum().item()
            index+=batch_size
    test_loss /= num_batches
    correct /= size
    correct2 /= size
    print(f"Test Error: Accuracy = {(100*correct):.4f}%, Avg loss = {test_loss:>8f} \n")
    print(f"Test Error: Accuracy = {(100*correct2):.4f}%, Avg loss = {test_loss:>8f} \n")
epochs = 300
count=0
sign=torch.rand(3,32,32)
sign=torch.where(sign>0.5,1,1)
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    #train(train_dataloader2, model, loss_fn, optimizer,True)
    #train(train_dataloader3, model, loss_fn, optimizer,True)
    test(test_dataloader, model, loss_fn)
    #test(test_dataloader2, model, loss_fn)
    #test(test_dataloader3, model, loss_fn)
    if (t%30==29):
        print("Saving network\n")
        torch.save(model, "model_sign"+str(count)+".pt")
        torch.save(sign, "sign"+str(count)+".pt")
        torch.save(hs, "hs"+str(count)+".pt")
        count+=1
print("Saving network\n")
torch.save(model, "model_sign"+str(count)+".pt")
torch.save(sign, "sign"+str(count)+".pt")
torch.save(hs, "hs"+str(count)+".pt")
print("Done!")

#for t in range(epochs):
    #print(f"Epoch {t+1}\n-------------------------------")
    #train(train_dataloader, model, loss_fn, optimizer)
    #test(test_dataloader, model, loss_fn)
    #scheduler.step()
#print("Saving network\n")
#torch.save(model, "model_sign"+str(count)+".pt")
#torch.save(sign, "sign"+str(count)+".pt")
#torch.save(hs, "hs"+str(count)+".pt")
#print("Done!")
