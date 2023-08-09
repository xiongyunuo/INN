import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from tqdm import tqdm
import matplotlib.pyplot as plt

class FNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.proc = nn.Sequential(
            nn.Linear(784, 150), nn.ReLU(),
            nn.Linear(150, 10), nn.ReLU(),
        )
    def forward(self,x):
        return self.proc(x)
loss_fn=nn.CrossEntropyLoss()
batch_size=10
training_data=datasets.MNIST(root="data-mnist", train=True, download=True, transform=ToTensor())
test_data=datasets.MNIST(root="data-mnist", train=False, download=True, transform=ToTensor())
train_dataloader=DataLoader(training_data, batch_size=batch_size)
test_dataloader=DataLoader(test_data, batch_size=batch_size)
training_data2=datasets.FashionMNIST(root="data-fmnist", train=True, download=True, transform=ToTensor())
test_data2=datasets.FashionMNIST(root="data-fmnist", train=False, download=True, transform=ToTensor())
training_data3=datasets.KMNIST(root="data-kmnist", train=True, download=True, transform=ToTensor())
test_data3=datasets.KMNIST(root="data-kmnist", train=False, download=True, transform=ToTensor())
train_dataloader2=DataLoader(training_data2, batch_size=batch_size)
test_dataloader2=DataLoader(test_data2, batch_size=batch_size)
train_dataloader3=DataLoader(training_data3, batch_size=batch_size)
test_dataloader3=DataLoader(test_data3, batch_size=batch_size)
ns = 0.0
k=20.4204
hs = 500#
border=torch.zeros((1,28,28))
for i in range(0,28):
    for j in range(0,28):
        if (i<3 and i>24 and j<3 and j>24):
            border[:,i,j]=0.8
def train(dataloader, model, loss_fn, optimizer):
    size, num_batches=len(dataloader.dataset), len(dataloader)
    correct=test_loss=0
    model.train()
    for X, y in tqdm(list(dataloader), desc="Training..."):
        X=X-hs
        pred=model(X)
        print(pred.argmax(1))
        pred=model(1-X)
        print(pred.argmax(1))
        print()
        loss=loss_fn(pred,y)
        correct += (pred.argmax(1) == y).type(torch.float64).sum().item()
        test_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    test_loss /= num_batches
    correct /= size
    print(f"Train Error: Accuracy = {(100*correct):.4f}%, Avg loss = {test_loss:>8f} \n")
models=[]
signs=[]
hss=[]
loops=11
for i in range(0,loops):#
    models.append(torch.load("model_sign"+str(i)+".pt").to("cpu"))
    signs.append(torch.load("sign"+str(i)+".pt"))
    hss.append(torch.load("hs"+str(i)+".pt"))
for i in range(loops,100):
    models.append([])
    signs.append([])
    hss.append([])
models.append(torch.load("model_sign"+"100"+".pt"))
signs.append(torch.load("sign"+"100"+".pt"))
hss.append(torch.load("hs"+"100"+".pt"))
loss_fn=nn.CrossEntropyLoss()
optimizer=torch.optim.SGD(models[0].parameters(),lr=0.01,momentum=0.9)
for i, xp in enumerate(models[0].parameters()):
    xp.requires_grad=True
fter=torch.ones((1,1,1,1))/1.0
shuf2=torch.randperm(10000)
def test(dataloader, loss_fn):
    size, num_batches = len(dataloader.dataset), len(dataloader)
    test_loss, correct = 0, 0
    correct2=0
    matc=0
    for i in range(0,1):
        models[i].eval()
    if True:
        index=0
        for X, y in tqdm(list(dataloader), desc="Testing..."):
            with torch.no_grad():
                #xx=(X+torch.rand_like(X)*ns)/(ns+1)
                xx=torch.clone(X)
                #for j in range(0,batch_size):
                    #xx[j,:]=xx[j,:]+1*border
                x=torch.clone(xx)
                for j in range(0,batch_size):
                    xx[j,:]=xx[j,:]+1*border
                x=torch.clone(xx)
                hs=hss[100]#
                xx-=0*hs
                for j in range(0,batch_size):
                    xx[j,:,:,:]=xx[j,:,:,:]*signs[100]
            xx.requires_grad=True
            models[100].train()
            #print(xx.size())
            #print(models[0])
            pred=models[100](xx)
            loss=loss_fn(pred,y)
            optimizer.zero_grad()
            loss.backward()
            #print(xx)
            attack=0.3
            with torch.no_grad():
                xx+=attack/10*torch.sign(xx.grad)#
            #print(xx)
            #exit()
            #x=torch.clone(xx)
            gra=attack/10*torch.sign(xx.grad)#
            for j in range(1,10):
                pred=models[100](xx)
                loss=loss_fn(pred,y)
                optimizer.zero_grad()
                loss.backward()
                with torch.no_grad():
                    xx+=attack/10*torch.sign(xx.grad)#
                gra+=attack/10*torch.sign(xx.grad)#
            for j in range(0,batch_size):
                gra[j,:,:,:]=gra[j,:,:,:]*signs[100]
            optimizer.zero_grad()
            xx.grad = None
            #r=torch.rand(x.size())
            #r2=torch.randint(0,2,x.size())
            #x=torch.where(r<0.3,r2,x)
            #if index==0:
                #image = x[0,:,:,:].squeeze().numpy()  # Remove the batch dimension and convert to a NumPy array
                #plt.imshow(image, cmap="gray")
                #plt.axis('off')  # Turn off axis labels
                #plt.show()
            with torch.no_grad():
                apred=models[100](xx).argmax(1)
            with torch.no_grad():
                pred=torch.zeros((10,10))
                for i in range (loops-3,loops-2):#
                    for jj in range(0,5):
                        hs=hss[i]#
                        xx=x+gra
                        xx=(xx+torch.rand_like(xx)*ns)/(ns+1)
                        r=torch.rand(xx.size())
                        r2=torch.randint(0,2,xx.size())
                        xx=torch.where(r<0.0,r2,xx)
                        xx=nn.functional.conv2d(xx,fter,padding='same')
                        #xx=xx-hs
                        #for j in range(0,batch_size):
                            #xx[j,:,:,:]=xx[j,:,:,:]*signs[i]
                        #xx+=gra
                        tar=torch.zeros((1)).int()
                        id=torch.randint(60000,(1,)).item()%60000
                        img,tar2=training_data[id]
                        tar[0]=tar2
                        for j in range(1,batch_size):
                            tar2=torch.zeros((1)).int()
                            id=torch.randint(60000,(1,)).item()%60000
                            img2,tar3=training_data[id]
                            tar2[0]=tar3
                            img=torch.cat((img,img2),0)
                            tar=torch.cat((tar,tar2),0)
                        img2=torch.zeros((batch_size,1,28,28))
                        img2[:,0,:,:]=img
                        xx=1.0*xx+0.3*img2
                        #xx=torch.sin(k*xx)**2
                        xx=torch.exp(xx)
                        preddd=models[i](xx)
                        #print(y)
                        #print(preddd.argmax(1))
                        pred2=models[i](1-xx)
                        #print(pred2.argmax(1))
                        #print()
                        pred3=(preddd.argmax(1)%10).type(torch.int)
                        for z in range(0,10):
                            pred[z][pred3[z]]+=1
                correct += (pred.argmax(1) == y).type(torch.float64).sum().item()
                correct2 += (preddd.argmax(1) == y).type(torch.float64).sum().item()
                #matc+=(pred.argmax(1)==apred).type(torch.float64).sum().item()
                matc+=(pred2.argmax(1)==preddd.argmax(1)).type(torch.float64).sum().item()
                index+=batch_size
    test_loss /= num_batches
    correct /= size
    correct2 /=size
    matc/=size
    print(f"Test Error: Accuracy = {(100*correct):.4f}%, Avg loss = {test_loss:>8f} \n")
    print(f"Test Error: Accuracy = {(100*correct2):.4f}%,  \n")
    print(f"Match: {(100*matc):.4f}% \n")
epochs = 1
count=0
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    #train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, loss_fn)
print("Done!")
