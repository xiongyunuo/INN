import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from torchvision import transforms
from tqdm import tqdm
class FNN(nn.Module):
    def __init__(self):
        super().__init__()
        #self.proc = nn.Sequential(
            #nn.Linear(784, 150), nn.ReLU(),
            #nn.Linear(150, 10), nn.ReLU(),
        #)
        self.proc = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=3, padding=2), nn.ReLU(),
            nn.AvgPool2d(kernel_size=3, stride=2),
            nn.Conv2d(6, 16, kernel_size=3), nn.ReLU(),
            nn.AvgPool2d(kernel_size=3, stride=2),
            nn.Flatten(),
            nn.Linear(16 * 5 * 5, 150), nn.ReLU(),
            nn.Linear(150, 100), nn.ReLU(),
            nn.Linear(100, 100)
        )
    def forward(self,x):
        return self.proc(x)
device="mps"
model=FNN().to(device)
loss_fn=nn.CrossEntropyLoss()
optimizer=torch.optim.SGD(model.parameters(),lr=0.01,momentum=0.9)
#optimizer=torch.optim.Adam(model.parameters(),lr=0.005,betas=(0.9,0.99))
batch_size=100
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
ns = 0.9
nt=0.0
hs = 0.0#
k=20.4204
sign=torch.ones((1,28,28))
fter=torch.ones((1,1,1,1))/1.0
sh=torch.ones((784))
count=0
ratio=0.3
for i in range(0,784):
    sh[count]=1
    count+=1
    if (count >= 784):
        break
    sh[count]=1
    count+=1
    if (count >= 784):
        break
sh=sh.view(1,28,28)
shuf=torch.randperm(60000)
shuf2=torch.randperm(10000)
def train(dataloader, model, loss_fn, optimizer, fm):
    size, num_batches=len(dataloader.dataset), len(dataloader)
    correct=test_loss=0
    model.train()
    index=0
    for X, y in tqdm(list(dataloader), desc="Training..."):
        #print(sign)
        #print(X.norm())
        X,y=X.to("cpu"),y.to("cpu")
        X=(X+torch.rand_like(X)*ns)/(ns+1)
        r=torch.rand(X.size())
        r2=torch.randint(0,2,X.size())
        X=torch.where(r<0.4,r2,X)
        X=nn.functional.conv2d(X,fter,padding='same')
        hs=((torch.rand(1).item()-0.5)/0.5)*0.0
        #X=X-hs
        for i in range(0,batch_size):
            X[i,:,:,:]=X[i,:,:,:]+hs*sh
            X[i,:,:,:]=X[i,:,:,:]*sign
        #X=torch.cat((X,1-X),0)
        #y=torch.cat((y,y),0)
        #print(X.norm())
        #exit()
        #if fm:
            #X=torch.cat((X,1-X),0)
            #y=torch.cat((y,y),0)
        tar=torch.zeros((1)).int()
        img,tar2=training_data[shuf[index]]
        tar[0]=tar2
        for j in range(1,batch_size):
            tar2=torch.zeros((1)).int()
            img2,tar3=training_data[shuf[index+j]]
            tar2[0]=tar3
            img=torch.cat((img,img2),0)
            tar=torch.cat((tar,tar2),0)
        img2=torch.zeros((batch_size,1,28,28))
        img2[:,0,:,:]=img
        X=X+ratio*img2
        for j in range(0,batch_size):
            y[j]=10*tar[j]+y[j]
        #X=torch.cat((X,img2),0)
        #y=torch.cat((y,tar+10),0)
        #X=torch.sin(k*X)**2
        X=torch.exp(X)
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
    #if fm:
        #correct /= size*2
    #else:
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
            X=(X+torch.rand_like(X)*nt)/(nt+1)
            X=nn.functional.conv2d(X,fter,padding='same')
            hs=((torch.rand(1).item()-0.5)/0.5)*0.0
            #X=X-hs
            for i in range(0,batch_size):
                X[i,:,:,:]=X[i,:,:,:]+hs*sh
                X[i,:,:,:]=X[i,:,:,:]*sign
            tar=torch.zeros((1)).int()
            img,tar2=test_data[shuf2[index]]
            tar[0]=tar2
            for j in range(1,batch_size):
                tar2=torch.zeros((1)).int()
                img2,tar3=test_data[shuf2[index+j]]
                tar2[0]=tar3
                img=torch.cat((img,img2),0)
                tar=torch.cat((tar,tar2),0)
            img2=torch.zeros((batch_size,1,28,28))
            img2[:,0,:,:]=img
            X=X+ratio*img2
            #X=torch.sin(k*X)**2
            X=torch.exp(X)
            X,y=X.to(device),y.to(device)
            tar=tar.to(device)
            pred = model(X)
            #print(pred.argmax(1))
            #print((pred.argmax(1)/10))
            #print(tar)
            test_loss += loss_fn(pred, y).item()
            correct += ((pred.argmax(1)%10) == (y)).type(torch.float32).sum().item()
            correct2+=((pred.argmax(1)/10).int() == (tar)).type(torch.float32).sum().item()
            index+=batch_size
    test_loss /= num_batches
    correct /= size
    correct2 /= size
    print(f"Test Error: Accuracy = {(100*correct):.4f}%, Avg loss = {test_loss:>8f} \n")
    print(f"Test Error: Accuracy = {(100*correct2):.4f}% \n")
epochs =300
count=0
sign=torch.rand(1,28,28)
sign=torch.where(sign>0.5,1,1)
#sign=sign.view(784)
#c=0
#while c<784:
#    for i in range(0,1):
#        if c>=784:
#            break
#        sign[c]=1
#        c+=1
#    for i in range(0,1):
#        if c>=784:
#            break
#        sign[c]=-1
#        c+=1
#sign=sign.view(1,28,28)
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer,False)
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
