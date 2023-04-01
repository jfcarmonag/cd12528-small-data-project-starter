import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

def load_and_standardize_data(path):
    df = pd.read_csv(path, sep=',')
    df = df.fillna(-99)
    df = df.values.reshape(-1, df.shape[1]).astype('float32')
    X_train, X_test = train_test_split(df, test_size=0.3, random_state=42)
    scaler = preprocessing.StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)   
    return X_train, X_test, scaler

class DataBuilder(Dataset):
    def __init__(self, path, train=True):
        self.X_train, self.X_test, self.standardizer = load_and_standardize_data(path)
        if train:
            self.x = torch.from_numpy(self.X_train)
            self.len=self.x.shape[0]
        else:
            self.x = torch.from_numpy(self.X_test)
            self.len=self.x.shape[0]
        del self.X_train
        del self.X_test 
    def __getitem__(self,index):      
        return self.x[index]
    def __len__(self):
        return self.len

class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()
        self.mse_loss = nn.MSELoss(reduction="sum")
    
    def forward(self, x_recon, x, mu, logvar):
        loss_MSE = self.mse_loss(x_recon, x)
        loss_KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return loss_MSE + loss_KLD

class Autoencoder(nn.Module):
    def __init__(self,D_in,H=50,H2=12,latent_dim=3):
        #Encoder
        super(Autoencoder,self).__init__()
        self.linear1=nn.Linear(D_in,H)
        self.lin_bn1 = nn.BatchNorm1d(num_features=H)
        self.linear2=nn.Linear(H,H2)
        self.lin_bn2 = nn.BatchNorm1d(num_features=H2)
        self.linear3=nn.Linear(H2,H2)
        self.lin_bn3 = nn.BatchNorm1d(num_features=H2)
        
        # Latent vectors
        self.fc1 = nn.Linear(H2, latent_dim)
        self.bn1 = nn.BatchNorm1d(num_features=latent_dim)
        self.fc21 = nn.Linear(latent_dim, latent_dim)
        self.fc22 = nn.Linear(latent_dim, latent_dim)

        # Sampling vector
        self.fc3 = nn.Linear(latent_dim, latent_dim)
        self.fc_bn3 = nn.BatchNorm1d(latent_dim)
        self.fc4 = nn.Linear(latent_dim, H2)
        self.fc_bn4 = nn.BatchNorm1d(H2)
        
        # Decoder
        self.linear4=nn.Linear(H2,H2)
        self.lin_bn4 = nn.BatchNorm1d(num_features=H2)
        self.linear5=nn.Linear(H2,H)
        self.lin_bn5 = nn.BatchNorm1d(num_features=H)
        self.linear6=nn.Linear(H,D_in)
        self.lin_bn6 = nn.BatchNorm1d(num_features=D_in)
        
        self.relu = nn.ReLU()
        
    def encode(self, x):
        lin1 = self.relu(self.lin_bn1(self.linear1(x)))
        lin2 = self.relu(self.lin_bn2(self.linear2(lin1)))
        lin3 = self.relu(self.lin_bn3(self.linear3(lin2)))

        fc1 = F.relu(self.bn1(self.fc1(lin3)))

        r1 = self.fc21(fc1)
        r2 = self.fc22(fc1)
        
        return r1, r2
    
    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu
        
    def decode(self, z):
        fc3 = self.relu(self.fc_bn3(self.fc3(z)))
        fc4 = self.relu(self.fc_bn4(self.fc4(fc3)))

        lin4 = self.relu(self.lin_bn4(self.linear4(fc4)))
        lin5 = self.relu(self.lin_bn5(self.linear5(lin4)))
        return self.lin_bn6(self.linear6(lin5))
        
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

def generate_fake(mu, logvar, no_samples, scaler, model):
    #With trained model, generate some data
    sigma = torch.exp(logvar/2)
    q = torch.distributions.Normal(mu.mean(axis=0), sigma.mean(axis=0))
    z = q.rsample(sample_shape=torch.Size([no_samples]))
    with torch.no_grad():
        pred = model.decode(z).cpu().numpy()
    fake_data = scaler.inverse_transform(pred)
    return fake_data

# When you have all the code in place to generate synthetic data, uncomment the code below to run the model and the tests. 
def main():
    # Get a device and set up data paths. You need paths for the original data, the data with just loan status = 1 and the new augmented dataset.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ORIGINAL_DATA_PATH = "data/loan_continuous.csv"
    LOAN_DATA_PATH = "data/loan_positive.csv"
    DATA_PATH = 'data/loan_continuous_expanded.csv'

    df = pd.read_csv(ORIGINAL_DATA_PATH)
    # Split the data out with loan status = 1
    positive_loan = df[df['Loan Status']==1]
    positive_loan.to_csv(LOAN_DATA_PATH)
    # Create DataLoaders for training and validation 
    



    train_dataset = DataBuilder(ORIGINAL_DATA_PATH, train=True)
    val_dataset = DataBuilder(ORIGINAL_DATA_PATH, train=False)
    batch_size = 64  

    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Train and validate the model 

    D_in = train_dataset.x.shape[1]
    

    model = Autoencoder(D_in=D_in).to(device)
    criterion = CustomLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    num_epochs = 100

    model.train()

    for epoch in range(num_epochs):
        train_loss = 0
        for batch in trainloader:
            batch = batch.to(device)
            
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(batch)
            
            loss = criterion(recon_batch, batch, mu, logvar)
            loss.backward()
            
            train_loss += loss.item()
            optimizer.step()

        print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss / len(trainloader.dataset):.6f}')
    model.eval()

    val_loss = 0
    with torch.no_grad():
        for batch in valloader:
            batch = batch.to(device)
            recon_batch, mu, logvar = model(batch)
            loss = criterion(recon_batch, batch, mu, logvar)
            val_loss += loss.item()

    print(f'Validation Loss: {val_loss / len(valloader.dataset):.6f}')

    scaler = trainloader.dataset.standardizer
    fake_data = generate_fake(mu, logvar, 50000, scaler, model)

    # Combine the new data with original dataset
    fake_df = pd.DataFrame(fake_data, columns=df.columns)
    merged_df = pd.concat([df, fake_df], ignore_index=True)
    merged_df.to_csv(DATA_PATH)
    test_model(DATA_PATH)

if __name__ == '__main__':
    main()
    print("done")