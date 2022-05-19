import torch; torch.manual_seed(0)
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions
##import matplotlib.pyplot as plt; plt.rcParams['figure.dpi'] = 200

## https://avandekleut.github.io/vae/
class VariationalEncoder(nn.Module):
  def __init__(self, latent_dims):
    super(VariationalEncoder, self).__init__()

    self.fc    = nn.Linear(784, 512)
    self.mu    = nn.Linear(512, latent_dims)
    self.sigma = nn.Linear(512, latent_dims)
    self.N = torch.distributions.Normal(0, 1)
    #self.N.loc = self.N.loc.cuda() # hack to get sampling on the GPU
    #self.N.scale = self.N.scale.cuda()
    self.kl = 0

    self.decoder = nn.Sequential( 
                      nn.Linear(latent_dims, 512), 
                      nn.ReLU(),
                      nn.Linear(512, 784),         
                      nn.Sigmoid()
                  )

  def encoder(self, x):
    z = F.relu( self.fc(x) )
    mu = self.mu( z )
    sigma = self.sigma( z )
    z = mu + sigma*self.N.sample(mu.shape)  # reparemetter
    self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()
    return z

  def forward(self, x):
    z = self.encoder(x)
    z = self.decoder(z)
    return z  


from tqdm import tqdm
from MNIST_loader import mnist_dataset
x_train, y_train, x_test, y_test = mnist_dataset()
x_train = torch.tensor( x_train.reshape((len(x_train),1,784))).float()
x_test  = torch.tensor( x_test.reshape((len(x_test),1,784))  ).float()
#y_train = torch.tensor( y_train.reshape((len(y_train),1,10)) ).float()
#y_test  = torch.tensor( y_test.reshape((len(y_test),1,10))   ).float()

device ="CPU"
latent_dims = 2
autoencoder = VariationalEncoder(latent_dims) #.to(device) # GPU

LOSS =[] 
print("training")
opt = torch.optim.Adam(autoencoder.parameters())
for epoch in tqdm(range(1)):
  print("Epoch: ", epoch)
  for x in tqdm(x_train):
    #x = x.to(device) # GPU
    opt.zero_grad()
    x_hat = autoencoder(x)
    #print( x_hat.shape, x.shape)
    loss = ((x - x_hat)**2).sum()   # normal autoencoder loss function
    loss.backward()
    LOSS.append(loss.detach().numpy())
    opt.step()

  from bokeh.plotting import figure, show
  # create a new plot with a title and axis labels
  p = figure(title="dynamics & reward model losses", x_axis_label="Iteration", y_axis_label="loss")
  # add a line renderer with legend and line thickness
  p.line(np.arange(len(LOSS)), LOSS, legend_label="Loss", line_color="blue", line_width=2)
  show(p) # show the results
