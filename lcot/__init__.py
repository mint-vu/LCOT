import numpy as np
from lcot.measures import measure

def vonmises_kde(data, kappa, n_bins=100):
    from scipy.special import i0
    bins = np.linspace(-np.pi, np.pi, n_bins)
    # integrate vonmises kernels
    kde = np.exp(kappa*np.cos(bins[:, None]-data[None, :])).sum(1)/(2*np.pi*i0(kappa))
    kde /= np.trapz(kde, x=bins)
    return bins, kde

class LCOT():
  def __init__(self,x=None):
    if x is None:
      self.x = np.linspace(0,1,1001)[:-1]
    else:
      self.x = x
    self.N = len(self.x)
    self.reference = measure([self.x,np.ones_like(self.x)])
    self.samples = np.linspace(0,1,5000)

  def forward(self,measure):
      mean = measure.expected_value()
      alpha = mean-.5
      xnew=np.linspace(-1,2,3*self.N)
      embedd = np.interp(self.x-alpha,measure.ecdf(xnew),xnew)-self.x
      return embedd

  def inverse_kde(self,embedding,kappa=50.):
      monge = embedding+self.x
      ysamples = np.interp(self.samples,self.x,monge)
      ysamples[ysamples>1]-=1
      ysamples[ysamples<0]+=1
      _,pde = vonmises_kde(2*np.pi*(ysamples-.5),kappa=kappa,n_bins=len(self.x))
      return measure([self.x,pde])

  def inverse(self,embedding):
      monge = embedding+self.x
      xtemp = np.linspace(monge.min(),monge.max(),self.N)
      imonge = np.interp(xtemp,monge,self.x)
      imonge_prime = np.gradient(imonge,xtemp[1]-xtemp[0],edge_order=2)
      if monge.min()<0:
        ind= -np.argwhere(self.x>-monge.min())[0][0]
      else:
        ind= np.argwhere(self.x>monge.min())[0][0]
      return measure([self.x,np.roll(imonge_prime,ind)])

  def cost(self,nu1,nu2):
       nu1_hat = self.forward(nu1)
       nu2_hat = self.forward(nu2)
       return np.sqrt((np.minimum(abs(nu2_hat-nu1_hat),1-abs(nu2_hat-nu1_hat))**2).sum())
