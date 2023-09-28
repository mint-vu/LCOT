import numpy as np

#@title # LCOT Codes
class measure():
  '''
    Measure class with methods to compute densities at any point, cdfs, extended cdfs, inverse cdfs and expected value.
  '''

  def __init__(self,density):
    '''
        Input:
          Density: 2xN dimensional array of discretized density function.
            density[0] descretization of the domain
            density[1] values of the density at discretized domain
    '''
    self.density_x = density[0].copy()
    self.density_y = density[1].copy()
    self.density_len = len(self.density_x)

  def pdf(self,x):
    # probability density function at x (Computed by interpolating the discretized density)
    return np.interp(x,self.density_x,self.density_y)

  def cdf(self,x):
    # cumulative distribution funcion for x in [0,1]
    cdf_x = self.density_x
    cdf_y = np.cumsum(self.density_y)/self.density_len
    return np.interp(x,cdf_x,cdf_y)

  def ecdf(self,x):
    # extended cdf to the real line as in 'Transportation distances on the circle and applications' - Rabin et al.
    int_x = np.floor(x)
    rest_x = x-int_x
    return int_x + self.cdf(rest_x)

  def tcdf(self,x,x0):
    # translated cdf, that is F_{x0} as in 'Transportation distances on the circle and applications' - Rabin et al.
    return self.ecdf(x+x0)- self.ecdf(x0)

  def itcdf(self,y,x0):
    # inverse of translated cdf
    domain = np.linspace(-1,2,3*self.density_len)
    return np.interp(y,self.tcdf(domain,x0),domain)

  def expected_value(self):
    # expected value
    return np.mean(self.density_x*self.density_y)

  def sample(self,sample_size):
    number_different_values = 10000
    extended_samples = np.linspace(0,1,number_different_values)
    extended_samples_pdf = self.pdf(extended_samples)
    extended_samples_pdf = extended_samples_pdf/np.cumsum(extended_samples_pdf)[-1]
    #return np.random.choice(self.density_x, size=sample_size, p=self.density_y/self.density_len)
    return np.random.choice(extended_samples, size=sample_size, p=extended_samples_pdf)


class target_measure(measure):
  '''
    Target measure class with same methods as measure class plus the method to compute alpha for uniform reference and quadratic cost
  '''
  def __init__(self, density):
    super().__init__(density)

  def alpha(self):
    # returns alpha as in 'Transportation distances on the circle and applications' - Rabin et al. for quadratic transport cost
    return self.expected_value() - 0.5

  def get_cut(self):
    max_it = 50
    t = 0.5
    alpha = self.alpha()
    TOL = 1e-9
    for i in range(max_it):
      old_t = t
      t = t-(t-self.cdf(t)-alpha)/(1-self.pdf(t))
      if abs(t-old_t) < TOL:
        break
    return t

  def old_embedding(self,x):
    x_min = self.get_cut()
    return self.itcdf(x-x_min,x_min) - (x-x_min)

  def embedding(self,x):
    alpha = self.alpha()
    return self.itcdf(x-alpha,0) - x



class empirical_measure():

  def __init__(self, samples):
    self.samples = samples
    self.sorted_samples = np.sort(self.samples)

  def empirical_cdf(self):
    # Returns samples in order and cumulutative probs at those points, to plot the cdf do plt.plot(sorted_samples,cumulative_probs)
    sorted_samples = np.sort(self.samples)
    n = len(sorted_samples)
    cumulative_probs = np.arange(1, n + 1) / n

    return sorted_samples, cumulative_probs

  def expected_value(self):
      return np.mean(self.samples)

  def ecdf(self,x):
    # extended cdf to the real line as in 'Transportation distances on the circle and applications' - Rabin et al.
    int_x = np.floor(x)
    rest_x = x-int_x
    xs, ys = self.empirical_cdf()
    return int_x + np.interp(rest_x,xs,ys)

  def alpha(self):
    return np.mean(self.samples) - 0.5

  def get_cut(self, max_it=100, tol=1e-5):
    target = self.sorted_samples
    a = 0
    b = len(target)-1

    while self._obj_func(a) * self._obj_func(b) > 0:
      b -= 1

    for i in range(max_it):
      c = math.floor((a+b)/2)
      if self._obj_func(a)*self._obj_func(c) < 0:
        b = c
      elif self._obj_func(c)*self._obj_func(b) < 0:
        a = c
      if b-a <= 1:
        break

    y1 = self._obj_func(a)
    y2 = self._obj_func(b)
    try:
      slope = (y1-y2)/(target[a]-target[b])
      constant_term = y1-slope*target[a]
      out = - constant_term/slope
    except:
      out = (target[a]+target[b])/2
    return out

  def _obj_func(self, index):
    alpha = self.alpha()
    target,target_cdf = self.empirical_cdf()
    return target[index]-target_cdf[index]-alpha

  def embedding(self,x):
    x_cut = self.get_cut()%1
    #print(x_cut)
    target = self.sorted_samples
    num_samples = len(target)

    first_element = bisect.bisect_left(target, x_cut)  #first index in "target" that is greater than x_cut

    aux1 = target[first_element:]
    aux2 = target[:first_element]

    embedding = np.concatenate((aux1,aux2), axis=None)

    out = []
    for i in x:
      value = embedding[int(((i-x_cut)%1)//(1/(num_samples-1)))] -i
      if value <= 0.5:
        out.append(value)
      else:
        out.append(value-1)

    out2 = []
    for i in x:
      value = embedding[int(((i-x_cut)%1)//(1/(num_samples-1)))]
      if value <= 1:
        out2.append(value)
      else:
        out2.append(value-1)

    return np.array(out)


def ot_1d(u_values, v_values):
    u_sorted = np.sort(u_values)
    v_sorted = np.sort(v_values)
    u_pdf = np.ones_like(u_values) / len(u_values)
    v_pdf = np.ones_like(v_values) / len(v_values)

    u_cdf = np.cumsum(u_pdf)
    v_cdf = np.cumsum(v_pdf)

    m = min(len(u_values), len(v_values))

    z = np.linspace(1/m, 1, m)
    u_interp = np.interp(z, u_sorted, u_cdf)
    v_interp = np.interp(z, v_sorted, v_cdf)


    cost = np.mean((u_interp - v_interp) ** 2)**0.5
    return cost
