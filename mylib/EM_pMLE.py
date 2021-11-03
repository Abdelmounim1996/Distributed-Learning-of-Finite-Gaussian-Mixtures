# library mathematic & statistic 
import numpy as np
from scipy.stats import multivariate_normal
from scipy import linalg
from numpy.linalg import multi_dot
#from sklearn.mixture.base import BaseMixture, _check_shape
from sklearn.utils import check_array, check_random_state
# library for machine learning
from sklearn.cluster import KMeans
np.seterr(divide='ignore', invalid='ignore')


class EM_pMLE:
    def __init__(self ,  n_components  ):
        self.n_components = n_components

    def fit(self , X  : int , n_iters = 500 , tol = 0.01 , init_params = None , reg_covar=1e-6 ):
        self.X            = X
        self.n_iters      = n_iters
        self.tol          = tol
        self.init_params  = init_params
        self.N , self.d   = self.X.shape
        self.converged    = False
        self.S_x          = np.cov(self.X , rowvar=False)
        self.Means   = None 
        self.Covars  = None 
        self.Weights = None
        self. reg_covar =  reg_covar
        
        
        # initializations 
        if  self.init_params is  None:
          start = time.time()
          self.Means        = KMeans( n_clusters = self.n_components, n_init=1 , random_state=0).fit(self.X).cluster_centers_
          self.Covars       = np.full((self.n_components , self.d , self.d),np.identity(self.d) ) 
          self.Weights      = np.full( shape=self.n_components , fill_value=1./self.n_components)
          end = time.time()
          print("kmeans sklearn time : ", end - start)
        else :
          self.Means   = self.init_params[0]
          self.Covars  = self.init_params[1]
          self.Weights = self.init_params[2]
        
        # start algorithm : 
        dis_likelihood = [[np.infty , np.infty] ]
        for it in range(self.n_iters):
            self.likelihood ( self.X  )
            dis_likelihood.append(self.penalized_likelihood())
            if abs(dis_likelihood[it+1][1]-dis_likelihood[it][1]) <= self.tol : 
                self.converged = True
                self.max_iters = it 
                break
            self.E_step() 
            self.M_step()
        self.max_likelihood = dis_likelihood[1:]
        # end algorithm
        return self

    def likelihood (self , data  ):
        self.likelihood_weighted = np.asarray(
        [  multivariate_normal(mu , sigma , allow_singular = True).pdf( data) for mu , sigma in zip(self.Means ,
                                                                   self.Covars) ])* self.Weights[: , np.newaxis]

    def penalized_likelihood(self):
        an =1./np.sqrt(self.N) 
        log_likelihood_EM_pLME  = (np.log( self.likelihood_weighted)[(self.likelihood_weighted ==self.likelihood_weighted.max(0))]).sum()
        log_likelihood_EM       = np.log(     self.likelihood_weighted.sum(0)       ).sum() 
        penalty_quantity        = (an*( np.trace(self.S_x*np.linalg.pinv(self.Covars), axis1 = 1 , axis2 = 2) \
                                       + np.log(np.linalg.det(self.Covars)) ) ).sum()
        return log_likelihood_EM_pLME - penalty_quantity , log_likelihood_EM-penalty_quantity

    def E_step(self) :
        self.conditional_expectation =  self.likelihood_weighted  / self.likelihood_weighted .sum(0) 
        self.likelihood_weighted = None 

    def M_step(self):
        an = 1/np.sqrt(self.N) 
        self.Weights =  (1./self.N)*self.conditional_expectation.sum(1)
        self.Means = np.dot(self.conditional_expectation , self.X)*(  1/(self.N*self.Weights))[:,np.newaxis]
        
        Mu = self.X -self.Means[:, np.newaxis]
        S = np.add.reduce((self.conditional_expectation)[:,:, np.newaxis , np.newaxis] * \
                        (np.repeat( Mu, self.d , axis = 1 ).reshape(-1,  self.d )*\
                         Mu.ravel()[: , np.newaxis] ).reshape(self.n_components ,-1 ,self.d  , self.d )
                         , axis = 1 )
        self.conditional_expectation = None 
        self.Covars = (1./(2*an + self.N*self.Weights))[:,np.newaxis , np.newaxis]*(2*an*self.S_x + S)
        self.Covars.reshape(self.n_components , -1)[:, ::self.d+1]+=self.reg_covar
  
    def predict(self , data ):   
        self.likelihood( data )
        return self.likelihood_weighted .argmax(0)
