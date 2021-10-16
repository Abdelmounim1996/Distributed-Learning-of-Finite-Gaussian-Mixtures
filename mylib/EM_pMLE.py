""" library mathematic & statistic  """
import numpy as np
from scipy import linalg
from scipy.stats import multivariate_normal
""" library for machine learning """
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
# settings
np.seterr(divide='ignore', invalid='ignore')

class EM_pMLE:
    def __init__(self  ):
        
        self.Means   = None 
        self.Covars  = None 
        self.Weights = None
        
    def fit(self , X , n_components : int , n_iters = 500 , tol = 0.01):
        self.X            = X
        self.n_components = n_components
        self.n_iters      = n_iters
        self.tol          = tol
        self.N , self.d   = self.X.shape
        self.converged    = False
        self.S_x          = np.cov(self.X , rowvar=False)
        # initializations 
        self.Means        = KMeans( n_clusters = self.n_components, random_state=0).fit(self.X).cluster_centers_
        self.Covars       = np.full((self.n_components , self.d , self.d),np.identity(self.d) ) 
        self.Weights      = np.full( shape=self.n_components , fill_value=1./self.n_components)
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
        # end algorithm"""
        return self

    def likelihood (self , data  ):
        self.likelihood_weighted = np.asarray(
        [  multivariate_normal(mu , sigma , allow_singular = True).pdf( data) for mu , sigma in zip(self.Means , self.Covars) ])* self.Weights[: , np.newaxis]

        
    def penalized_likelihood(self):
        an =1./np.sqrt(self.N) 
        log_likelihood_EM_pLME  = (np.log( self.likelihood_weighted)[(self.likelihood_weighted ==self.likelihood_weighted.max(0))]).sum()
        log_likelihood_EM       = np.log(     self.likelihood_weighted.sum(0)       ).sum() 
        penalty_quantity        = (an*( np.trace(self.S_x*np.linalg.pinv(self.Covars), axis1 = 1 , axis2 = 2) + np.log(np.linalg.det(self.Covars)) ) ).sum()
        return log_likelihood_EM_pLME - penalty_quantity , log_likelihood_EM-penalty_quantity

    def E_step(self) :
        self.conditional_expectation =  self.likelihood_weighted  / self.likelihood_weighted .sum(0) ; self.likelihood_weighted = None 

    def M_step(self):
        an = 1/np.sqrt(self.N) 
        self.Weights =  (1./self.N)*self.conditional_expectation.sum(1)
        self.Means = np.dot(self.conditional_expectation , self.X)*(  1/(self.N*self.Weights))[:,np.newaxis]
        S = np.add.reduce((self.conditional_expectation)[:,:,np.newaxis , np.newaxis] * \
                          np.apply_along_axis(lambda x : x[:, np.newaxis]*x, 2,  self.X -self.Means[:, np.newaxis]  ) , axis = 1 )
        self.conditional_expectation = None 
        self.Covars = (1./(2*an + self.N*self.Weights))[:,np.newaxis , np.newaxis]*(2*an*self.S_x + S)
  
    def predict(self , data ):   
        self.likelihood( data )
        return self.likelihood_weighted .argmax(0)
    
