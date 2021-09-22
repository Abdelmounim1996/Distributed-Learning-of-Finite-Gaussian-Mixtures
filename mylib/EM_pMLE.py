""" mathematics library """
import numpy as np
from scipy import linalg
from scipy.stats import multivariate_normal
"""machaine learning  library"""
from sklearn.cluster import KMeans
"""   Generate data library"""
from sklearn.datasets import make_blobs
""" data visualization  and graphical plotting library """
import matplotlib
import matplotlib . pyplot as plt
from jupyterthemes import jtplot
jtplot.style()
# settings
np.seterr(divide='ignore', invalid='ignore')
class EM_pMLE:
    def __init__(self  ):
        
        self.Means   = None 
        self.Covars  = None 
        self.Weights = None
        
    def fit(self , X , n_components : int , n_iters = 1000  , tol = 0.01):
        self.X            = X
        self.n_components = n_components
        self.n_iters      = n_iters
        self.tol          = tol
        self.N            = self.X.shape[0]
        self.d            = self.X.shape[1]
        self.converged    = False
        self.S_x          = np.cov(self.X , rowvar=False)
        # initializations 
        self.Means        = KMeans(n_clusters=self.n_components, random_state=0).fit(self.X).cluster_centers_
        self.Covars       = np.full((self.n_components , self.d , self.d), self.S_x)
        self.Weights      = np.array([1/self.n_components]*self.n_components)
        # start algorithm : 
        dis_likelihood_limited = [np.infty] ; dis_likelihood = [np.infty]
        for it in range(self.n_iters):
            self.likelihood_membership_vector()
            dis_likelihood_limited.append(self.penalized_likelihood_limited() )
            dis_likelihood.append(self.penalized_likelihood())
            if abs(dis_likelihood_limited[it+1]-dis_likelihood_limited[it]) <= self.tol : 
                self.converged = True 
                break
            self.E_step() 
            self.M_step()
        self.max_likelihood = dis_likelihood[1:]
        self.max_likelihood_limited = dis_likelihood_limited[1:]
        # end algorithm"""
        return self
    
    def likelihood_membership_vector(self ):
        likelihood          = np.zeros((self.N , self.n_components ))
        for k in range(self.n_components):
            likelihood[:,k] = multivariate_normal(self.Means[k],self.Covars[k] , allow_singular=True).pdf(self.X) 
        numerator           = likelihood * self.Weights
        membership_vector = (numerator  == numerator.max(axis=1)[:, None]).astype(int)
        self.likelihood = likelihood ;  self.membership_vector = membership_vector
    
    def penalized_likelihood(self):
        an =1./np.sqrt(self.N) 
        log_likelihood   = (self.membership_vector*np.log( self.likelihood*self.Weights)).sum()
        penalty_quantity = an*np.trace(self.S_x*np.linalg.pinv(self.Covars), axis1 = 1 , axis2 = 2).sum()
        return log_likelihood-penalty_quantity
    
    def penalized_likelihood_limited(self):
        an =1./np.sqrt(self.N) 
        log_likelihood = np.log( (self.likelihood*self.Weights).sum(1)).sum(0)
        penalty_quantity = an*np.trace(self.S_x*np.linalg.pinv(self.Covars), axis1 = 1 , axis2 = 2).sum()
        return log_likelihood-penalty_quantity
    
    def E_step(self) :
        weighted_likelihood          = self.Weights*self.likelihood
        self.conditional_expectation =  weighted_likelihood/( weighted_likelihood.sum(1))[:, np.newaxis]
 
    def M_step(self):
        an = 1/np.sqrt(self.N) 
        self.Weights =  (1./self.N)*self.conditional_expectation.sum(0)
        self.Means = np.dot((self.conditional_expectation.T), self.X)*self.Weights[:,np.newaxis]
        S = np.add.reduce((self.conditional_expectation.T)[:,:,np.newaxis , np.newaxis]\
        *np.asarray(list(map(lambda matrix : [ row[:, np.newaxis]*row for row in matrix ] , self.X -self.Means[:, np.newaxis]))) , axis = 1)
        self.Covars = (1./(2*an +self.N*self.Weights))[:,np.newaxis , np.newaxis]*(2*an*self.S_x + S)

    def predict(self , Test):   
        likelihood = np.zeros( (Test.shape[0] , self.n_components) )
        for i in range(self.n_components):
            likelihood[:,i] = multivariate_normal(self.Means[i],self.Covars[i] , allow_singular= True ).pdf(Test)
        numerator           = likelihood * self.Weights
        denominator         = numerator.sum(axis=1)[:, np.newaxis]
        return np.argmax(numerator / denominator , axis=1)
        
