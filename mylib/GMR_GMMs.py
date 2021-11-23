import numpy as np
from scipy.stats import multivariate_normal
import time
from scipy import linalg
"""machaine learning  library"""
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import pyspark
from pyspark.sql import SparkSession
from pyspark.rdd import RDD
from pyspark.mllib.linalg import  DenseVector , DenseMatrix , _convert_to_vector
from pyspark.mllib.common import callMLlibFunc
import array as pyarray
# library mathematic & statistic 
import numpy as np
import time 
from scipy.stats import multivariate_normal
from scipy import linalg
from numpy.linalg import multi_dot
#from sklearn.mixture.base import BaseMixture, _check_shape
from sklearn.utils import check_array, check_random_state
# library for machine learning
from sklearn.cluster import KMeans
np.seterr(divide='ignore', invalid='ignore')
spark = SparkSession.builder.master("local[*]") \
                    .appName('Distributed Learning of Finite Gaussian Mixtures') \
                    .getOrCreate()
sc = spark.sparkContext

def  Cholesky(Covars ):
  n_components, n_features, n_features = Covars.shape
  cov_chol = np.empty((n_components, n_features, n_features))
  det = []
  for k, covariance in enumerate(Covars):
    try :
      chol = linalg.cholesky(covariance, lower=True)
      det.append(np.prod(chol.diagonal() ))
      cov_chol[k] = linalg.solve_triangular(chol, np.eye(n_features),lower=True).T
    except linalg.LinAlgError:
      cov_chol[k] = linalg.inv(covariance)
      det.append( linalg.det(covariance) )
  return cov_chol , np.array(det)

def  det_Cholesky(Covars ):
  det = []
  for covariance in Covars :
    try :
      chol = linalg.cholesky(covariance, lower=True)
      det.append(np.prod(chol.diagonal() ))
    except linalg.LinAlgError:
      det.append( linalg.det(covariance) )
  return np.array(det)

def Kullback_Leibler_Distance( AVG_params  , reduce_params ):
  AVG_Means , AVG_Covars_Inv , AVG_det = AVG_params 
  reduce_Means , reduce_Covars = reduce_params
  reduce_det = det_Cholesky(reduce_Covars  )
  reduce_clusters , n_features = reduce_Means.shape ; AVG_clusters , n_features = AVG_Means.shape ; dis = []
  for k in range(AVG_clusters):
    Means_sub = reduce_Means-AVG_Means[k]
    dis.append( 0.5*(  np.log(np.abs( AVG_det[k]/ reduce_det ) ) 
    + np.trace(np.dot( reduce_Covars , AVG_Covars_Inv[k]  ) , axis1 = 1 , axis2 = 2 )  
    + np.einsum('ij,ik,kj->i', Means_sub , Means_sub, AVG_Covars_Inv[k]) - n_features ) )
  return np.array(dis)

def Kullback_Leibler_Distance_init( AVG_params  , reduce_Means ):
    AVG_Means , AVG_Covars_Inv , AVG_det = AVG_params 
    reduce_clusters , n_features = reduce_Means.shape ; AVG_clusters , n_features = AVG_Means.shape ; dis = []
    for k in range(AVG_clusters):
      Means_sub = reduce_Means-AVG_Means[k]
      tr = np.trace( AVG_Covars_Inv[k])
      dis.append( 0.5*(  np.log(np.abs( np.full(reduce_clusters , AVG_det[k] ) ) )
      + np.full(reduce_clusters , tr)  +  np.power(Means_sub , 2).sum(1) - n_features  ))
    return np.array(dis)
    
class  Gaussian_mixture_reduction_GMMs :
    def __init__(self , n_components : int   ):
        self.n_components = n_components
        """
        parametres : 
        ===========
        n_components : The number of aggregated mixture components
        X                 : data to training format RDD (Resilient Distributed Dataset )
        n_iters           :  int, defaults to 100. The number of EM iterations to perform.
        tol.              :  float, defaults to 1e-3. The convergence threshold. MM agregation iterations will stop when the
                             lower bound average gain is below this threshold.
        n_batch           : The batch size is a hyperparameter that defines the number of samples 
        n_partition       : divided data under N partition (data distributed )
        Parallel_Estimator: take EM_pMLE or GMM estimator params for each partition 
  
        """
        
    def fit(self , X :RDD   , n_batch : int  , n_partition : int = None  , 
            n_iters  : int = 100 , tol : float = 0.000001 ):
      
        start = time.time()   
        self.n_batch                = n_batch
        self.n_partition            = n_partition
        self.n_iters                = n_iters
        self.tol                    = tol 

        if isinstance(X, RDD):
            X.cache()
            self.N = X.count()
            ls = X.first()
            if isinstance(ls  , list ):
              self.d = len(X.first())
            else :
              self.d = X.first().size
            if self.n_partition is None :
              self.n_partition = X.getNumPartitions()
            elif isinstance(self.n_partition , int)  :
              if self.n_partition  < X.getNumPartitions() and  self.n_partition >= 1 :
                X.repartition(self.n_partition) 
              elif X.getNumPartitions() < self.n_partition :
                X.coalesce(self.n_partition)
            else : 
              raise ValueError('Invalid value for n_partition : %s' % self.n_partition) 
        else : 
          raise TypeError("data should be a RDD, received %s" % type(X)) 

        if self.n_iters < 1: 
          raise ValueError('estimation requires at least one run')
        if self.tol < 0.:
           raise ValueError('Invalid value for covariance_type: %s' %tol)

        end = time.time()
        print("sys time : " , end -start )
        n_clusters = self.n_components 
        def local_estimator(  partitioned_data  , n_components =  n_clusters , init_params= None ):
          import numpy as np
          from sklearn.mixture import GaussianMixture 
          partitioned_data= np.vstack(partitioned_data) 
          n_samples_partition = partitioned_data.shape[0]
          model = GaussianMixture(n_components = n_components , random_state=0).fit( partitioned_data  ) 
          return   [ model.means_  , model.covariances_ , model.weights_  *n_samples_partition ]
        start_AVG = time.time()
        list_rdd = X.randomSplit([1/self.n_batch]*self.n_batch) ; X.unpersist()
        lst_means = [] ; lst_covs = [] ; lst_weights = [] 
        for i  in range(self.n_batch) :
          start = time.time()
          list_rdd[i].cache()
          zip_params =(list_rdd[i]).mapPartitions(local_estimator).collect() ; list_rdd[i].unpersist() ; p=len(zip_params)
          lst_means.append([zip_params[i] for i in range( 0, p , 3)])
          lst_covs.append([zip_params[i] for i in range( 1, p  , 3)])
          lst_weights.append([zip_params[i] for i in range( 2, p , 3)])

          end = time.time()
          print('---->batch=%.1d, time(s)=%.1f' % ( i , end - start))
        self.AVG_means = np.array(lst_means).reshape(-1 , self.d)
        self.AVG_covs = np.array(lst_covs).reshape(-1 , self.d , self.d)
        self.AVG_weights = np.array(lst_weights).flatten()/self.N
        end_AVG = time.time()
        print ("getting distributed parametres in :", end_AVG - start_AVG, "seconds")
        start = time.time()
        self.majorization_minimization() 
        end = time.time()
        print("getting aggregate parameters in : : ",end-start , "seconds " )

        self.Means_broadcast = sc.broadcast(tuple(map(DenseVector, self.Means))) 
        self.Covars_broadcast = sc.broadcast(tuple(map(lambda x : DenseMatrix(2,2, x) , list(map(np.ravel ,self.Covars )) )))
        self.Weights_broadcast = sc.broadcast(self.Weights.tolist()) 
        return self
    
    def majorization_minimization(self  ) :
        K   = self.n_components ; MK  = self.n_partition*self.n_components*self.n_batch ; d = self.d
        self.Means  = KMeans(n_clusters = self.n_components, random_state=0).fit(self.AVG_means).cluster_centers_
        self.Covars = np.full((self.n_components,d,d) , np.identity(d)) 
        self.Weights = np.array([1/self.n_components]*self.n_components)
        self.trans_divergence = [np.Infinity]
        AVG_Inv  , AVG_det = Cholesky( self.AVG_covs ) ; dis = 0
        
        for it in range(self.n_iters):
            print("iters : ", it )
            start = time.time()
            if it == 0 :
              dis = Kullback_Leibler_Distance_init((self.AVG_means,  AVG_Inv , AVG_det ) , self.Means)
            else :
              dis = Kullback_Leibler_Distance( (self.AVG_means,  AVG_Inv , AVG_det ) , (self.Means , self.Covars)  )
            end = time.time()
            print("KL distance time :", end-start,"seconds ")
            
            start = time.time()
            dis_min = dis.min(1) ; args = dis.argmin(1) ; PI = self.AVG_weights[args]
            self.trans_divergence.append(   (PI*dis_min).sum()  )

            for ind in range(K) : 
              PI_ind = PI[args == ind] ; Beta = PI_ind/PI_ind.sum()
              self.Means[ind] = ( Beta[: , np.newaxis]*self.AVG_means[args == ind] ).sum(0)
              self.Covars[ind] = (Beta[:, np.newaxis , np.newaxis] *( self.AVG_covs[args == ind] \
              + np.einsum('ij , im ->ijm', self.AVG_means[args == ind] - self.Means[ind] ,
                          self.AVG_means[args == ind] - self.Means[ind]  )) ).sum(0)
            end = time.time()
            print("update params MM algorithm time :", end-start,"seconds ")
            if ( np.absolute( self.trans_divergence[it+1]-self.trans_divergence[it]) <= self.tol or it == self.n_iters-1 ):
              for ind in range(K):
                self.Weights[ind] = PI[args == ind].sum()
              self.max_iters = it 
              break 
    
    def score(self , data ):
        Means = self.Means ; Covars = self.Covars ; Weights = self.Weights ; n_components = self.n_components
        def local_score( data , Means = Means , Covars = Covars , Weights = Weights , n_components = n_components):
            data = np.vstack(data)
            if data.ndim == 1: data = data[:, np.newaxis]
            if data.size == 0: return np.array([]), np.empty((0, n_components))
            if data.shape[1] != Means.shape[1]: raise ValueError('The shape of x is not compatible with self')
            min_covar=1.e-7 ; n_samples, n_dim = data.shape ; nmix = n_components ; log_prob = np.empty((n_samples, nmix))
            for c, (mu, cv) in enumerate(zip(Means, Covars)):
                try: cv_chol = linalg.cholesky(cv, lower=True)
                except linalg.LinAlgError:
                    try: cv_chol = linalg.cholesky(cv + min_covar * np.eye(n_dim),lower=True)
                    except linalg.LinAlgError: raise ValueError("'covars' must be symmetric, ""positive-definite")
                cv_log_det = 2 * np.sum(np.log(np.diagonal(cv_chol)))
                cv_sol = linalg.solve_triangular(cv_chol, (data - mu).T, lower=True).T
                log_prob[:, c] = - .5 * (np.sum(cv_sol ** 2, axis=1) + n_dim * np.log(2 * np.pi) + cv_log_det)
            log_prob +=np.log(Weights) ; log_prob = logsumexp(log_prob, axis=1)
            return log_prob
        return data.mapPartitions(local_score).collect()
        
    def predictSoft(self, x):
        weights =self.Weights_broadcast.value  ; means = self.Means_broadcast.value ; sigmas =self.Covars_broadcast.value
        if isinstance(x, RDD):
            membership= callMLlibFunc("predictSoftGMM", x.map(_convert_to_vector), _convert_to_vector(weights), means, sigmas)
            return membership.map(lambda x: pyarray.array('d', x))
        else:
            raise TypeError("data should be a RDD, received %s" % type(x)) 
            
    def predict(self, x):
        if isinstance(x, RDD):
            cluster_labels = self.predictSoft(x).map(lambda z: z.index(max(z)))
            return cluster_labels.collect()
        else:
            raise TypeError("data should be a RDD, received %s" % type(x))
