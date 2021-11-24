import numpy as np
from scipy.stats import multivariate_normal
import time

"""machaine learning  library"""
from sklearn.cluster import KMeans
import pyspark
from pyspark.sql import SparkSession
from pyspark.rdd import RDD
from pyspark.mllib.linalg import  DenseVector , DenseMatrix , _convert_to_vector
from pyspark.mllib.common import callMLlibFunc
import array as pyarray

spark = SparkSession.builder.master("local[*]") \
                    .appName('Distributed Learning of Finite Gaussian Mixtures') \
                    .getOrCreate()
sc = spark.sparkContext

def Kullback_Leibler_Distance(G1 , G2 ):
  Means_1 , Covars_1 = G1
  Means_2 , Covars_2 = G2
  Sigma_inv = np.linalg.pinv(Covars_2)
  k,d = Means_2.shape ; m, d = Means_1.shape ;  KL_dis_gmm = np.empty((m, k))
  xdiff = Means_1-Means_2[0]
  for i in range(k):
    xdiff = Means_1-Means_2[i]
    KL_dis_gmm[:, i] = 0.5*(  np.log(np.abs( np.linalg.det(Covars_2[i])/ np.linalg.det(Covars_1)) )\
                            + np.trace(np.dot( Covars_1 , Sigma_inv[i]  ) , axis1 = 1 , axis2 = 2 )  
  +  (xdiff.dot( Sigma_inv[i])*xdiff).sum(axis=-1) -d ) 
  return np.abs(KL_dis_gmm)



  class  Gaussian_mixture_reduction_pMLE :
    
    def __init__(self , n_components : int   ):
        self.n_components = n_components
        
    def fit(self , X :RDD   , n_batch : int  , n_partition : int = None  , 
            n_iters  : int = 10 , tol : float = 0.000001 ):
        #<<_____________________start time ____________________>>#
        start = time.time()   
        self.n_batch                = n_batch
        self.n_partition            = n_partition
        self.n_iters                = n_iters
        self.tol                    = tol 
        # check parameters : 
        # ================
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
        print("processing data time : " , end -start )
        # <<_________________end time processing____________>> #

        # parallelized distributed estimator : 
        #=====================================
        n_clusters = self.n_components 
        def local_estimator(  partitioned_data  , n_components = n_clusters , init_params= None  ):
            import numpy as np
            partitioned_data= np.vstack(partitioned_data) 
            n_samples_partition = partitioned_data.shape[0]
            if  init_params is None :
              model = EM_pMLE(n_components).fit(partitioned_data )
            else :
              model = EM_pMLE(n_components).fit(partitioned_data ,  init_params)
            return   [ model.Means , model.Covars , model.Weights  *n_samples_partition ]
        
        
        # Get parametres of the weighted average distribution :
        # =====================================================
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
        print ("weighted average distribution  time:", end_AVG - start_AVG)
        start = time.time()
        self.majorization_minimization() 
        end = time.time()
        print("this is time of majorization_minimization (s): ",end-start  )
        self.Means_broadcast = sc.broadcast(tuple(map(DenseVector, self.Means))) 
        self.Covars_broadcast = sc.broadcast(tuple(map(lambda x : DenseMatrix(2,2, x) , list(map(np.ravel ,self.Covars )) )))
        self.Weights_broadcast = sc.broadcast(self.Weights.tolist()) 
        return self
    
    def majorization_minimization(self) :
        K   = self.n_components ; MK  = self.n_partition*self.n_components*self.n_batch ; d = self.d
        self.Means  = KMeans(n_clusters = self.n_components, random_state=0).fit(self.AVG_means).cluster_centers_
        self.Covars = np.full((self.n_components,d,d) , np.identity(d)) 
        self.Weights = np.array([1/self.n_components]*self.n_components)
        self.trans_divergence = [np.Infinity]
        
        for it in range(self.n_iters):
            print("iters : ", it )
            start = time.time()
            dis = Kullback_Leibler_Distance( (self.AVG_means, self.AVG_covs) , (self.Means , self.Covars)  )
            end = time.time()
            print("KL distance time :", end-start)
            dis_min = dis.min(1) ; args = dis.argmin(1) ; PI = self.AVG_weights[args]
            self.trans_divergence.append(   (PI*dis_min).sum()  )

            for ind in range(K) : 
              start = time.time()
              PI_ind = PI[args == ind] ; Beta = PI_ind/PI_ind.sum()
              self.Means[ind] = ( Beta[: , np.newaxis]*self.AVG_means[args == ind] ).sum(0)
              self.Covars[ind] = (Beta[:, np.newaxis , np.newaxis] *( self.AVG_covs[args == ind] \
              + np.einsum('ij , im ->ijm', self.AVG_means[args == ind] - self.Means[ind] ,
                          self.AVG_means[args == ind] - self.Means[ind]  )) ).sum(0)
              end = time.time()
              print("update params time :", end-start,"/k = ", ind)
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
