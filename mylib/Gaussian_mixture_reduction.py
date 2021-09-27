"""pyspark library """
import pyspark
from pyspark.sql import SparkSession
from pyspark.rdd import RDD
from pyspark.mllib.linalg import  DenseVector , DenseMatrix , _convert_to_vector
from pyspark.mllib.common import callMLlibFunc
import array as pyarray
""" mathematics library """
import numpy as np
from scipy.stats import multivariate_normal
import time
""" My library"""
from mylib.EM_pMLE import EM_pMLE
"""machaine learning  library"""
from sklearn.cluster import KMeans

spark = SparkSession.builder.master("local[*]") \
                    .appName('Distributed Learning of Finite Gaussian Mixtures') \
                    .getOrCreate()
sc = spark.sparkContext
sqlContext = pyspark.SQLContext(sc)

class  Gaussian_mixture_reduction :
    
    def __init__(self  ):
        """
                                 Gaussian_mixture_reduction 
                                 ==========================
                    aggregate the local estimates via the reduction estimator
        Parametres  : 
        =============
        X              : distributed training data (format RDD)
        n_components   : The number of mixture components of local Gaussians mixture estimators 
        n_partition    : Default to 4 The number of local Gaussians mixture estimators & The maximum number of partitions
                         that can be used for parallelism in table reading and writing
        n_iters        : Number of iterations. Default to 10000
        tol.           : tolerance for  convergence. Default to 0.000001
        
        Attributes :
        ============
                       ____________________________________________________________
                       |  Estimate means of the weighted average distribution      |
                       |___________________________________________________________|
        params : {     | Estimate covariances of the weighted average distribution |
                       |___________________________________________________________| 
                       | Estimate weights of the weighted average distribution     |
                       |___________________________________________________________|
                       
        n_samples    : Number of samples of training data
        d            : n_features of training data

        """
    def fit(self , X  , n_components : int  , n_partition = 4  , n_iters  : int = 100 ,tol : float = 0.000001 ):
        self.n_components           = n_components
        self.X                      = X 
        self.n_partition            = n_partition
        self.n_iters                = n_iters
        self.tol                    = tol 
        self.n_samples              = self.X.count()
        self.d                      = X.first().size
        # check parameters
        if self.n_partition < 1 : raise ValueError('Invalid value for n_partition : %s' % self.n_partition)
        if isinstance(self.X, RDD):
            if self.X.getNumPartitions() != self.n_partition :
                self.X.repartition(self.n_partition) 
                self.N = self.X.count()
                self.d = self.X.first().size    
        else : raise TypeError("data should be a RDD, received %s" % type(sample)) 
            
        if self.n_iters < 1: raise ValueError('estimation requires at least one run')
        if self.tol < 0.: raise ValueError('Invalid value for covariance_type: %s' %tol)
        n_samples      = self.n_samples ; n_components   = self.n_components
        # parallelized estimator : 
        def local_estimators(  partitioned_data  , n_components = n_components, n_samples = n_samples ):
            partitioned_data= np.vstack(partitioned_data) 
            n_samples_partition = partitioned_data.shape[0]
            model = EM_pMLE().fit(partitioned_data , n_components)
            weighted_average = n_samples_partition/n_samples 
            weights = model.Weights  *weighted_average
            return   [ model.Means , model.Covars , weights ]
        
        def add(a, b):
            a.extend(b)
            return a
        start = time.time()
        # parametres of the weighted average distribution 
        self.map_params =(self.X).mapPartitions(local_estimators).mapPartitions(lambda lst :[(t,lst[t]) for t in range(len(lst)) ] )\
            .combineByKey(lambda x : [x] , lambda x,y : x.append(y), add ).mapValues(np.vstack ).collectAsMap() 
        self.map_means = self.map_params[0]
        self.map_covs = self.map_params[1]
        self.map_weights = self.map_params[2].flatten()
        end = time.time()
        print ("weighted average distribution  time:", end - start)
        start = time.time()
        self.majorization_minimization() 
        end = time.time()
        self.Means_broadcast = sc.broadcast(tuple(map(DenseVector, self.Means))) 
        self.Covars_broadcast = sc.broadcast(tuple(map(lambda x : DenseMatrix(2,2, x) , list(map(np.ravel ,self.Covars )) )))
        self.Weights_broadcast = sc.broadcast(self.Weights.tolist()) 
        print (" majorization_minimization time:", end - start)
        return self
    
    def Gaussian_distance(self , Gaussian_1 , Gaussian_2 , which="W2"):
        mu1, Sigma1 = Gaussian_1 ; mu2, Sigma2 = Gaussian_2
        if which == "KL":
            m0, S0 = Gaussian_1 ;  m1, S1 = Gaussian_2
            N = m0.shape[0] ; iS1 = np.linalg.pinv(S1) ; diff = m1 - m0
            tr_term   = np.trace(np.dot(iS1 , S0)) ; det_term  = np.log(np.linalg.det(S1)/np.linalg.det(S0))
            quad_term = np.dot(np.dot(diff.T , np.linalg.pinv(S1) ), diff) 
            return .5 * (tr_term + det_term + quad_term - N)
        elif which == "W2":
            if mu1.shape[0] == 1 or mu2.shape[0] == 1: # 1 dimensional
                W2_squared = (mu1 - mu2)**2 + (np.sqrt(Sigma1) - np.sqrt(Sigma2))**2
                W2_squared = np.asscalar(W2_squared)
            else: # multi-dimensional
                sqrt_Sigma1 = linalg.sqrtm(Sigma1)
                Sigma = Sigma1 + Sigma2 - 2 * linalg.sqrtm(sqrt_Sigma1 @ Sigma2 @ sqrt_Sigma1)
                W2_squared = np.linalg.norm(mu1 - mu2)**2 + np.trace(Sigma) + 1e-13
                return np.sqrt(W2_squared)
        else:
            raise ValueError("This ground distance is not implemented!")
        
    def Joint_Distribution(self):
        K   = self.n_components ;   MK  = self.n_partition*self.n_components ; dis = np.zeros((MK,K)) ; PI  = np.zeros_like(dis) 
        for i in range(MK):
            dis[i , :]= np.array(list(map(lambda aggr_params : np.absolute(self.Gaussian_distance([self.map_means[i], 
                       self.map_covs[i]], aggr_params , which="KL")) , zip(self.Means, self.Covars))))
            PI[i,np.argmin(dis[i, :], axis=0)] = self.map_weights[i]
        distance = (PI *dis).sum()
        return  PI , distance
    
    def majorization_minimization(self) :
        K   = self.n_components ; MK  = self.n_partition*self.n_components ; d = self.d
        self.Means  = KMeans(n_clusters = self.n_components, random_state=0).fit(self.map_means).cluster_centers_
        self.Covars = np.full((self.n_components,d,d) , np.identity(d)) 
        for it in range(self.n_iters):
            PI , dis = self.Joint_Distribution() 
            PI_mu= (PI/PI.sum(0))
            for k in range(self.n_components):
                self.Means[k]  = np.dot(PI_mu[:,k] , self.map_means)
                self.Covars[k] = np.sum(PI_mu[:,k][:, np.newaxis , np.newaxis]*self.map_covs, axis =0)
                + np.sum(PI_mu[:,k][:, np.newaxis ,np.newaxis ] * np.array(list(map(lambda row : row[:,np.newaxis]*row, 
                                                self.map_means-self.Means[k]))) ,axis = 0)
        self.Weights = PI.sum(0)
        return self.Means , self.Covars ,  self.Weights
    
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
            membership_matrix = callMLlibFunc("predictSoftGMM", x.map(_convert_to_vector), _convert_to_vector(weights), means, sigmas)
            return membership_matrix.map(lambda x: pyarray.array('d', x))
        else:
            raise TypeError("data should be a RDD, received %s" % type(x)) 
            
    def predict(self, x):
        if isinstance(x, RDD):
            cluster_labels = self.predictSoft(x).map(lambda z: z.index(max(z)))
            return cluster_labels.collect()
        else:
            raise TypeError("data should be a RDD, received %s" % type(x)) 




















