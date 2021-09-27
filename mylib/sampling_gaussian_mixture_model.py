# mathematical libraries
import numpy as np
import random
from scipy import linalg
import itertools as it
# machine learning library 
from sklearn import mixture
from sklearn.decomposition import PCA
# plotting library
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import axes3d
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as patches
import matplotlib.cm as cmx
from distinctipy import distinctipy
from bokeh.io import push_notebook, show, output_notebook
from bokeh.plotting import figure
from bokeh.layouts import column
from bokeh.models import PrintfTickFormatter
from ipywidgets import interact, IntSlider, Dropdown
from ipywidgets import FloatSlider
from bokeh.core.properties import Instance, String
from bokeh.io import show
from bokeh.models import ColumnDataSource, LayoutDOM
from bokeh.util.compiler import TypeScript
# Errors and Exceptions library 
import warnings
# jupyter plotting sombre 
from jupyterthemes import jtplot
jtplot.style()
# util functions for plotting 
def gmm_ellipse_2d( Means, Covars, sdwidth=1, points=100):
    """ vectors corresponding to ellipsoid at standard deviation sdwidth """
    Means =Means[: , np.newaxis]
    z = np.linspace(0, 2*np.pi, points)[:, np.newaxis]
    x = np.cos(z); y=np.sin(z);
    cercle_axis = np.concatenate((x,y), axis=1).T
    d, v = np.linalg.eig(Covars) ; d = sdwidth * np.sqrt(np.diag(d))                                 
    Vectors = np.dot(v, np.dot(d, cercle_axis )) + np.tile(Means, (1, points)) 
    return Vectors[0,:], Vectors[1,:] 

def plot_sphere(c=[0,0,0] , r=[1, 1, 1], w=0 , subdev=10, ax=None , sigma_multiplier=3):
    if ax is None:
        fig = plt.figure() ; ax = fig.add_subplot(111, projection='3d')
    pi = np.pi ; cos = np.cos ; sin = np.sin
    phi, theta = np.mgrid[0.0:pi:complex(0,subdev), 0.0:2.0 * pi:complex(0,subdev)]
    x = sigma_multiplier*r[0]* sin(phi) * cos(theta) + c[0]
    y = sigma_multiplier*r[1] * sin(phi) * sin(theta) + c[1]
    z = sigma_multiplier*r[2] * cos(phi) + c[2]
    cmap = cmx.ScalarMappable() ; cmap.set_cmap('jet') ; c = cmap.to_rgba(w)
    ax.plot_surface(x, y, z, color=c, alpha=0.2, linewidth=1)
    return ax

def gmm_sphere_3d( data, labels , Means , Covars , Weights  , export=True):
    n_gaussians =  Means.shape[1] ; N = int(np.round(data.shape[0] / n_gaussians))
    fig = plt.figure(figsize=(8, 8)) ; axes_1 = fig.add_subplot(111, projection='3d') 
    plt.set_cmap('Set1') ; colors = cmx.Set1(np.linspace(0, 1, n_gaussians))
    for i in range(n_gaussians):
        idx = range(i * N, (i + 1) * N)
        axes_1.scatter(data[idx, 0], data[idx, 1], data[idx, 2], alpha=0.3, c=colors[i])
        plot_sphere( c=Means[:, i], r=Covars[:, i], w=Weights[i] , ax=axes_1)
    plt.title('3D GMM')
    axes_1.set_xlabel('X')
    axes_1.set_ylabel('Y')
    axes_1.set_zlabel('Z')
    axes_1.view_init(35.246, 45)
    plt.show()
    
class sampling_gaussian_mixture_model :
    
    def __init__(self, n_components  , dim ):
        self.n_components = n_components ; self.dim          = dim 
        
    def fit( self , samples : int , Adjusting_weights = 'equal_weights' , edge = 2 , upper = 100, 
                                        n_points = 100 , epsilon_noise = 10. ,draw_ellipse = True):
        self.samples                = samples
        self.Adjusting_weights      = Adjusting_weights
        self.edge                   = edge
        self.upper                  = upper              
        self.n_points               = n_points                  
        self.epsilon_noise          = epsilon_noise    
        self.X                      = None
        self.labels                 = None
        self.Params_Generator()
        EPS = 1E-15
        if np.abs(1-np.sum(self.Weights)) > EPS: raise ValueError( "The sum of mc must be 1.0")
        cs_mc = np.concatenate(([0], np.cumsum(self.Weights)))
        sel_idx = np.random.rand(self.samples)
        X = np.zeros((self.samples, self.dim)) ; labels= np.zeros((self.samples, ))
        for k in range(self.n_components):
            idx = (sel_idx >= cs_mc[k]) * (sel_idx < cs_mc[k+1])
            ksamples = np.sum(idx)
            drawn_samples = np.random.multivariate_normal(self.Means[k], self.Covars[k], ksamples)
            X[idx,:] = drawn_samples ; labels[idx]= [k] * ksamples
        self.X = X ; self.labels = labels.astype(int)
        self.plot_gmm( draw_ellipse = draw_ellipse )
        return self
    
    def Params_Generator(self):
        """  generate Means"""
        upper   = self.upper
        lower  = - self.upper
        subintervals  = 1/self.n_points
        h   = (upper -lower)*subintervals 
        discretization = h*np.random.permutation(np.tile(np.arange(self.n_points),(self.dim ,1)).ravel()).reshape(-1,self.dim)
        if self.edge < self.n_components : self.edge = self.n_components
        discretization = discretization[np.random.choice(discretization.shape[0] , self.edge) , :][: self.n_components ]
        discretization+=np.random.uniform(-self.epsilon_noise ,self.epsilon_noise ,
                                          self.n_components*self.dim).reshape(self.n_components,self.dim)
        self.Means     = discretization
        """   generate Weights """
        if self.Adjusting_weights == 'equal_weights': 
            self.Weights  = [1./self.n_components]*self.n_components
        if self.Adjusting_weights == 'random_weights' :
            self.Weights  = list(np.random.dirichlet(np.ones(self.n_components),size=1).reshape(self.n_components))
        """  generate Covars """
        self.Covars       = np.full((self.n_components , self.dim , self.dim) , np.identity(self.dim))
        
    def control_gmm_clusters_2d(self , method ,edge , upper , n_points , epsilon_noise ):
        def func_animated(method , edge , upper , n_points , epsilon_noise):
            samples = self.samples
            model   =  self.fit(samples  , Adjusting_weights = method , edge = edge , 
                   upper = upper, n_points = n_points , epsilon_noise = epsilon_noise)
            return model.X 
        
        X = func_animated(method , edge , upper , n_points , epsilon_noise)
        tini= X[:,0].min()-1000 ; tend=X[:,1].max() +1000
        p = figure(x_range=(tini, tend), plot_height=300, plot_width=900, title="edge_evaluation")
        plt_p = p.x(X[:,0], X[:,1], line_width=2 , legend_label="data")
        show(column(p), notebook_handle=True)
        method = np.random.choice(['random_weights','eqaul_weights']) 
        def update(method , edge, upper , n_points , epsilon_noise):
            X  = func_animated(method , edge ,upper , n_points, epsilon_noise) 
            tini=X[:,0].min()  ; tend=X[:,1].max()
            plt_p.data_source.data = dict(x=X[:,0], y=X[:,1] )
            output_notebook() 
            push_notebook()
        interact(update ,
                edge            = IntSlider(min=1, max=self.n_components,value = edge  , step=1, continuous_update=False ),
                upper        = FloatSlider(min=-100., max=100.,value = upper  , step=10, continuous_update=False ),
                n_points             = IntSlider(min=1, max=100,value = n_points , step=5, continuous_update=False ),
                epsilon_noise          = FloatSlider(min=-100., max=100.,value = epsilon_noise  , step=10, continuous_update=False ),
                method  = Dropdown(options=['random_weights','eqaul_weights'], value = method, description='Method:'))
        
    def plot_gmm(self , draw_ellipse = False) :
        if self.dim == 2 :
            if draw_ellipse :
                for i in range(self.n_components):
                    x1, x2 = gmm_ellipse_2d(self.Means[i] , self.Covars[i] , sdwidth=1, points=100)
                    plt.plot(x1, x2, 'k', linewidth=2)
                plt.xlabel('$x_1$'); plt.ylabel('$x_2$')
            plt.scatter(self.X[:,0],self.X[:,1], c=self.labels)
            plt.show()
        elif self.dim == 3 :
            if draw_ellipse :
                gmm_sphere_3d( self.X ,self.labels , self.Means.T , np.sum(np.sqrt(self.Covars).T, 1) , self.Weights )
            fig_1 = plt.figure(figsize=(18, 18)) 
            axes_2 = fig_1.add_subplot(222, projection='3d') 
            axes_2.scatter(self.X[:,0],self.X[:,1], self.X[:,2], c = self.labels)
            plt.show()
        elif self.dim >3 :
            X_reduced = PCA(n_components=3 ).fit_transform(self.X)
            fig = plt.figure(figsize=(18, 18)) 
            ax = fig.add_subplot(222, projection='3d') 
            ax.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2], c = self.labels , cmap=plt.cm.Set1, edgecolor='k', s=40)
            ax.set_title("First three PCA directions")
            ax.set_xlabel("1st eigenvector")
            ax.w_xaxis.set_ticklabels([])
            ax.set_ylabel("2nd eigenvector")
            ax.w_yaxis.set_ticklabels([])
            ax.set_zlabel("3rd eigenvector")
            ax.w_zaxis.set_ticklabels([])
            plt.show()
