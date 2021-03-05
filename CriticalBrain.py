# +
# -- linear algebra + numpy --
import numpy as np
import scipy.sparse as sparse
from math import factorial
import networkx as nx
import pandas as pd

# -- visualization --
from tqdm import tqdm
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt

# -- miscellaneous --
from copy import deepcopy


# -

def sample_from(cum, N):
    sample = []
    for _ in range(N):
        u = np.random.rand()
        idx = np.where( (cum[:, 1] - u) > 0 )[0]
        if len(idx)>0: 
            idx = idx[-1]
            s = cum[idx, 0]
        elif len(idx)==0:
            s = cum[0, 0]
        sample.append( s )
        
    return np.array(sample) 


def GH(sys, Adj, T, nsteps=300, r1=1e-3, r2=0.2, time_save = False):
    """
        Computes the evolution of the system under the Greenberg-Hastings dynamics
        
        Parameters
        ----------
        sys    : ndarray          
            system of neurons
        T      : float             
            Threshold
        Adj    : csr sparse matrix
            Weighted Adjacency matrix 
        nsteps : int               
            number of iterations of the algorithm
        r1     : float             
            probability I->A
        r2     : float             
            probability R->I
        
        Returns
        -------
        evol   : ndarray          
            evolution of [nR, nI, nA] over the simulation
        sys    : ndarray          
            final configuration of the system
    """
    
    N = len(sys)
    evol = np.array([0, 0, 0])
    if time_save: log = sys.reshape(N, 1)
    for step in tqdm(range(nsteps)):
        # Prepare update R -> I
        R = np.where( sys == -1 )[0]
        nR = R.size
        if nR > 0:
            RtoI = ( np.random.uniform(0, 1, nR) < r2 )
            RtoI = R[ RtoI]
        else:
            RtoI = []
        # Prepare update A -> R
        AtoR = np.where( sys == 1 )[0]
        nA = AtoR.size
        # Prepare update I -> A
        I =  np.where( sys == 0 )[0]
        nI = I.size
        ItoA = ( np.random.uniform(0, 1, nI) < r1 )
        nbr = Adj[ I[:, np.newaxis], AtoR ]         
        strength = np.array([ a[:, a.nonzero()[1] ].sum() for i, a in enumerate(nbr)])
        ItoA = np.logical_or( ItoA, strength > T )
        ItoA = I[ ItoA]
        
        # Updates
        sys[ RtoI ] += 1
        sys[ AtoR ] -= 2
        sys[ ItoA ] += 1
        evol = np.vstack( ( evol, [nR, nI, nA] ) )
        if time_save: log = np.hstack( ( log, sys.reshape(N,1) ) )
    evol = np.delete(evol, 0, axis=0)
    
    if time_save:
        return sys, evol/sys.size, log
    else:
        return sys, evol/sys.size


# +
# ---- Metrics ----

def interevent( N, evol):
    """
        Computes the interevent time
        
        Parameters
        ----------
        N: int
             size of the system
        evol: ndarray (N, nsteps)
             evolution of the system. Each column is a time step
        
        Returns
        -------
        inter_ev: float
            Average interevent time
    """
    nsteps = evol.shape[1]
    act_time = np.array( [ (evol[i, :]==1).nonzero()[0] for i in range(N) ], dtype=object ) 
    inter_ev = np.array([ (act_time[i][1:]-act_time[i][:-1]).mean() if len(act_time[i])>1
                         else 300
                         for i in range(N) ] )
    return inter_ev.mean()


# +
def hemod(t, k=3, tau=30):
    """
        Hemodynamic function
    """
    p1 = 1/(k*tau*factorial(k-1))
    p2 = (t/tau)**k
    p3 = np.exp(-t/tau)
    return p1*p2*p3

def specs_mat(mat):
    """ Get average and std of matrix mat"""
    N = mat.shape[0]
    
    selfMat = np.diagonal(mat)
    avg = 2/(N*(N-1))*(np.sum(mat-selfMat) )
    std = np.sqrt( 2/(N*(N-1))*( np.sum( (mat-avg)**2 )  - np.sum( (selfMat-avg)**2 ) ) )
    return avg, std


# -

def Correlation(N, Nsteps, evol):
    """
        Computes the correlation matrix of the system
        
        Parameters
        ----------
        N: int
            size of the system
        Nsteps: int
            number of time steps
        evol: ndarray shape (N, Nsteps)
            evolution of the system. Each column is a time step
    """
    hemo_funct = hemod(Nsteps)
    conv = np.array([np.convolve( hemo_funct, evol[s,:]) for s in range(N) ])
    avg = np.mean(conv, axis=1)
    stds = np.std(conv, axis=1)

    Cij = np.zeros( (N, N) )
    for i in range(N):
        for j in range(N):
            Cij[i, j] = np.mean( conv[i]*avg[i]*conv[j]*avg[j]) / stds[i]*stds[j]

    avg_Cij, std_Cij = specs_mat(Cij)
    
    return Cij, avg_Cij, std_Cij


def linear_corr(Wij, Cij):
    """ Linear correlation between the matrices Wij and Cij"""
    N = Wij.shape[0]
    
    avgW, stdW = specs_mat(Wij)
    avgC, stdC = specs_mat(Cij)
    
    coef = 2/(N*(N-1)*stdW*stdC)
    corr = coef*np.sum( (Wij-avgW-np.diagonal(Wij))*(Cij-avgC-np.diagonal(Cij)) )
    return corr


def binary_linear_corr(Wij, Cij, th=4e-11):
    """ Linear correlation between the binarized matrices Wij Cij"""
    W1 = np.sign(Wij)
    th = Cij.mean()
    Cij[ Cij < th]  = 0
    Cij[ Cij >= th] = 1
    
    corr = linear_corr(W1, Cij)
    return corr


def susceptibility( N, nsteps, evol, hemo_flag=True):
    """
     N: size of the system
     evol: evolution of the system. Each column is a time step
    """
    #sys = np.zeros( N)
    #mask = np.random.choice([0, 1], size=N, p=[0.4, 0.6]).astype(np.bool)
    #sys[mask] += 1
    if hemo_flag:
        hemo = hemod(nsteps)
        conv = np.array([np.convolve(hemo, evol[i, :]) for i in range(N) ])
        mean_response = np.mean( conv, axis=0)
        std_response = np.array([ np.std( mean_response[max(0, i-20):min(nsteps, i+20)] ) for i in range(nsteps)])
        susc = ( std_response < 1e-3 ).nonzero()[0][0]
    else:
        mean_response = np.mean( evol, axis=0)
        std_response = np.array([ np.std( mean_response[max(0, i-20):min(nsteps, i+20)] ) for i in range(nsteps)])
        susc = ( std_response < 1e-1 ).nonzero()[0]
        
   
    return susc


# +
# ---- Full model simulation ----
# -

class BrainSim():
    def __init__(self, N, Na, Adj=None, check_corr=False):
        """
            Initialize the class
            
            Parameters
            ----------
            N: int
                Number of neurons
            Na: int
                Number of active neurons, must be <N
            Adj: csr sparse matrix, optional
                adjacency matrix of the neural network. If None a random adj with density 0.1 is defined
            check_corr: bool, optional
                if True check for the correlation matrix. Really increase the computational time
        """
        self.N  = int(N)
        self.Na = int(Na)
        
        # System initialization
        self.in_sys = np.zeros(self.N, dtype=int)
        random_active = np.random.randint(0, self.N, self.Na)
        self.in_sys[random_active] += 1
        self.cur_sys = self.in_sys
        self.check_corr = check_corr
        
        # Adj initialization
        if Adj==None:
            self.Adj = sparse.random(int(self.N), int(self.N), density=0.1, 
                           data_rvs=self._randint) 
            self.Adj = sparse.csr_matrix(self.Adj)
        else:
            self.Adj = Adj
            
    def _randint(self, size=None, random_state=None):
        """Function to initialize adj matrix"""
        return np.random.uniform(0, 1, size)
    
    def GH(self, T, nsteps=300, r1=1e-3, r2=0.2, time_save = False):
        """
            Computes the evolution of the system under the Greenberg-Hastings dynamics

            Parameters
            ----------
            sys    : ndarray          
                system of neurons
            T      : float             
                Threshold
            Adj    : csr sparse matrix
                Weighted Adjacency matrix 
            nsteps : int               
                number of iterations of the algorithm
            r1     : float             
                probability I->A
            r2     : float             
                probability R->I

            Returns
            -------
            evol   : ndarray          
                evolution of [nR, nI, nA] over the simulation
        """
        self.cur_sys = deepcopy(self.in_sys)
        evol = np.array([0, 0, 0])
        if time_save: self.log = self.cur_sys.reshape(self.N, 1)
        for step in range(nsteps):
            # Prepare update R -> I
            R = np.where( self.cur_sys == -1 )[0]
            nR = R.size
            if nR > 0:
                RtoI = ( np.random.uniform(0, 1, nR) < r2 )
                RtoI = R[ RtoI]
            else:
                RtoI = []
            # Prepare update A -> R
            AtoR = np.where( self.cur_sys == 1 )[0]
            nA = AtoR.size
            # Prepare update I -> A
            I =  np.where( self.cur_sys == 0 )[0]
            nI = I.size
            ItoA = ( np.random.uniform(0, 1, nI) < r1 )
            nbr = self.Adj[ I[:, np.newaxis], AtoR ]         
            strength = np.array([ a[:, a.nonzero()[1] ].sum() for i, a in enumerate(nbr)])
            ItoA = np.logical_or( ItoA, strength > T )
            ItoA = I[ ItoA]

            # Updates
            self.cur_sys[ RtoI ] += 1
            self.cur_sys[ AtoR ] -= 2
            self.cur_sys[ ItoA ] += 1
            
            evol = np.vstack( ( evol, [nR, nI, nA] ) )
            if time_save: self.log = np.hstack( (self.log, self.cur_sys.reshape(self.N,1)) )
        evol = np.delete(evol, 0, axis=0)

        return  evol/self.N
    
    def tau(self, activation, nsteps):
        """
            Returns lifetime activity tau, i.e. the number of steps at which the system is still active
        """
        t =  np.where(activation < 1e-4)[0]
        if t.size==0:
            t = nsteps
        else:
            t= np.min(t)
        return t
    
    def Cl_sizes(self):
        """
            Seturns first and second cluster size, with their error
        """
        A = np.where(self.cur_sys==1)[0]
        nbr = self.Adj[ A[:, np.newaxis], A ].todense()   
        graph = nx.from_numpy_matrix(nbr)
        cl_size = [len(c) for c in sorted(nx.connected_components(graph), key=len, reverse=True)]
        return (cl_size + [0]*2)[:2]
    
    def average(self, T, n_avg=10, nsteps=300, r1=1e-3, r2=0.2):
        """
            Computes all the averages needed in an experiment, using the GH algorithm
        """
        avg_evol = self.GH(T, nsteps=nsteps, r1=r1, r2=r2, time_save=True)
        avg_evol = avg_evol[:, 2]
        # Lifetime
        avg_tau = np.array( [self.tau(avg_evol, nsteps) ] )
        # Cluster size
        avg_cl = np.array( self.Cl_sizes() )
        # Correlation matrix
        if self.check_corr:
            Cij, avgCij, stdCij = Correlation(self.N, nsteps, self.log)
            avg_lin_cor = np.array( [linear_corr(self.Adj.todense(), Cij)] )
            avg_bin_cor = np.array( [binary_linear_corr(self.Adj.todense(), Cij)] )
            avg_avgCij = np.array( [avgCij] )
            avg_stdCij = np.array( [stdCij] )
        #print(avg_lin_cor)
        #print(avg_bin_cor)
        # Interevent time
        avg_interv = np.array( interevent( self.N, self.log) )
        
        for i in range(n_avg-1):
            evol = self.GH(T, nsteps=nsteps, r1=r1, r2=r2, time_save=True)
            avg_evol = np.vstack( (avg_evol, evol[:,2] ) )
            # temp variables
            tau_temp = np.array( [self.tau(evol[:, 2], nsteps) ] )
            cl_temp = np.array( self.Cl_sizes() )
            if self.check_corr:
                Cij, avgCij, stdCij = Correlation(self.N, nsteps, self.log)
                lin_cor_temp = np.array( [linear_corr(self.Adj.todense(), Cij)] )
                bin_cor_temp = np.array( [binary_linear_corr(self.Adj.todense(), Cij)] )
            # Average variables
            avg_tau = np.vstack((avg_tau, tau_temp))
            avg_cl = np.vstack( ( avg_cl, cl_temp) )
            if self.check_corr:
                avg_lin_cor = np.vstack( (avg_lin_cor, lin_cor_temp ))
                avg_bin_cor = np.vstack( (avg_bin_cor, bin_cor_temp ) )
                avg_avgCij = np.vstack( (avg_avgCij ,np.array( [avgCij] ) ) )
                avg_stdCij = np.vstack( (avg_stdCij ,np.array( [stdCij] ) ) )
            avg_interv = np.vstack( (avg_interv, np.array( interevent( self.N, self.log)) ) )
                                   
        # stats is vector of shape (1, 16)
        if self.check_corr:
            stats = np.hstack( (avg_tau.mean(), avg_tau.std(), 
                                avg_cl.mean(axis=0), avg_cl.std(axis=0), # Here there are 4 elements
                                avg_avgCij.mean(), avg_avgCij.std(),
                                avg_stdCij.mean(), avg_stdCij.std(),
                                avg_lin_cor.mean(), avg_lin_cor.std(),
                                avg_bin_cor.mean(), avg_bin_cor.std(), 
                                avg_interv.mean(), avg_interv.std()
                                ) )
        else: 
            stats = np.hstack( (avg_tau.mean(), avg_tau.std(), 
                                avg_cl.mean(axis=0), avg_cl.std(axis=0), # Here there are 4 elements
                                avg_interv.mean(), avg_interv.std()
                                ) )
        return np.hstack( ( avg_evol.mean(axis=0), avg_evol.std(axis=0) ) ), stats
    
    def experiment(self, Ts, n_avg=10, nsteps=300, r1=1e-3, r2=0.2):
        """
            Run an experiment with different thresholds
        """
        self.thresh = Ts
        self.total_evol = np.zeros( nsteps).reshape(nsteps, 1)
        if self.check_corr:
            self.total_stats = np.zeros(16)
        else:
            self.total_stats = np.zeros(8)
        for i, T in tqdm(enumerate(Ts)):
            evol, stats = self.average(T, n_avg=n_avg, nsteps=nsteps, r1=r1, r2=r2)
            self.total_evol = np.hstack( (self.total_evol, evol.reshape(nsteps, 2)) )
            self.total_stats= np.vstack( (self.total_stats, stats))
        self.total_evol = np.delete(self.total_evol, 0, axis=1)
        self.total_stats = np.delete(self.total_stats, 0, axis=0)
        
        return self.total_evol, self.total_stats
    
    def save_evol(self, filename):
        """
            Save the evolution
            
            Parameters
            ----------
            filename: string
                path to save the evolution
            
            Returns
            -------
            None: None
        """
        np.save('filename', self.total_evol)
        
    def save_stats(self, filename):
        """
            Save all the statistics
            
            Parameters
            ----------
            filename: string
                path to save the statistics
            
            Returns
            -------
            None: None
        """
        if self.check_corr:
            columns = ['Threshold', 'tau', 'tau_err', 'Cl1', 'Cl2', 'Cl1_err', 'Cl2_err', 'AvgCor', 'AvgCor_err', 
                       'StdCor', 'StdCor_err', 'LinCor', 'LinCor_err', 'BinCor', 'BinCor_err', 'Interevent', 'Intervent_err']
        else:
            columns = ['Threshold', 'tau', 'tau_err', 'Cl1', 'Cl2', 'Cl1_err', 'Cl2_err',  
                        'Interevent', 'Intervent_err']
        
        final = np.hstack( (np.array(self.thresh).reshape(len(self.thresh), 1), self.total_stats) )
        dataframe = pd.DataFrame( final, columns=columns)
        dataframe.to_csv(filename)


# +
# ---- Simulating attacks ----

# +
def attack_node(graph, metric, nsteps=50):
    """
        Remove @nsteps nodes with highest @metric from @graph
        
        Parameters
        ----------
        nsteps: int, optional
            number of nodes to be removed
        metric: ndarray
            array with metric to be used for the removal
        graph: networkx graph
            original graph where to attack
            
        Returns
        -------
        graph: networkx graph
            New attacked graph
    """
    graph = graph.copy()
    nodes = np.array( [n for n in graph.nodes()] )
    sorting = np.argsort(metric)[::-1]
    sorted_nodes = nodes[sorting]
    for i in range(nsteps):
        graph.remove_node( sorted_nodes[i] )

    return graph

def attack_link(graph, metric, nsteps=50):
    """
        Remove @nsteps edges with highest @metric from @graph
        
        Parameters
        ----------
        nsteps: int, optional
            number of nodes to be removed
        metric: ndarray
            array with metric to be used for the removal
        graph: networkx graph
            original graph where to attack
            
        Returns
        -------
        graph: networkx graph
            New attacked graph
    """
    graph = graph.copy()
    edges = np.array( [e for e in graph.edges()] )
    sorting = np.argsort(metric)[::-1]
    sorted_edges = edges[sorting]
    for i in range(nsteps):
        graph.remove_edge( sorted_edges[i][0], sorted_edges[i][1] )

    return graph


# -

class Animate:
    """ Animate the evolution along the different tresholds of the activity"""
    def __init__(self, evolutions, thresholds):
        self.fig , self.ax = plt.subplots( figsize=(12,8))
        self.ax.set_xlabel('Timestep', fontsize=16)
        self.ax.set_title("Activity evolution", fontsize=20)
        self.ax.set_ylabel('Activity', fontsize=16)
        #self.ax.set_ylim([0, 0.04])#np.max(distribution)])
        # Parameters
        self.N = len(evolutions[:, 0])
        self.Nth = len(thresholds)
        self.evol = evolutions[:, np.arange(2*self.Nth)%2==0]
        self.stds = evolutions[:, np.arange(2*self.Nth)%2==1]
        self.thresholds = thresholds
        self.line = self.ax.plot(self.evol[:, 0], label=f'Threshold T={self.thresholds[0]}')
        self.fill = self.ax.fill_between( np.arange(self.N), self.evol[:, 0]+self.stds[:, 0], 
                                         self.evol[:, 0]-self.stds[:, 0], color='green', 
                                         alpha=0.5, label='Standard deviation')
        self.legend = self.ax.legend(fontsize=15)
    
    def _update(self, i):
        self.ax.collections.clear()
        if i< self.Nth:
            self.line[0].set_ydata(self.evol[:, i])
            self.fill = self.ax.fill_between( np.arange(self.N), self.evol[:, i]+self.stds[:, i], 
                                         self.evol[:, i]-self.stds[:, i], color='green', 
                                         alpha=0.5, label='Standard deviation')
            
            self.legend.texts[0].set_text(f'Threshold T={self.thresholds[i]}')
            self.ax.set_ylim([0, np.max(self.evol[:, i]+self.stds[:, i])+np.max(self.evol[:, i]+self.stds[:, i])/10])
        else:
            self.line[0].set_ydata(self.evol[:, -1])
            self.fill = self.ax.fill_between( np.arange(self.N), self.evol[:, -1]+self.stds[:, -1], 
                                         self.evol[:, -1]-self.stds[:, -1], color='green', 
                                         alpha=0.5, label='Standard deviation')
            
            self.legend.texts[0].set_text(f'Threshold T={self.thresholds[-1]}')
            self.ax.set_ylim([0, np.max(self.evol[:, -1]+self.stds[:, -1])+np.max(self.evol[:, -1]+self.stds[:, -1])/10])
        
    def sys_anim(self, interval=100, filename='filename'):
        self.anim = FuncAnimation(self.fig, self._update, frames=self.Nth+2, interval=interval)
        self.anim.save(filename)
