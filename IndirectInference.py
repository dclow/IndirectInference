import numpy as np
import scipy.optimize as spo

# =============================================================================
# ============== Define a function to simulate data  ==========================
# =============================================================================

def simulate_data(N, T, rho):
    """
    A function to simulate an AR(1) process for N individuals over T periods,
    allowing for individual-specific fixed effects.
    
    The process is given by:    
        y_it = alpha_i + rho y_i(t-1) + epsilon_it
    
    Note it does not allow for tempory shocks, though that would be easy to add.    
    
    Parameters
    ----------
    N: int
        Number of individuals in the simulated data
        
    T: int
        Number of time periods in the simulate data
        
    rho: float
        Autoregressive parameter of the AR(1) process
    
    Returns
    ----------
    simulated_data:  an N x T np.ndarray
        The simulated data, which consists of an outcome of the AR(1) process
        for individual i (0<=i<=N-1) at time t (0<=t<=T-1)
    """
    fixed_effects       = np.random.normal(size = (N,1))    
    shocks              = np.random.normal(size = (N,T))

    # Initialize an array to hold the simulated data
    simulated_data      = np.zeros((N,T))
    
    # Tale the first draws of the data from the stationary distribution
    simulated_data[:,0:1] = np.random.normal(loc = (fixed_effects / (1. - rho)), 
                                             scale = 1./((1. - rho**2)**(.5)), 
                                             size = (N,1)) 

    # Simulate the data forward
    for tt in range(1,T):
        simulated_data[:,tt] = fixed_effects[:,0] + \
                               rho * simulated_data[:,tt-1] + \
                               shocks[:,tt]
                               
    return simulated_data


# =============================================================================
# ===== Define the Fixed Effects estimator of an autoregressive parameter =====
# =============================================================================

def estimate_rho_FE(data):
    """
    A function to estimate the autoregressive parameter of an AR(1) process
    (e.g. the persistence of income shocks) with the Fixed Effects estimator
    (for more information, see Hayashi p. 327)
    
    Parameters
    ----------
    data: an N x T np.ndarray
        panel data on an AR(1) process, whose autoregressive parameter
        will be estimated

    Returns
    ----------
    rhohat_FE: float
        the Fixed Effects estimate of the autoregressive parameter
    """

    # Figure out how many time periods are in the data
    T = data.shape[1]

    # Define a time-demeaning matrix (the matrix Q in Hayashi, p. 327)
    # Multiplication by this matrix removes individual-level means 
    demeaning_matrix = np.matrix(np.eye(T) - (1./T))
    
    # Now, demean the data
    demeaned_data = data * demeaning_matrix    

    # Get the outcome variable (the demeaned data) and the predictor variable 
    # (the lagged demeaned data) to be the right shape for estimation
    Y = demeaned_data[:,1:].flatten().swapaxes(0,1)
    X = demeaned_data[:,:-1].flatten().swapaxes(0,1)

    # Finally, take the Fixed Effects estimator, which is just (X'X)^(-1)(X'Y)
    rhohat_FE = np.linalg.inv(X.transpose() * X) * \
                (X.transpose() * Y)

    return float(rhohat_FE)

# =============================================================================
# ================= Define an Indirect Inference estimator  ==================
# =============================================================================

def indirect_inference(data, simulation_function, auxiliary_function, **kwds):
    """
    A function to estimate a single parameter by Indirect Inference.  
    Estimation works as follows.
        (1) Calculate auxiliary_function(data).
        (2) Simulate data for different values of the parameter of interest
        (3) Find the value of the parameter of interest such that, at this
            value, the distance between auxiliary_function(data) and 
            auxiliary_function(simulated_data) is minimized.  This value is
            the Indirect Inference estimate of the parameter.
            
    A major advantage of Indirect Inference is the considerable flexibility
    econometricians have in choosing auxiliary_function.  Indirect inference is
    consistent as long as the mapping from structural parameters to the
    auxiliary model parameters has (locally) full rank.
    
    For more info, see Smith (1993) or Gourieroux, Monfort, and Renault (1993).

    Parameters
    ----------
    data: an N x T np.ndarray
        panel dataset
        
    simulation_function: a function
        this function takes as input the parameter of interest, and returns as
        output simulated data of the appropriate shape

    auxiliary_function: a function
        this function must take as input an N x T panel dataset, and return a
        scalar as output
        
    kwds: keywords
        keywords to pass to the minimization routine, if desired

    Returns
    ----------
    parameter_estimate: float
        the Indirect Inference estimate of the parameter of interest


    """
    
    # Figure out how many individuals and time periods are in the panel data
    N = data.shape[0]
    T = data.shape[1]

    # Now, get the target value for the axuiliary_function in estimation.
    # This is just the value of auxiliary_function, evaluated with the data.
    target = auxiliary_function(data)
    
    # Define the objective function that Indirect Inference minimizes.
    # This is the distance between the target value for auxiliary_function,
    # and the value of auxiliary_function evaluated with simulated data
    def objective_function(parameter):
        simulated_data = simulation_function(N,T,parameter)        
        
        return np.abs(auxiliary_function(simulated_data) - target)
    
    # The parameter estimate minimizes the objective function
    parameter_estimate = spo.minimize_scalar(objective_function,
                                             **kwds).x
    
    return parameter_estimate
    
# =============================================================================
# =========== Define a function to compare parameter estimates =============
# =============================================================================    
    
def compare_estimates(N,T,rho = .6):
    """
    A simple function to compare the Fixed Effects and Indirect Inference
    estimates of the persistence of an AR(1) process.

    Note, for example, that the Fixed Effects estimator is significantly biased
    in short panels (T around 5).  However, the Indirect Inference estimator
    has no problem with short panels.


    Parameters
    ----------
    N: integer
        number of individuals to simulate data for
        
    T: integer
        number of time periods to simulate data for
        
    rho: float
        true persistence of the AR(1) process to use


    Returns
    ----------
    Nothing, just prints the estimates for comparison.
    
    """

    # Simulate data to use
    data = simulate_data(N,T,rho)    
    
    # Using simulated data, get the Fixed Effects estimate of rho
    rhohat_FE = estimate_rho_FE(data)
    
    # Using simulated data, get the Indirect Inference estimate of rho
    rhohat_II = indirect_inference(data,
                                   simulate_data,
                                   estimate_rho_FE,
                                   bounds=(.01,.99),method='bounded')         
    
    print 'Actual value is', rho
    print 'Fixed Effects estimate is', rhohat_FE
    print 'Indirect Inference estimate is', rhohat_II

  

# Note the Indirect Inference estimator performs much better with short panels.
T = 5
print ''
print 'Short Panel, T = ', T
compare_estimates(1000,T)

print ''
print ''

# With long panels, the estimators are comparable.
T = 100
print ''
print 'Long Panel, T = ', T
compare_estimates(1000,T)