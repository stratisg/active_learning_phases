import os
import glob
import numpy as np


def optimize(data):
    """
    Function for recommending parameter values to use in the subsequent
    simulation.  
    """
    
    # TODO: Use results from all simulations to fit a model that 
    # outputs the order parameter as a function of the model parameters.
    # TODO: Find the set of points that satisfy a certain objective 
    # such as we have the highest uncertainty or most sensitivity to 
    # input parameters.