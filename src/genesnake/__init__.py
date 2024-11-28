"""
Main init file for the geneSNAKE module
Imports all name space variables in to the env so that geneSNAKE can be called 
easily called.
To limit how many sub calls are needed
Classes should in general be imported so that they can be used as: 
    genesnake.class.function 
Stand alone functions should be imported so that they are in the right sub group e.g.
    network·random_network(10,10)
"""

# import the classes that should be under genesnake.class
from .data.GRNmodel import GRNmodel

# import the entier folders and corresponding functions
# Note any function added to a directory here needs to be added in the
# corresponding __init__ e.g. for fflat_network the network/__init__
# must have the line from .FFLAT import FFLAT
from . import grn
from . import data
from . import util 
from . import benchmarking
from . import analysis
from . import inference