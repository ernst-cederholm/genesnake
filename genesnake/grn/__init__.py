"""
Init file for adding all network functions in geneSNAKE to the enviroment. 
Any added function should be added as a defined function and imported here. 
Any added class should be imported in to the genesnake/__init__.py file instead. 
"""

from .make_FFLATnetwork import make_FFLATnetwork
from .make_randomnetwork import make_random 
from .make_DAG import make_DAG
