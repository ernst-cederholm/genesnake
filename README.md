# GeneSNAKE 

Repository for the python package genesnake. 
The repository contains a few different things but the main things are:

- genesnake - the main tool, this contains all functions for data and network simulation, benchmarking of infered regulatory networks and the relevant files allowing this content to be imported in python.
- methods - a folder for methods which infer gene regulatory networks used either by geneSPIDER (https://bitbucket.org/sonnhammergrni/genespider/src/master/) or that were used in relation to genesnake. 


# GeneSNAKE tutorial
[Visit the website with the GeneSNAKE tutorial](https://sonnhammer-tutorials.bitbucket.io/genesnake.html)

An example script of how to use most function in genesnake is avalible in the example.py script. For more detailed instructions most of the complex functions to use are in the GRNmodel.py script which contains detailed instruction on how to use it in the doc-header. 


# Antons genesnake branch

Will make several changes to various things, so made my own branch. Will merge parts into master as necessary and permitted. Figured I should add my changes and bugfixes as I come across them instead of waiting and trying to remember what it was about.

There used to be a number of tests lying in the repository root dir. I ran all with 
run_tests.py
and they don't give pass / fail outputs. Instead lots of data is printed. In the interest of not getting held up for a long time wrapping my head around how to interpret and use those tests, I have not considered them. I put them in 
tests/
. The tests I actually consider on this branch is in
anton_tests
.













# Contributing

If you submit code try to follow the python standard syntax https://peps.python.org/pep-0008/ as much as possible. This to ensure people in the future can read the code. 
Make sure you have proper documentation of what the code does preferably line by line. For functions (and most scripts) make sure you add a docstring to document what the function does and what each in and output parameter should be. 
For recomendatios see https://github.com/google/styleguide/blob/gh-pages/pyguide.md#38-comments-and-docstrings. 

When working on the code it is recomended to creata a branch of the code you can submit changes to during the work. 
This can be done according to these instructions https://support.atlassian.com/bitbucket-cloud/docs/branch-a-repository/. 
Once the work is done merge the changes in to the master branch and prune the tree as to not leave unused and unsuported branches in the repository. These will not recive any updates and so will become unusable fairly quickly. 
In a similar way if you develop anything new for the genesnake project make sure to add these to the right folder of genesnake in the master branch and update all __init__.py files accordingly. 
Any tool that is not on the master branch will not be cloned by git and therefore unavalible to most users.

