# Modified FitzHugh-Nagumo
This model was created by Hector Augusto Velasco Perez. A detailed analysis of the model itself can be found in his thesis entitled "Methods for Modeling Reduction in Cardiac Dynamics".

This code is meant to run a basic simulation of the modified-FHN model. It's current state is a bare bones simulation that simply updates matrices for the fast and slow variables of the modified-FHN model based on the paramters input into the main.py file. The default parameters in the code are based on an In Proceedings paper of mine called "A Modified Fitzhugh-Nagumo Model that Reproduces the Action Potential and Dynamics of the Ten Tusscher et al. Cardiac Model in Tissue."

The code currently runs across 8 CPU cores, this setting can be changed in the SimCompModifiedFHN.py file.

Future add-ons will include:

Graphing function file

Selection of time stamps to graph
