# script for testing pickle and json save and load

import genesnake as gs

save_as = "json"
genes = 10

network = gs.grn.make_DAG(genes,1.2,sign_bias=0.5)
M = gs.GRNmodel.make_model(network,combinations={"comb_prob":0.5})
M.set_pert("diag",effect=(0.89,0.91),noise=0,multipert=3,exp=1,reps=3,sign=-1)

M.save("test."+save_as)

M2 = gs.GRNmodel.load("test."+save_as)
print(vars(M2))
