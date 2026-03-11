import dimod
from SteinerTree import SteinerTree
from steiner_to_bqm import steiner_to_bqm_Li_et_al

#nodes = ["a", "b", "c", "d"]
#edges = [("a","b",100),("b","c",100),("a","c",100),("a","d",5),("b","d",5),("c","d",5)]
#terminals = ["a","b","c"]

nodes = ["a", "b"]
edges = [("a","b",100)]
terminals = ["a","b"]

problem = SteinerTree(nodes,edges,terminals)
print("SteinerTree object created")
bqm = steiner_to_bqm_Li_et_al(problem, 10000)
print("Problem converted to BQM")

sampler = dimod.SimulatedAnnealingSampler()
sampleset = sampler.sample(bqm, num_reads=100)

best = sampleset.first

print("best energy:", best.energy)
print("best sample:")
for var, value in best.sample.items():
    if value == 1:
        print(var, value)
