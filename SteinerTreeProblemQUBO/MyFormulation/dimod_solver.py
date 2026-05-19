import dimod
from SteinerTreeProblemQUBO.SteinerTree import SteinerTree
from SteinerTreeProblemQUBO.MyFormulization.steiner_to_bqm_daghan import steiner_to_bqm_daghan
from tqdm import tqdm

nodes = ["a", "b", "c", "d", "e", "f", "g"]
edges = [("a","b",2),("b","c",4),("b","d",1),("a","d",6),("a","e",7),("d","f",1),("d","g",2),("f","g",4),("e","f",3)]
terminals = ["a","c","f","g"]

#nodes = ["a", "b"]
#edges = [("a","b",100)]
#terminals = ["a","b"]

problem = SteinerTree(nodes,edges,terminals)
print("SteinerTree object created")
bqm = steiner_to_bqm_daghan(problem, 10000)
print("Problem converted to BQM")
print(f"Number of variables {bqm.num_variables}")
print(f"Number of interactions {bqm.num_interactions}")

sampler = dimod.SimulatedAnnealingSampler()
sampleset = sampler.sample(bqm, num_reads=1)

best = None

for _ in tqdm(range(1000)):
    ss = sampler.sample(bqm, num_reads=1)
    sample = ss.first
    if best is None or sample.energy < best.energy:
        best = sample

print("best energy:", best.energy)
print("best sample:")
for var, value in best.sample.items():
    if value == 1:
        print(var, value)
