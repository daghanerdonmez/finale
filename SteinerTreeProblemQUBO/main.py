import dimod
from SteinerTree import SteinerTree
from steiner_to_bqm import steiner_to_bqm_Li_et_al
from tqdm import tqdm

nodes = ["a", "b", "c", "d"]
edges = [("a","b",10),("b","c",10),("a","c",10),("a","d",5),("b","d",5),("c","d",5)]
terminals = ["a","b","c"]

#nodes = ["a", "b"]
#edges = [("a","b",100)]
#terminals = ["a","b"]

problem = SteinerTree(nodes,edges,terminals)
print("SteinerTree object created")
bqm = steiner_to_bqm_Li_et_al(problem, 100)
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
