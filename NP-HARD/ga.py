import random
from collections import Counter

POPULATION_SIZE = 100

class Individual:
    @classmethod
    def set_data(self, N):
        self.N = N
        self.GENE = range(N)
        self.TARGET = N * (N - 1) // 2
    
    def __init__(self, chromosome):
        self.chromosome = chromosome
        self.fitness = self.cal_fitness()
        
    @classmethod
    def mutated_genes(self, base_gene):
        a, b = random.sample(self.GENE, 2)
        gene = list(base_gene).copy()
        gene[a], gene[b] = gene[b], gene[a]
        return tuple(gene)
        
    @classmethod
    def create_gnome(self):
        gene = list(self.GENE)
        random.shuffle(gene)
        return tuple(gene)
    
    def mate(self, partner):
        prob = random.random()
            
        if prob < 0.25:
            return Individual(self.mutated_genes(self.chromosome))
                
        elif prob < 0.5: 
            return Individual(self.mutated_genes(partner.chromosome))

        else:
            return Individual(self.create_gnome())

    def cal_fitness(self):
        diag1 = Counter()
        diag2 = Counter()

        for i in range(self.N):
            diag1[i - self.chromosome[i]] += 1
            diag2[i + self.chromosome[i]] += 1

        def count_conflicts(counter):
            return sum(v * (v - 1) // 2 for v in counter.values() if v > 1)

        conflicts = count_conflicts(diag1) + count_conflicts(diag2)
        return conflicts
    
    
def GeneticAlgorithm(N):
    Individual.set_data(N)

    #current generation
    generation = 1

    found = False
    population = []
    
    # create initial population
    for _ in range(POPULATION_SIZE):
                gnome = Individual.create_gnome()
                population.append(Individual(gnome))

    while not found:

        population = sorted(population, key = lambda x:x.fitness)

        # if the individual having lowest fitness score ie. 
        # 0 then we know that we have reached to the target
        # and break the loop
        if population[0].fitness == 0:
            found = True
            break

        # Otherwise generate new offsprings for new generation
        new_generation = []

        # Perform Elitism, that mean 10% of fittest population
        # goes to the next generation
        s = int((10*POPULATION_SIZE)/100)
        new_generation.extend(population[:s])

        s = int((50*POPULATION_SIZE)/100)
        while len(new_generation) < POPULATION_SIZE:
            parent1, parent2 = random.sample(population[:s], 2)
            child = parent1.mate(parent2)
            new_generation.append(child)

        population = new_generation

        # print("Generation: {}\tFitness: {}".format(generation, population[0].fitness))

        generation += 1

    
    # print("Generation: {}\tFitness: {}".format(generation, population[0].fitness))
    
    return population[0].chromosome
    
if __name__ == '__main__':
    GeneticAlgorithm(50)