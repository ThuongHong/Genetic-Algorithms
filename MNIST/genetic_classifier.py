from config import POPULATION_SIZE, NUM_OF_CLASSES, IMAGE_SIZE, GNOME_LEN
from sklearn.metrics import accuracy_score
import numpy as np
import random


class Classifier:
    def __init__(self, chromosome):
        """
        A softmax-based classifier using weights and biases encoded in a chromosome.

        Attributes:
            bias (np.ndarray): Bias vector of shape (10,).
            weight (np.ndarray): Weight matrix of shape (10, 784), where each row corresponds to a class.
        """
        self.bias = chromosome[-10:].flatten()
        chromosome = chromosome[:-10]
        self.weight = chromosome.reshape(NUM_OF_CLASSES, IMAGE_SIZE)
        
    def softmax(self, z):
        exp_logits = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        
    def predict(self, X):
        logits = np.dot(X, self.weight.T) + self.bias
        probs = self.softmax(logits)
        return np.argmax(probs, axis=1)
        

class Individual:
    @classmethod
    def set_data(cls, X: np.ndarray, y: np.ndarray):
        cls.X_train = X
        cls.y_train = y
        
    def __init__(self, chromosome):
        """
        Represents a single individual (solution) in the genetic population.

        Attributes:
            chromosome (np.ndarray): The flat vector representing weights and biases.
            X_train (np.ndarray): Training data.
            y_train (np.ndarray): Training labels.
            fitness (float): Classification accuracy of this individual's model.
        """
        self.chromosome = chromosome
        self.fitness = self.cal_fitness()
        
    @classmethod
    def create_gnome(cls):
        return np.random.randn(GNOME_LEN)
    
    def mate(self, partner):
        mask = np.random.rand(GNOME_LEN)
        child_chromosome = np.where(mask < 0.45, self.chromosome, partner.chromosome)
        mutate_mask = mask > 0.90
        child_chromosome[mutate_mask] = child_chromosome[mutate_mask] + np.random.normal(0, 0.01, size=mutate_mask.sum())
        return Individual(np.array(child_chromosome))
                
        
    def cal_fitness(self):
        classifier = Classifier(self.chromosome)
        y_pred = classifier.predict(self.X_train)
        fitness = accuracy_score(y_true=self.y_train, y_pred=y_pred)
        return fitness
    

class Evolution:
    def __init__(self, X_train, y_train):
        """
        Runs the genetic algorithm to evolve a population of individuals for classification.

        Attributes:
            X_train (np.ndarray): Training features.
            y_train (np.ndarray): Training labels.
            generation (int): Current generation number.
            population (list): List of individuals in the population.
        """
        self.X_train = X_train
        self.y_train = y_train
        Individual.set_data(X_train, y_train)
        
        self.generation = 1
        self.population = []
        
        # Initial population
        for _ in range(POPULATION_SIZE):
            gnome = Individual.create_gnome()
            self.population.append(Individual(gnome))
    
    def start(self, fitness_threshold=0.92, max_generation=300):
        while self.generation <= max_generation:
            # Sort the population in increasing order of fitness score
            self.population = sorted(self.population, key = lambda x:x.fitness, reverse=True)

            # If the individual reach the expected fitness, break
            if self.population[0].fitness > fitness_threshold:
                break

            # Otherwise generate new offsprings for new generation
            new_generation = []

            # Perform Elitism, that mean 10% of fittest self.population goes to the next generation
            s = int((10*POPULATION_SIZE)/100)
            new_generation.extend(self.population[:s])

            # From 50% of fittest self.population, Individuals will mate to produce offspring
            s = int((90*POPULATION_SIZE)/100)
            for _ in range(s):
                parent1 = random.choice(self.population[:50])
                parent2 = random.choice(self.population[:50])
                child = parent1.mate(parent2)
                new_generation.append(child)

            self.population = new_generation

            print("Generation: {}\tFitness (Accuracy): {}".format(self.generation,self.population[0].fitness))

            self.generation += 1

        print("Generation: {}\tFitness (Accuracy): {}".format(self.generation,self.population[0].fitness))
        
        return Classifier(self.population[0].chromosome)