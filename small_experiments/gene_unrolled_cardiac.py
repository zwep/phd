
import numpy
import reconstruction.ReadListData as read_list
import reconstruction.ReadCpx as read_cpx
import os
import helper.plot_class as hplotc
import numpy as np
import small_experiments.example_unfolding as exp_unfold

dir_data = '/home/bugger/Documents/data/7T/cardiac'
list_files = [os.path.join(dir_data, x) for x in os.listdir(dir_data) if x.endswith('list')]

list_obj = read_list.DataListImage(list_files[0])
data_array, data_img = list_obj.get_image_data()


class GeneticUnfolding:
    def __init__(self, folded_image, fold_factor, n_generations, n_parents, n_child, n_mutate,
                 sel_perc=10, init_population=None):
        self.folded_image = folded_image
        self.fold_factor = fold_factor
        self.n_parents = n_parents
        self.n_child = n_child
        self.n_mutate = n_mutate
        self.sel_perc = sel_perc
        self.n_generations = n_generations
        self.n_c, self.ny, self.nx = folded_image.shape
        pop_shape = (n_parents, self.n_c, self.ny * fold_factor, self.nx)

        self.population = init_population
        if init_population is None:
            self.population = np.random.rand(*pop_shape) + 1j * np.random.rand(*pop_shape)

    def population_fitness(self):
        # Calculate the fitness of the current population
        fitness = []
        for i_person in self.population:
            temp_unfold = self.unfold_image(self.folded_image, i_person, self.fold_factor)
            fitness_level = np.mean(np.abs(temp_unfold)) / np.std(np.abs(temp_unfold))
            fitness.append(fitness_level)
        return fitness

    @staticmethod
    def unfold_image(folded_image, sensitivity_map, fold_factor):
        unfolded_img = exp_unfold.unfold(folded_image, sensitivity_map, fold_factor)
        return unfolded_img

    def parent_selection(self, fitness):
        least_perforing = np.argsort(fitness)[:self.sel_perc]
        best_perforing = np.argsort(fitness)[-self.sel_perc:]
        new_parents_index = np.concatenate([least_perforing, best_perforing])
        new_parents = self.population[new_parents_index]
        return new_parents

    def create_offspring(self, parents):
        offspring = []
        n_parents = len(parents)
        for k in range(n_parents):
            # Index of the first parent to mate.
            parent1_idx = k % n_parents
            # Index of the second parent to mate.
            parent2_idx = (k + 1) % n_parents
            # The new offspring will have its first half of its genes taken from the first parent.
            temp_offspring = (parents[parent2_idx] + parents[parent1_idx]) / 2
            offspring.append(temp_offspring)

        return offspring

    def mutate_population(self, population):
        # MUTATE THEM KIDS!!
        # add noise?
        mutated_population = []
        n_population = len(population)
        for i_child in range(n_population):
            for i_mutate in range(self.n_mutate):
                temp_shape = (self.n_c, self.ny * self.fold_factor, self.nx)
                random_positions = np.random.randint(0, 2, temp_shape)
                temp = population[0]
                temp[random_positions.astype(bool)] += np.random.random(random_positions.sum())

                mutated_population.append(temp)
        return mutated_population

    def run(self):
        for i_generation in range(self.n_generations):
            print('Generation number: ', i_generation)
            # See performance of current population
            fitness_score = self.population_fitness()

            # Make a selection on parents
            new_parents = self.parent_selection(fitness_score)

            # Create new kiddos
            new_kiddos = self.create_offspring(new_parents)

            # Mutate them and assign as new population
            self.population = np.array(self.mutate_population(new_kiddos))


A_folded = np.squeeze(data_img)[-8:]

import helper.dummy_data as hdummy
n_c, ny, nx = A_folded.shape
derp_sens = hdummy.get_sentivitiy_maps(ny * 3, nx, n_c)
n_fold_factor = 3
Gene_obj = GeneticUnfolding(A_folded, init_population=derp_sens[None],
                            fold_factor=n_fold_factor, n_generations=10, n_parents=2,
                            n_child=2, n_mutate=2)
Gene_obj.run()
hplotc.SlidingPlot(Gene_obj.population)

fitness_score = Gene_obj.population_fitness()
print('fitness score', fitness_score)

# Make a selection on parents
new_parents = Gene_obj.parent_selection(fitness_score)

# Create new kiddos
new_kiddos = Gene_obj.create_offspring(new_parents)

# Mutate them and assign as new population
population_mutated = Gene_obj.mutate_population(new_kiddos)
