from typing import Type

from erpy import random_state
from erpy.framework.population import Population
from erpy.framework.reproducer import ReproducerConfig
from erpy.instances.reproducers.default import DefaultReproducer


class CrossoverReproducerConfig(ReproducerConfig):
    @property
    def reproducer(self) -> Type[DefaultReproducer]:
        return CrossoverReproducer


class CrossoverReproducer(DefaultReproducer):
    def reproduce(self, population: Population) -> None:
        amount_to_create = population.config.population_size - len(population.to_evaluate)

        for _ in range(amount_to_create):
            parent_ids = random_state.choice(list(population.to_reproduce), size=2, replace=False)
            parent1_genome = population.genomes[parent_ids[0]]
            parent2_genome = population.genomes[parent_ids[1]]

            # Apply crossover
            child_genome = parent1_genome.cross_over(parent2_genome, self.next_genome_id)

            # Apply mutation
            child_genome = child_genome.mutate(self.next_genome_id)

            # Add the child to the population
            population.genomes[child_genome.genome_id] = child_genome

            # New children should always be evaluated
            population.to_evaluate.add(child_genome.genome_id)