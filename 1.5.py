import random
import numpy as np
from deap import base, creator, tools, algorithms
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from tensorflow import keras

# Create a dataset for binary classification
X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the fitness function
def evaluate(individual):
    # Decode the individual to define the neural network architecture
    hidden_layer_sizes = [int(x) for x in individual]
    
    # Create a neural network model
    model = keras.Sequential()
    model.add(keras.layers.Input(shape=(X_train.shape[1],)))
    for size in hidden_layer_sizes:
        model.add(keras.layers.Dense(size, activation='relu'))
    model.add(keras.layers.Dense(1, activation='sigmoid'))
    
    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    # Train the model
    model.fit(X_train, y_train, epochs=5, verbose=0)
    
    # Evaluate the model on the test set
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    
    # Return the negative accuracy as the fitness (maximize accuracy)
    return -accuracy,

# Define the genetic algorithm parameters
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("attr_int", random.randint, 1, 64)  # Hidden layer sizes range from 1 to 64
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_int, n=3)  # Three hidden layers
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutUniformInt, low=1, up=64, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)

# Create the initial population
population = toolbox.population(n=10)

# Create the hall of fame to track the best individual
hof = tools.HallOfFame(1)

# Run the genetic algorithm
algorithms.eaMuPlusLambda(population, toolbox, mu=5, lambda_=5, cxpb=0.7, mutpb=0.3, ngen=10, stats=None, halloffame=hof, verbose=True)

# Get the best individual
best_individual = hof[0]
print("Best Individual:", best_individual)
print("Best Accuracy:", -evaluate(best_individual)[0])
