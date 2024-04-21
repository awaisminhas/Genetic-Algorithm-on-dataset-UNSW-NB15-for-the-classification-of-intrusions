#!/usr/bin/env python
# coding: utf-8

# ### Importing necessary modules:

# In[51]:


import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np
from sklearn.metrics import classification_report


# ### Reading training and testing data

# In[52]:


train_data= pd.read_parquet('UNSW_NB15_training-set.parquet')


# In[53]:


train_data


# In[54]:


test_data= pd.read_parquet('UNSW_NB15_testing-set.parquet')


# In[55]:


test_data


# ### Setting columns that we need

# In[56]:


columns = ['dur', 'proto', 'sbytes', 'dbytes', 'spkts', 'dpkts', 'rate', 'sload', 'dload', 'label']


# In[57]:


train_data = train_data[columns]


# In[58]:


train_data.head()


# In[59]:


test_data = test_data[columns]


# In[60]:


test_data.head()


# ### Preprocessing:

# ##### extracting numerical columns

# In[61]:


numeric_cols = train_data.select_dtypes(include=np.number).columns.tolist()


# ##### removing 'label' column:`

# In[62]:


if 'label' in numeric_cols:
    numeric_cols.remove('label')


# In[63]:


print("Numerical Columns:", numeric_cols)


# ##### extracting categorical columns

# In[64]:


categorical_cols = train_data.select_dtypes(exclude=np.number).columns.tolist()


# In[65]:


print("Categorical Columns:", categorical_cols)


# ##### Initializing a StandardScaler for numerical features

# In[66]:


numerical_transformer = StandardScaler()


# ##### Initializing a OneHotEncoder for categorical features

# In[67]:


categorical_transformer = OneHotEncoder(handle_unknown='ignore')


# ##### Applying transformers to relevant columns using ColumnTransformer (Defining the processing steps)

# In[68]:


preprocessing_steps = ColumnTransformer(transformers=[('num', numerical_transformer, numeric_cols),('cat', categorical_transformer, categorical_cols)])


# ###### Update the pipeline with the new steps

# In[69]:


preprocess_pipeline = Pipeline(steps=[('preprocessor', preprocessing_steps)])


# ##### Reapply transformations to both training and testing data

# In[70]:


x_train_preprocessed = preprocess_pipeline.fit_transform(train_data.drop('label', axis=1))


# In[71]:


x_test_preprocessed = preprocess_pipeline.transform(test_data.drop('label', axis=1))


# In[72]:


# Display the shapes of the preprocessed datasets
x_train_preprocessed.shape,x_test_preprocessed.shape


# In[73]:


y_train = train_data['label']
y_test = test_data['label']


# In[74]:


y_train.shape


# In[75]:


y_test.shape


# ### Applying genetic algorithm:

# In[76]:


population_size = 100


# In[77]:


chromosome_length = x_train_preprocessed.shape[1]
print(chromosome_length)


# In[78]:


population = np.random.randint(0, 2, (population_size, chromosome_length))
print(population)


# ##### function to calculate fitness

# In[79]:


accuracy = 0.0
precision =0.0

def calculate_fitness(individual, features, labels):
    if isinstance(features, np.ndarray):
        features_dense = features
    else:
        features_dense = features.toarray()
    individual = np.array(individual).reshape(-1)
    prediction_scores = np.dot(features_dense, individual)
    predictions = prediction_scores > 0.5
    true_positives = np.sum((predictions == 1) & (labels == 1))
    true_negatives = np.sum((predictions == 0) & (labels == 0))
    false_positives = np.sum((predictions == 1) & (labels == 0))
    false_negatives = np.sum((predictions == 0) & (labels == 1))
    return true_positives * 2 + true_negatives - false_positives - 2 * false_negatives


# ##### Function to delect parents to generate a new generation:

# In[80]:


def select(population, fitness):
    fitness_shifted = fitness - np.min(fitness) + 1e-3
    probability = fitness_shifted / np.sum(fitness_shifted)
    indices = np.random.choice(np.arange(population_size), size=population_size, p=probability)
    return population[indices]


# ##### Crossover :)

# In[81]:


def crossover(parent1, parent2):
    point = np.random.randint(1, chromosome_length - 1)
    child1 = np.concatenate((parent1[:point], parent2[point:]))
    child2 = np.concatenate((parent2[:point], parent1[point:]))
    return child1, child2


# ##### Mutations:

# In[82]:


def mutate(individual, mutation_rate=0.01):
    for i in range(chromosome_length):
        if np.random.rand() < mutation_rate:
            individual[i] = 1 - individual[i]
    return individual


# ##### Executing the genetic algorithm:

# ##### For multiple populations

# In[83]:


population_sizes = [50, 100, 200]
for pop_size in population_sizes:
    print(f"Population Size: {pop_size}")
    population_size = pop_size
    population = np.random.randint(0, 2, (population_size, chromosome_length))
    best_chromosome = None
    best_fitness_score = -np.inf
    for generation in range(int(30)):  
        fitness = np.array([calculate_fitness(ind, x_train_preprocessed, y_train) for ind in population])
        if np.max(fitness) > best_fitness_score:
            best_fitness_score = np.max(fitness)
            best_chromosome = population[np.argmax(fitness)]
        population = select(population, fitness)
        next_population = []
        for i in range(0, population_size, 2):
            parent1, parent2 = population[i], population[i+1]
            child1, child2 = crossover(parent1, parent2)
            next_population.extend([child1, child2])
        population = np.array([mutate(ind) for ind in next_population])
        print(f"Generation {generation}: Best Fitness - {best_fitness_score}")
    print()


# ##### Experiment to on varied mutation ratess:

# In[84]:


mutation_rates = [0.01, 0.05, 0.1]

for mutation_rate in mutation_rates:
    print(f"Mutation Rate: {mutation_rate}")
    best_chromosome = None
    best_fitness_score = -np.inf
    for generation in range(int(30)):  
        fitness = np.array([calculate_fitness(ind, x_train_preprocessed, y_train) for ind in population])
        if np.max(fitness) > best_fitness_score:
            best_fitness_score = np.max(fitness)
            best_chromosome = population[np.argmax(fitness)]
        population = select(population, fitness)
        next_population = []
        for i in range(0, population_size, 2):
            parent1, parent2 = population[i], population[i+1]
            child1, child2 = crossover(parent1, parent2)
            next_population.extend([child1, child2])
        population = np.array([mutate(ind, mutation_rate) for ind in next_population])
        print(f"Generation {generation}: Best Fitness - {best_fitness_score}")
    print()


# #### Crossover experiment (single-point, multi-point, uniform)

# In[85]:


def single_point_crossover(parent1, parent2):
    point = np.random.randint(1, len(parent1) - 1)
    child1 = np.concatenate((parent1[:point], parent2[point:]))
    child2 = np.concatenate((parent2[:point], parent1[point:]))
    return child1, child2


# In[86]:


def multi_point_crossover(parent1, parent2):
    num_points = np.random.randint(1, len(parent1) - 1)
    points = sorted(np.random.choice(range(1, len(parent1)), num_points, replace=False))
    child1 = np.copy(parent1)
    child2 = np.copy(parent2)
    for i in range(0, len(points), 2):
        if i < len(points) - 1:
            child1[points[i]:points[i+1]], child2[points[i]:points[i+1]] = child2[points[i]:points[i+1]], child1[points[i]:points[i+1]]
    return child1, child2


# In[87]:


def uniform_crossover(parent1, parent2, prob=0.5):
    child1 = np.copy(parent1)
    child2 = np.copy(parent2)
    for i in range(len(parent1)):
        if np.random.rand() < prob:
            child1[i], child2[i] = child2[i], child1[i]
    return child1, child2


# In[88]:


# Experiment 2: Crossover Type Variation
crossover_types = ["single-point", "multi-point", "uniform"]

for cross_type in crossover_types:
    print(f"Crossover Type: {cross_type}")
    if cross_type == "single-point":
        crossover_func = single_point_crossover
    elif cross_type == "multi-point":
        crossover_func = multi_point_crossover
    elif cross_type == "uniform":
        crossover_func = uniform_crossover
    best_chromosome = None
    best_fitness_score = -np.inf
    for generation in range(int(30)):  
        fitness = np.array([calculate_fitness(ind, x_train_preprocessed, y_train) for ind in population])
        if np.max(fitness) > best_fitness_score:
            best_fitness_score = np.max(fitness)
            best_chromosome = population[np.argmax(fitness)]
        population = select(population, fitness)
        next_population = []
        for i in range(0, population_size, 2):
            parent1, parent2 = population[i], population[i+1]
            child1, child2 = crossover_func(parent1, parent2)
            next_population.extend([child1, child2])
        population = np.array([mutate(ind) for ind in next_population])
        print(f"Generation {generation}: Best Fitness - {best_fitness_score}")
    print()

