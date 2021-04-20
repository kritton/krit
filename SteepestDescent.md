---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.2'
      jupytext_version: 1.5.2
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

```python
import sys
!{sys.executable} -m pip install numpy
!{sys.executable} -m pip install matplotlib
!{sys.executable} -m pip install scipy
```

```python
import math
import numpy as np
from math import sqrt
from numpy import asarray
from numpy.random import rand
from numpy.random import seed
from numpy import arange
from numpy import meshgrid
from matplotlib import pyplot
from scipy.optimize import minimize_scalar
```

```python
def objective(args): # Extend any function with extend to n variables 
    return 10*args[0]**2 + args[1]**2

# derivative of objective function
def derivative(x,y):
    return asarray([20*x, 2*y])

# steepest descent algo.
def steepest(objective, derivative, epsilon):
    # track all solutions
    solutions = list()
    # generate an initial point
    solution = [1,1]
    # list of changes made to each variable
    change = [0,0]
    # run the gradient descent
    solution_eval = objective(solution)
    
    while solution_eval >= epsilon:
        #calculate gradient
        gradient = derivative(solution[0], solution[1])
        res = minimize_scalar(lambda alpha: objective([solution[i] - alpha * gradient[i] for i in range(len(solution))]))
        alpha = res.x
        solution = solution - gradient*alpha
        solutions.append(solution)
        # evaluate candidate point
        solution_eval = objective(solution)
        # report progress
        print(solution, solution_eval)
    return solutions
  
```

```python
epsilon = 10**(-5)
```

```python
solutions = steepest(objective, derivative, epsilon)
```

```python
# Plot two-dimensional path
# Plot error vs num iterations
# Generalise 
```

```python

```
