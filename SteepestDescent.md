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
from matplotlib import pyplot as plt
from scipy.optimize import minimize_scalar
```

```python
def objective(args): # Extend any function with extend to n variables 
    return 10*args[0]**2 + args[1]**2

# derivative of objective function
def derivative(args):
    return np.asarray([20*args[0], 2*args[1]])

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
    
    errors = []
    
    while solution_eval >= epsilon:
        #calculate gradient
        gradient = derivative(solution)
        res = minimize_scalar(lambda alpha: objective([solution[i] - alpha * gradient[i] for i in range(len(solution))]))
        alpha = res.x
        solution = solution - gradient*alpha
        solutions.append(solution)
        # evaluate candidate point
        solution_eval = objective(solution)
        # report progress
        print(solution, solution_eval)
        errors.append(solution_eval)
    return solutions, errors
  
```

```python
epsilon = 10**(-5)
```

```python
solutions = steepest(objective, derivative, epsilon)
```

```python
res
```

```python
# Plot two-dimensional path
# Plot error vs num iterations
```

```python
def plot_errors_vs_num_iterations(errors):
    plt.plot(errors)
    plt.ylabel("Errors")
    plt.xlabel("Number of iterations")
    plt.show()

```

```python
plot_errors_vs_num_iterations(solutions[1])
```

```python
xaxis
```

```python
plt.figure(figsize=(200,100))

# define range for input
bounds = np.asarray([[-1.0, 1.0], [-1.0, 1.0]])
# sample input range uniformly at 0.1 increments
xaxis = arange(bounds[0,0], bounds[0,1], 0.1)
yaxis = arange(bounds[1,0], bounds[1,1], 0.1)
# create a mesh from the axis
x, y = meshgrid(xaxis, yaxis)
results = objective([x, y])
# create a filled contour plot with 50 levels and jet color scheme
pyplot.contourf(x, y, results, levels=50, cmap='jet')
# plot the sample
xs = [arr[0] for arr in solutions[0]]
ys = [arr[1] for arr in solutions[0]]
pyplot.plot(xs, ys, marker='o', color='w', markersize=100)
# show the plot
```

```python
solutions[0]
```

```python
sols[:,0]
```

```python
xs = [arr[0] for arr in solutions[0]]
```

```python
xs
```

```python
ys = [arr[1] for arr in solutions[0]]
```

```python
ys
```

```python

```
