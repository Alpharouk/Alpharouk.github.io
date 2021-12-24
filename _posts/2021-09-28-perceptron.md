---
title: "The evolution of Neural Network’s learning"
date: 2021-11-12
tags: [Deep Learning, Maths, Neural Network]
header:
 image : "images/richard3.jpg"

  
excerpt: From Perceptron’s learning procedure to Backpropagation
mathjax: "true"
---
In this article we will see how the first generation of neural networks used to learn weights and biases and how Backpropagation made the learning possible for bigger networks.
![png](/images/perceptron/NN archi.jpg)

## Perceptrons
# What is a Perceptron ?
It’s a simple neuron that takes as inputs a set of features ‘xi’ and outputs ‘1’ if the weighted sum of every feature ‘xi’ by its weight ‘ωi’ is greater than a specific threshold ‘θ’, otherwise it outputs ‘0’ (a threshold is equivalent to having a weight on an extra input that always has 1 as input, it’s called biais)

![png](/images/perceptron/formule 1.jpg)
```
import math
import random
import numpy as np
import matplotlib.pyplot as plt

def coin_flip():
  return random.randint(0,1)
  
def monte_carlo(N):
  l=0
  result=[]
  for i in range(N):
    l+=coin_flip()
    prob=l/(i+1)
    result.append(prob)
  return result

N=10000
plt.axhline(0.5,color='green',linestyle='-')
plt.plot(monte_carlo(N))
plt.show()

```


![png](/images/monte_carlo/Copie_de_Copie_de_monte_carloos_1_0.png)


## Monty Hall problem

Suppose you are on a game show, and you have the choice of picking one of three doors: Behind one door is a car; behind the other doors, goats. You pick a door, let’s say door 1, and the host, who knows what’s behind the doors, opens another door, say door 3, which has a goat. The host then asks you: do you want to stick with your choice or choose another door?

Initially, for all three gates, the probability (P) of getting the car is the same (P = 1/3).

![png](/images/monte_carlo/monty_hall.png)

Now assume that the contestant chooses door 1. Next, the host opens the third door, which has a goat. Next, the host asks the contestant if he/she wants to switch the doors?

![png](/images/monte_carlo/monty_hall_2.png)

Let's take a look at each case :
- Not swapping : it means that we will stick with our initial choice which has the probability of 1/3 being a car and 2/3 being a goat.
- Swapping : in this case, the probability of picking a car after swapping is equal to the probability of picking a goat before swapping which is equal to 2/3.

So the best choice would be to always swap doors as it gives you the probability of 2/3 of picking a car.

Now we are going to use the Monte Carlo Method to perform this test case many times and find out its probabilities in an experimental way.


```
def monty_hall(n):

  doors=['goat','car','goat']
  won_with_swap=0
  swap_prob=[]
  won_with_not_swap=0
  notswap_prob=[]
  
  for i in range(n):
    random.shuffle(doors)
    choice=random.randrange(2)
    
    if doors[choice]!='car':
      won_with_swap+=1
      swap_prob.append(won_with_swap/(i+1))

    else:
      won_with_not_swap+=1 
      notswap_prob.append(won_with_not_swap/(i+1))
      
    plt.plot(swap_prob)
    plt.plot(notswap_prob)
    
  plt.axhline(0.33,color='green',linestyle='-')
  plt.axhline(0.66,color='red',linestyle='-')
  
monty_hall(2000)
```

