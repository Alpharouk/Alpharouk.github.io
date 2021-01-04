---
title: "Monte Calos"
date: 2021-01-04
tags: [LSTM, Con1D, PCA, T-SNE, Word Embedding]
header:
 image : "images/cover7.jpg"

  
excerpt: "Natural Language Processing, LSTM, Con1D, PCA, T-SNE, Word Embedding, Tensorflow, Keras, Embedding Projector"
mathjax: "true"
---
A Monte Carlo method is a technique that uses random numbers and probability to solve complex problems. The Monte Carlo simulation, or probability simulation, is a technique used to understand the impact of risk and uncertainty in financial sectors, project management, costs, and other forecasting machine learning models.
Risk analysis is part of almost every decision we make, as we constantly face uncertainty, ambiguity, and variability in our lives. Moreover, even though we have unprecedented access to information, we cannot accurately predict the future.
The Monte Carlo simulation allows us to see all the possible outcomes of our decisions and assess risk impact, in consequence allowing better decision making under uncertainty.
In this article, we will go through 2 differents examples to understand the Monte Carlo Simulation method.



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


![png](/images/monte_carlo/Copie_de_Copie_de_monte_carloos_2_0.png)

