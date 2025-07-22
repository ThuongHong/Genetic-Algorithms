# Application of Genetic Algorithm in Training a Classification Model

## Motivation

- Genetic Algorithms can be used to find optimized solutions, making them suitable for various Machine Learning problems.
- I wanted to explore how training a model using a **Genetic Algorithm** differs from traditional **Gradient Descent**.
- And of course—just for fun and experimentation.

## Idea

- I use **Softmax Regression** as a reference model for this project.
- **Dataset**: MNIST, where each image is a 28×28 grayscale matrix, flattened into a vector of size 784.
- **Softmax Regression** model includes:
    - **Weight matrix**:
    
      $$
      W \in \mathbb{R}^{C \times d},\quad \text{where } C = 10 \text{ (number of classes)},\quad d = 28 \times 28 = 784
      $$
    
    - **Bias vector**:
    
      $$
      b \in \mathbb{R}^{C} = \mathbb{R}^{10}
      $$

### Genetic Representation

- First, flatten the weight matrix $W$ into a 1D vector $W^* \in \mathbb{R}^{C \times d}$.
- Then, concatenate $W^*$ with the bias vector $b$ to form an **individual chromosome**:

  $$
  I = [W^* \,\|\, b] \in \mathbb{R}^{C \cdot d + C}
  $$

### Training with Genetic Algorithm

- Each individual is evaluated using a **fitness function**, which measures **classification accuracy** on the training set.
- Standard Genetic Algorithm operations are applied:
  - **Selection**
  - **Crossover**
  - **Mutation**
- A `Classifier` class is responsible for decoding the chromosome back into $W$ and $b$, applying the softmax function, and performing prediction.

---

This approach replaces gradient-based optimization with evolutionary search. Through experimenting with this problem, I had the opportunity to gain a deeper understanding of how Genetic Algorithms work, while also reinforcing my knowledge of NumPy and scikit-learn syntax.
