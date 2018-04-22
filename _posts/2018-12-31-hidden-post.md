---
title: "Hidden Post"
hidden: true
last_modified_at:
---

This post has YAML Front Matter of `hidden: true` and should not appear in `paginator.posts`.






1.
Salah satu definisi yang menarik : “A computer
program is said to learn from experience E with respect to some class of tasks T
and performance measure P , if its performance at tasks in T , as measured by P ,
improves with experience E.” -Mitchell(1997)

Contoh task : Klasifikasi, Regresi, Sintesis
Contoh performance measure : AKurasi dari klasifikasi
Contoh experience : proses pembelajaran (data dan algoritma yang digunakan)

2.
Machine Learning Tutorial 
disp.ee.ntu.edu

3. Meme Interpretasi coefficient di Twitter

4. Target bener2 tidak diketahui persamaanya, kalau model g diketahui tapi hanya 
koefisien persamaan yg tidak diketahui -> permasalahan cure fitting



---
title: Machine Learning
published: September 30, 2013
excerpt: Machine Learning concepts
comments: off
toc: left
push: off
---

I've been wanting to learn about the subject of machine learning for a while now. I'm familiar with some basic concepts, as well as reinforcement learning. What follows are notes on my attempt to comprehend the subject. The primary learning resource I'm using is Cal Tech's CS 1156 on edX, with supplementary material from Stanford's CS 229 on Coursera.

I pushed my code for the programming assignments for this class to [github](https://github.com/blaenk/learning-from-data).

* toc

# Learning Problem

The essence of machine learning:

1. pattern exists
2. cannot pin it down mathematically
3. have data on it to learn from

A movie recommender system might be modeled such that there are a set of factors of varying degrees of likability for the viewer, and varying degrees of presence in a given movie. These two sets of factors combine to produce a projected viewer rating for that given movie.

The machine learning aspect would take an actual user rating, and given two contributing-factor vectors for the movie and viewer full of random factors, it would modify each until a rating similar to the actual user rating is produced.

## Learning Components

* **Input**: $x = (x_1, x_2, \dots, x_n)$
    * feature vector
* **Output**: $y$
    * result given the input
* **Target Function**: $f\colon \cal {X} \to \cal {Y}$
    * The ideal function for determining the result, unknown, reason for learning
* **Data**: $(x_1, y_1), (x_2, y_2), \dots, (x_n, y_n)$
    * Historical records of actual inputs and their results
* **Hypothesis**: $g\colon \cal {X} \to \cal {Y}$
    * Result of learning, $g \approx f$
* **Learning Algorithm**: $\cal A$
    * The machine learning algorithm
* **Hypothesis Set**: $\cal H$
    * The set of candidate hypotheses where $g \in \cal H$

Together, $\cal A$ and $\cal H$ are known as the _learning model_.

## Perceptrons

### Model {#perceptron-model}

Given input $\def \feature {\mathbf x} \feature = (x_1, x_2, \dots, x_n)$:

$$
\begin{align}
\text {approve} &\colon \sum_{i=1}^{d} w_i x_i > \text {threshold} \\
\text {deny}    &\colon \sum_{i=1}^{d} w_i x_i < \text {threshold}
\end{align}
$$

This can be expressed as a linear formula $h \in \cal H$:

$$
\newcommand{sign}{\operatorname{sign}}
h(x) = \sign \left( \left( \sum_{i=1}^{d} w_i x_i \right) - \text {threshold} \right)
$$

The factors that are varied to determine the hypothesis function are the weights and threshold.

The threshold can be represented as a weight $w_0$ if an artificial constant $x_0$ is added to the feature vector where $x_0 = 1$. This simplifies the formula to:

$$ h(x) = \sign \left( \sum_{i=0}^{d} w_i x_i \right) $$

This operation is the same as the dot product of the two vectors:

$$
\def \weight {\mathbf w}
\def \weightT {\weight^\intercal}
h(x) = \sign(\weight \bullet \feature)
$$

### Learning Algorithm {#perceptron-learning-algorithm}

**Given the perceptron**:

$$ h(x) = \sign(\weight \bullet \feature) $$

**Given the training set**:

$$\feature = (x_1, x_2, \dots, x_n)$$

**Pick a misclassified point**. A point is misclassified if the perceptron's output doesn't match the recorded data's output given an input vector:

$$ \sign(w \bullet x_n) \not= y_n $$

**Update the weight vector**: The algorithm must correct for this by updating the weight vector.

$$ \weight' \gets \weight + y_n x_n $$

Visualizing $\vec w$ and $\vec x$, it's apparent that the perceptron is equivalent to the dot product, which is equivalent to $\cos \theta$ where $\theta$ is the angle between $\vec w$ and $\vec x$. Given this, $\vec w$ is updated depending on what the intended result $y_n$ is.

For example, if the perceptron modeled the result to be $-1$, then $\theta > 90^\circ$. However, if the intended result $y_n = +1$, then there is a mismatch, so the weight vector is "nudged" in the correct direction so that $\theta < 90^\circ$ by adding it to $x_n$.

## Types of Learning

* **Supervised learning**: when the input/output data is provided
* **Unsupervised learning**: only the input is provided; "unlabeled data"
* **Reinforcement learning**: input and _some_ output is provided

# Feasibility

It seems as if it isn't feasible to learn an unknown function because the function can assume any value outside of the data available to us.

To understand why it is indeed possible, consider a probabilistic example. Given a bin full of marbles that are either **red** or **green**:

$$
\begin{align}
&P \left( \text {picking a red marble} \right) = \mu \\
\\
&P \left( \text {picking a green marble} \right) = 1 - \mu \\
\end{align}
$$

If the value of $\mu$ is unknown, and we pick $N$ marbles independently, then:

$$ \text {frequency of red marbles in sample} = \nu $$

The question is: does $\nu$ say anything about $\mu$? It might appear that it **doesn't**. For example, if there are 100 marbles in the bin and only 10 of them are red, just because we happen to take out all 10 ($\nu  = 1$) doesn't mean that that sample is representative of the distribution in the bin ($\mu = 1$).

However, in a **big sample**, it's more probable that $\nu$ is close to $\mu$, that is, they are within $\epsilon$ of each other. This can be formally expressed as _Hoeffding's Inequality_:

$$
P \left( \left| \nu - \mu \right| \gt \epsilon \right) \leq 2e^{- 2\epsilon^2 N}
$$

$\nu$ is generally varied, and $\mu$ is a constant.

As is apparent from the inequality, if we choose a very small $\epsilon$ value, it has the effect of setting the right-hand side to near 1, thus rendering the effort pointless since we already knew that the probability would be $\leq 1$. **Therefore**, if we want a smaller $\epsilon$, we will have to increase the input size $N$ to compensate.

The appeal of Hoeffding's Inequality is that it is valid for any $N \in \mathbb {Z}^+$ and any $\epsilon > 0$. What we're trying to say with the inequality is that $\nu \approx \mu$ which means that $\mu \approx \nu$.

This relates to learning in the following way. In the bin, $\mu$ is unknown, but in learning the unknown is the target function $f \colon \mathcal {X} \to \mathcal {Y}$. Now think of the bin as the input space, where every marble is a point $x \in \mathcal {X}$ such as a credit application. As a result, the bin is really $\mathcal {X}$, where the marbles are of different colors such as green and gray, where:

* **green** represents that the hypothesis got it right, that is $h(x) = f(x)$
* **red** represents that the hypothesis got it wrong, that is $h(x) \not= f(x)$

However, in creating this analogy from the bin example to the learning model, the bin example has a probability component in picking a marble from the bin. This must be mapped to the learning model as well, in the form of introducing a probability distribution $P$, where $P$ is not restricted over $\mathcal {X}$, and $P$ doesn't have to be known. It is then assumed that the probability is used to generate the input data points.

The problem so far is that the hypothesis $h$ is fixed, and for a given $h$, $\nu$ generalizes to $\mu$, which ends up being a _verification_ of $h$, not learning.

Instead, to make it a learning process, then there needs to be no guarantee that $\nu$ will be small, and we need to choose from multiple $h$'s. To generalize the bin model to more than one hypothesis, we can use multiple bins. Out of the many bins that were created, the hypothesis responsible for the bin with the smallest $\mu$---the fraction of red marbles in the bin---is chosen.

## Notation {#notation-for-learning}

Both $\mu$ and $\nu$ depend on which hypothesis $h$:

$\nu$ is the error _in sample_, which is denoted by $\def \insample {E_{\text {in}}} \insample(h)$

$\mu$ is the error _out of sample_, which is denoted by $\def \outsample {E_{\text {out}}} \outsample(h)$

For clarification, if something performs well "out of sample" then it's likely that learning actually took place. This notation can be used to modify Hoeffding's Inequality:

$$
P \left( \left|\insample(h) - \outsample(h)\right| \gt \epsilon \right) \leq 2e^{- 2\epsilon^2 N}
$$

The problem now is that Hoeffding's Inequality doesn't apply to multiple bins. To account for multiple bins, the inequality can be modified to be:

$$
\begin{align*}
P \left( \left|\insample(g) - \outsample(g)\right| \gt \epsilon \right) &\leq
P \begin{aligned}[t]
              \left( \vphantom {\epsilon} \right. &\hphantom {\text {or}}\
                \left|\insample(h_1) - \outsample(h_1)\right| \gt \epsilon \\
              &\text {or}\ \left|\insample(h_2) - \outsample(h_2)\right| \gt \epsilon \\
              &\dots \\
              &\left. \vphantom{\insample(h_1)} \text {or}\ \left|\insample(h_M) - \outsample(h_M)\right| \gt \epsilon \right)
            \end{aligned} \\
&\leq \sum_{m=1}^M P \left( \left| \insample(h_m) - \outsample(h_m) \right| > \epsilon \right) \\
&\leq \sum_{m=1}^M 2e^{- 2\epsilon^2 N} \\
P \left( \left|\insample(g) - \outsample(g)\right| \gt \epsilon \right) &\leq 2Me^{- 2\epsilon^2 N}
\end{align*}
$$

# Linear Model

## Input Representation

If we want to develop a system that can detect hand-written numbers, the input can be different examples of hand-written digits. However, if an example digit is a 16x16 bitmap, then that corresponds to an array of 256 real numbers, which becomes 257 when $x_0$ is added for the threshold. This means the problem is operating in 257 dimensions.

Instead, the input can be represented in just three dimensions: $x_0$, intensity, and symmetry. The intensity corresponds to how many black pixels exist in the example, and the symmetry is a measure of how symmetric the digit is along the x and y axes. See slide 5 to see a plot of the data using these features.

## Pocket Algorithm

The PLA would never converge on non-linearly separable data, so it's common practice to forcefully terminate it after a certain number of iterations. However, this has the consequence that the hypothesis function ends up being whatever the result of the last iteration was. This is a problem because it could be that a better hypothesis function with lower in-sample error $\insample$ was discovered in a previous iteration.

The Pocket algorithm is a simple modification to the PLA which simply keeps track of the hypothesis function with the least in-sample error $\insample$. When this is combined with forceful termination after a certain number of iterations, PLA becomes usable with non-linearly separable data.

## Linear Regre