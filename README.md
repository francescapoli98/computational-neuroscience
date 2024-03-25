# Computational neuroscience [CNS]

<img src="https://apre.it/wp-content/uploads/2021/01/logo_uni-pisa.png" width="200" />

The objectives of the CNS course include bio-inspired neural modelling, spiking and reservoir computing neural networks, advanced computational neural models for learning, architectures and learning methods for dynamical/recurrent neural networks for temporal data and the analysis of their properties, the role of computational neuroscience in real-world applications (by case studies).

This repository stores the assignments, based on material from the laboratories' experience, to be delivered as the hands-on part of the exam.

### Lab 1
-  **Assignment:** implementing Spiking Neurons using Izhikevich's Model
    1) Implement the Izhikevich’s model
    2) Develop all the 20 neuro-computational features of biological neurons using the model developed at point _1_ and plot:
        - the resulting membrane potential's time courses into individual figures (one figure for each neuro-computational feature);
        - the phase portraits that correspond to each of the neuro-computational features (one figure for each neuro-computational feature)

Recall that the Izhikevich’s model is described by the following equations:

$𝑑𝑢/𝑑𝑡 = 0. 04 𝑢^2 + 5𝑢 + 140 − 𝑤 + 𝐼$

$𝑑𝑤/𝑑𝑡 = 𝑎(𝑏𝑢 − 𝑤)$

$𝑖𝑓 (𝑢 ≥ 30):$

$𝑢 ← 𝑐; 𝑤 ← 𝑤 + 𝑑$

-  **Bonus assignment 1:**  implementing Liquid State Machines (LSM)
    1) Sunspot task: consists in a next-step prediction (autoregressive, a particular case of transduction) on a time-series consisting in monthly averaged solar sunspots.
-  **Bonus assignment 2:**  implementing Spike-Time-Dependent Plasticity (STDP)
    1)  Use a liquid of Izhikevich neurons and train the coupling of these neurons with a simplified version of the STDP algorithm. 






### Lab 2
-  **Assignment 1:** implementing [...]
-  **Assignment 2:** implementing [...]  
