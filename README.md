# ML4SCI_test

Submission for GSoC_2023_QMLHEP_Tasks.

---
TASK I
See Task_I.ipynb.
---
TASK II
In the folder TASK II. There are two graph neural networks aim to solve the quark/gluon jet classification problem.
ParticleNet uses the same architecture as the ParticleNet paper arxiv.org/abs/1902.08570 using Tensorflow.
LorentzNet uses the same architecture as in arxiv.org/abs/2201.08187 the using Pytorch.
I did not have enough time or computation resource to train both networks to completion.
Based on the iterations that I have done, it seems the Lorentz equivariant graph network converges faster.
I limited use of pid information in the data set when constructing graph information. I think it is more robust to only use kinematic information of the jet.
The performance would suffer from mistag rate of particles if we use a lot of information coming from particle identification.
4-momenta, relative angular information, and mass were used to learn edges.
---
TASK III
See Task_III.pdf
---
TASK V
See Task_V.ipynb
A simple graph representation of a Network that has a learnable Ising model was implemented.
---
TASK VIII
See Task_VIII.ipynb
For this specific Task, a QNN without equivariant structure seems to be easier to train than a fully equivariant QNN.
After training, equivariant QNN produces a result that respects input data symmetry. With slightly better accuracy and less number of parameters.
The best parameters for equivariant QNN is also in the repository. 
