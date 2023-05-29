# Set up
```python
conda create -n jointdiff python=3.7
conda activate jointdiff
pip install pip==21.0.1
pip install -r requirements.txt
pip install torch==1.10.0+cpu torchvision==0.11.0+cpu torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html
```


# Run the examples, run each individual py files such as 
```python
python PJ4.py
python Ball4.py
python pendulum.py
python LK.py
python D6.py
python AttControl.py
```

# Citation
```
@inproceedings{10.1145/3576841.3585919,
author = {Wang, Yixuan and Zhan, Simon and Wang, Zhilu and Huang, Chao and Wang, Zhaoran and Yang, Zhuoran and Zhu, Qi},
title = {Joint Differentiable Optimization and Verification for Certified Reinforcement Learning},
year = {2023},
isbn = {9798400700361},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3576841.3585919},
doi = {10.1145/3576841.3585919},
abstract = {Model-based reinforcement learning has been widely studied for controller synthesis in cyber-physical systems (CPSs). In particular, for safety-critical CPSs, it is important to formally certify system properties (e.g., safety, stability) under the learned RL controller. However, as existing methods typically conduct formal verification after the controller has been learned, it is often difficult to obtain any certificate, even after many iterations between learning and verification. To address this challenge, we propose a framework that jointly conducts reinforcement learning and formal verification by formulating and solving a novel bilevel optimization problem, which is end-to-end differentiable by the gradients from the value function and certificates formulated by linear programs and semi-definite programs. In experiments, our framework is compared with a baseline model-based stochastic value gradient (SVG) method and its extension to solve constrained Markov Decision Processes (CMDPs) for safety. The results demonstrate the significant advantages of our framework in finding feasible controllers with certificates, i.e., Barrier functions and Lyapunov functions that formally ensure system safety and stability, available on Github.},
booktitle = {Proceedings of the ACM/IEEE 14th International Conference on Cyber-Physical Systems (with CPS-IoT Week 2023)},
pages = {132–141},
numpages = {10},
keywords = {stability, lyapunov function, RL, barrier function, safety},
location = {San Antonio, TX, USA},
series = {ICCPS '23}
}
```

Control Animation for the learned controller from our approach. Dashed lines are the target position.
![alt text](AttControl.gif "Control Animation for the learned controller from our approach. Dashed lines are the target position.")