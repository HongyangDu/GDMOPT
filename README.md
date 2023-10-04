# Beyond Deep Reinforcement Learning: A Tutorial on Generative Diffusion Models in Network Optimization

This repository contains the code accompanying the paper 

> **"Beyond Deep Reinforcement Learning: A Tutorial on Generative Diffusion Models in Network Optimization"**

Authored by Hongyang Du, Ruichen Zhang, Yinqiu Liu, Jiacheng Wang, Yijing Lin, Zonghang Li, Dusit Niyato, Jiawen Kang, Zehui Xiong, Shuguang Cui, Bo Ai, Haibo Zhou, and Dong In Kim, submitted to IEEE COMST.

The paper can be found at [ArXiv](https://arxiv.org/abs/2308.05384).

![Model](images/1.png)
GDM training approaches with and without an expert dataset. **Part A** illustrates the GDM training scenario when an expert database is accessible. The process learns from the GDM applications in the image domain: the optimal solution is retrieved from the expert database upon observing an environmental condition, followed by the GDM learning to replicate this optimal solution through forward diffusion and reverse denoising process. **Part B** presents the scenario where no expert database exists. In this case, GDM, with the assistance of a jointly trained solution evaluation network, learns to generate the optimal solution for a given environmental condition by actively exploring the unknown environment.

---
## 🔧 Environment Setup

To create a new conda environment, execute the following command:

```bash
conda create --name diffopt python==3.8
```

## ⚡Activate Environment

Activate the created environment with:

```bash
conda activate diffopt
```

## 📦 Install Required Packages

The following package can be installed using pip:

```bash
pip install tianshou==0.4.11
```

## 🏃‍♀️ Run the Program



Run `main.py` in the file `Main` to start the program.

## 🔍 Check the results

---

## Citation

```bibtex
@article{du2023beyond,
  title={Beyond Deep Reinforcement Learning: A Tutorial on Generative Diffusion Models in Network Optimization},
  author={Authors},
  journal={},
  year={2023},
  publisher={IEEE}
}
```