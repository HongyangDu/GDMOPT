# Beyond Deep Reinforcement Learning: A Tutorial on Generative Diffusion Models in Network Optimization

Generative Diffusion Models (GDMs) have emerged as a transformative force in the realm of Generative Artificial Intelligence (GAI), demonstrating their versatility and efficacy across a variety of applications. The ability to model complex data distributions and generate high-quality samples has made GDMs particularly effective in tasks such as image generation and reinforcement learning. Furthermore, their iterative nature, which involves a series of noise addition and denoising steps, is a powerful and unique approach to learning and generating data.

This repository contains the code accompanying the paper 

> **"Beyond Deep Reinforcement Learning: A Tutorial on Generative Diffusion Models in Network Optimization"**

Authored by *Hongyang Du, Ruichen Zhang, Yinqiu Liu, Jiacheng Wang, Yijing Lin, Zonghang Li, Dusit Niyato, Jiawen Kang, Zehui Xiong, Shuguang Cui, Bo Ai, Haibo Zhou, and Dong In Kim*, submitted to *IEEE Communications Surveys & Tutorials*.

The paper can be found at [ArXiv](https://arxiv.org/abs/2308.05384).

## ⚡ Structure of Our Tutorial
<img src="images/0.jpg" width = "100%">

We initiate our discussion with the foundational knowledge of GDM and the motivation behind their applications in network optimization. This is followed by exploring GDM’s wide applications and fundamental principles and a comprehensive tutorial outlining the steps for using GDM in network optimization. In the context of intelligent networks, we study the impact of GDM on algorithms, e.g., **Deep Reinforcement Learning (DRL)**, and its implications for key scenarios, e.g., **incentive mechanism design**, **Semantic Communications(SemCom)**, **Internet of Vehicles (IoV) networks**, channel estimation, error correction coding, and channel denoising. We conclude our tutorial by discussing potential future research directions and summarizing the key contributions.

## ⚡ Network Optimization via Generative Diffusion Models

<img src="images/1.png" width = "100%">

GDM training approaches with and without an expert dataset. **Part A** illustrates the GDM training scenario when an expert database is accessible. The process learns from the GDM applications in the image domain: the optimal solution is retrieved from the expert database upon observing an environmental condition, followed by the GDM learning to replicate this optimal solution through forward diffusion and reverse denoising process. **Part B** presents the scenario where no expert database exists. In this case, GDM, with the assistance of a jointly trained solution evaluation network, learns to generate the optimal solution for a given environmental condition by actively exploring the unknown environment.

---

## 🔧 Tutorial with an Example

In this part, we representatively formulate an optimization problem in a wireless network and show a step-bystep tutorial to solve it by using GDMs.

Consider a wireless communication network where a base station with total power *P_T* serves a set of users over multiple orthogonal channels. The objective is to **maximize the sum rate** of all channels by optimally allocating power among the channels. Let *g_n* denote the channel gain for the *n_th* channel and *p_n* denote the power allocated to that channel. The sum rate of all *M* orthogonal channels is given by the sum of their individual rates. Let the noise level be set as *1* without loss of generality for the analysis. The optimization goal is to find the power allocation scheme \{*p_1*, ..., *p_M*\} that maximizes the sum rate *C* under the power budget and the non-negativity constraints as:

<img src="images/3.png" width = "30%">

The dynamic nature of the wireless environment presents a significant challenge, as the values of the channel gains, denoted as \(\left\{ {{g_1}, \ldots,{g_M}} \right\}\), can fluctuate within a range. This variability is illustrated in Fig.~\ref{fig:three_figures}, which depicts the sum rate values for different power allocation schemes and channel gains when \(M = 3\). It is evident that changes in channel conditions can significantly impact the optimal power allocation scheme.


To create a new conda environment, execute the following command:

```bash
conda create --name gdmopt python==3.8
```

## ⚡ Activate Environment

Activate the created environment with:

```bash
conda activate gdmopt
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