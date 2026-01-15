## 4WIS Robot Multi-Modal Trajectory Tracker

> Published in *2025 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)*
> DOI: [10.1109/IROS60139.2025.11247526](https://ieeexplore.ieee.org/document/11247526)

---

### Basic Functions

* Implements a kinematic iterator and a multi-modal trajectory generator for 4WIS (Four-Wheel Independent Steering) robots, used to construct reinforcement learning training environments
* Designs DRL-based trajectory trackers with MPC-like preview structural characteristics for different motion modes
* Constructs an autonomous motion mode decision mechanism based on target trajectory characteristics, enabling reasonable switching within the multi-modal motion space

<p align="center"><img src="document/method.jpg" width="90%"></p>

> At present, the code for the kinematic iterator, trajectory generator, and the Ackermann single-mode trajectory tracker has been organized. The remaining modes and decision modules will be gradually open-sourced in the near future.

---

### Prerequisites

> The versions of the dependent libraries are not strictly fixed; the following configuration is provided only as a reference for the author’s current experimental and development environment.

```
pip install gym==0.26.2
pip install matplotlib==3.10.8
pip install numpy==2.4.1
pip install stable_baselines3==2.6.0
pip install torch==2.7.0+cu128 --extra-index-url https://download.pytorch.org/whl/cu128
```

---

### Code Description

* `utils/Trajectory_Generater.py`: Kinematic iterator and random reference trajectory generation module for the 4WIS robot

* `utils/Trajectory_Transfer.py`: Auxiliary utility functions used during trajectory generation and format conversion

* `env/mode1_env.py`: The tracker reinforcement learning environment, currently only supporting Ackerman mode, will be updated soon.

* `Lower_update/Lower_update_trainer.py`: Training code for a single-mode trajectory tracker based on DNN-DRL

* `Lower_update/Lower_update_predicter.py`: Testing code for a single-mode trajectory tracker based on DNN-DRL

* `Lower_update/Lower_update_LSTM_trainer.py`: Training code for a single-mode trajectory tracker based on LSTM-DRL

* `Lower_update/Lower_update_LSTM_predicter.py`: Testing code for a single-mode trajectory tracker based on LSTM-DRL

---

### Discussion and Notes

* The design of the state space and action space adopts an MPC-like preview optimization modeling concept, endowing the tracker with a certain level of foresight, allowing it to adjust control strategies in advance according to upcoming trajectory changes

* We added wheel angle and speed calculations based on the Ackerman steering principle to the paper, which can reduce wear on robot tires and increase execution efficiency in actual deployment.

* Compared with a pure DNN structure, the introduction of LSTM shows certain improvements in both trajectory tracking accuracy and stability, as illustrated by the quantitative comparison shown below

<p align="center"><img src="document/compare.png" width="80%"></p>

* It should be noted that the LSTM structure significantly increases the difficulty of convergence during training. Therefore, in practical training, a DNN-based tracker can be used as a prior or guiding model to improve stability and convergence speed in the early training stage. Relevant implementation details can be found in `Lower_update/Lower_update_LSTM_trainer.py`.

* In our subsequent work, the trajectory planner will provide trajectories that include modal information; therefore, the importance of the modal decision module will decrease slightly.

---

### References

If this code repository is helpful to your research, please cite the following paper:

```bibtex
@INPROCEEDINGS{11247526,
  author={Bao, Runjiao and Xu, Yongkang and Zhang, Lin and Yuan, Haoyu and Si, Jinge and Wang, Shoukun and Niu, Tianwei},
  booktitle={2025 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  title={Deep Reinforcement Learning-Based Trajectory Tracking Framework for 4WS Robots Considering Switch of Steering Modes},
  year={2025},
  pages={3792--3799},
  doi={10.1109/IROS60139.2025.11247526}
}
```

<details>
<summary>点击查看中文版</summary>

## 4WIS 机器人多模态轨迹跟踪器

> 发表于 *2025 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)*
> DOI：[10.1109/IROS60139.2025.11247526](https://ieeexplore.ieee.org/document/11247526)

---

### 基本功能

* 实现了面向 4WIS（Four-Wheel Independent Steering）机器人的运动学迭代器与多模态轨迹生成器，用于构建强化学习训练环境
* 针对不同运动模态，设计了具有 MPC 前视结构特性的 DRL-based 轨迹跟踪器
* 构建了基于目标轨迹特性的自主运动模态决策机制，用于在多模态运动空间中进行合理切换

<p align="center"><img src="document/method.jpg" width="90%"></p>

> 目前已完成运动学迭代器、轨迹生成器以及 Ackermann 单模态轨迹跟踪器的代码整理，其余模态与决策模块将于近日陆续开源

---

### 基本前置

> 各依赖库版本并非严格绑定，以下仅为作者当前的实验与开发环境配置参考

```
pip install gym==0.26.2
pip install matplotlib==3.10.8
pip install numpy==2.4.1
pip install stable_baselines3==2.6.0
pip install torch==2.7.0+cu128 --extra-index-url https://download.pytorch.org/whl/cu128
```

---

### 代码描述

* `utils/Trajectory_Generater.py`: 4WIS 机器人运动学迭代器与随机参考轨迹生成模块

* `utils/Trajectory_Transfer.py`: 轨迹生成与格式转换过程中使用的辅助工具函数

* `env/mode1_env.py`: 跟踪器强化学习环境，目前仅支持阿克曼模式，将于近日更新

* `Lower_update/Lower_update_trainer.py`: 基于 DNN-DRL 的单模态轨迹跟踪器训练代码

* `Lower_update/Lower_update_predicter.py`: 基于 DNN-DRL 的单模态轨迹跟踪器测试代码

* `Lower_update/Lower_update_LSTM_trainer.py`: 基于 LSTM-DRL 的单模态轨迹跟踪器训练代码

* `Lower_update/Lower_update_LSTM_predicter.py`: 基于 LSTM-DRL 的单模态轨迹跟踪器测试代码

---

### 讨论与记录

* 状态空间与动作空间设计采用类似 MPC 前视优化的建模思想，使得跟踪器具备一定的前瞻能力，能够针对前方轨迹变化提前调整控制策略

* 我们在论文的基础上增加了基于阿克曼转向原理的车轮转角与速度解算，在实际部署时可以降低对机器人轮胎的磨损并增加执行效率

* 相较于纯 DNN 结构，引入 LSTM 后在轨迹跟踪精度与稳定性方面均表现出一定提升，其定量对比如下图所示

<p align="center"><img src="document/compare.png" width="80%"></p>

* 需要注意的是，LSTM 结构在训练阶段的收敛难度显著增加。因此，在实际训练中可采用 DNN 跟踪器作为先验或指导模型，以提升训练初期的稳定性与收敛速度,相关实现细节可参考 `Lower_update/Lower_update_LSTM_trainer.py`

* 在我们后续的工作中，轨迹规划器所给出的轨迹将包含模态信息，因此模态决策模块的重要性将略微下降
---

### 相关引用

如果本代码仓库对您的研究工作有所帮助，请引用以下论文：

```bibtex
@INPROCEEDINGS{11247526,
  author={Bao, Runjiao and Xu, Yongkang and Zhang, Lin and Yuan, Haoyu and Si, Jinge and Wang, Shoukun and Niu, Tianwei},
  booktitle={2025 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  title={Deep Reinforcement Learning-Based Trajectory Tracking Framework for 4WS Robots Considering Switch of Steering Modes},
  year={2025},
  pages={3792--3799},
  doi={10.1109/IROS60139.2025.11247526}
}
```
</details>
