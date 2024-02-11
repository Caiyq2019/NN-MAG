<div align="center">
  <img width="500px" src="https://github.com/Caiyq2019/NN-MAG/blob/ad4c2cff5ade3bc209df62c28ca67bf96ff424a8/figs/magsimu.png"/>
</div>



## Introduction to NeuralMAG Project

The NeuralMAG Project is an open-source neural network framework designed for micromagnetic simulation. It employs a multi-scale U-Net neural network model to learn the physical relationship between magnetization states and demagnetizing fields, effectively extending traditional micromagnetic simulations into the realm of advanced deep learning frameworks. This approach fully leverages the cutting-edge technologies available on deep learning platforms.

![alt text](framework.png)

The project finds wide application in the study of magnetic materials among other areas.

## Capabilities of NeuralMAG

- **Integration with PyTorch Platform**: NeuralMAG integrates MAG micromagnetic simulations into the PyTorch framework, utilizing the extensive parallelization capabilities of GPUs for enhanced performance.
- **Cross-Scale Neural Network Simulation**: By employing a multi-scale Unet neural network architecture, the system can efficiently simulate magnetic material systems on a large scale.
- **Generalization Across Material Systems**: The model has been designed to generalize effectively to previously unseen material systems, thereby broadening the scope for exploration of new materials.
- **Versatility in Micromagnetic Simulation Tasks**: NeuralMAG is adept at performing a wide array of micromagnetic simulations, including but not limited to predicting magnetic ground states, simulating hysteresis loops, and accommodating arbitrary geometric shapes in simulations.
- **Utilization of Advanced Deep Learning Optimization Techniques**: The framework incorporates the latest advancements in model compression and acceleration optimization technologies that are prevalent in modern deep learning platforms.



## Getting Started

### Installation Requirements

Ensure your system meets the following prerequisites:

- **Python Version**: 3.9.0
- **PyTorch Version**: 2.0.1 with CUDA 11.7 support
- **Additional Dependencies**: Refer to the `requirements.txt` file for a complete list of required libraries and their versions.

### Usage Instructions

#### Example Tasks (`./egs`)

This directory houses sample tasks, including:
- **NMI**: Replicates the main experimental results presented in the manuscript.
- **Demo**: Contains code for quick experimentation and familiarization with the tool. Pre-trained Unet model parameters used in the manuscript are located in `./egs/NMI/ckpt/k16`.

#### Libraries (`./libs`)

Contains the core libraries of the project:
- Traditional micromagnetic simulation frameworks based on RK4 and LLG equations.
- Unet neural network model architecture.
- Auxiliary functions pertinent to this project.

#### Utilities (`./utils`)

This directory includes scripts for data generation, essential for training the Unet model:
- Scripts generate (m, Hd) pair data required for Unet training.

By following these instructions, users can set up the necessary environment to run simulations, replicate study findings, or train the Unet model with custom data.



## Example Execution

### Running MH Simulations

#### Quick Trial

To swiftly simulate the MH curve of a magnetic thin film material, such as a material shaped as a triangle with two layers of thickness, having magnetic properties specified by `{ --Ms 1000, --Ax 0.5e-6, --Ku 0.0 }`, execute the following script:

```bash
./egs/demo/MH/runMH.sh
```

This script will furnish a comparative analysis of the results obtained via FFT and Unet simulations.

#### Replication of Published Results

For replicating the MH experimental results presented in the manuscript, refer to the script below:

```bash
./egs/NMI/MH_evaluate/runMH.sh
```

This script allows for the adjustment of the film's dimension through the `--width` parameter. It is configured with 18 distinct sets of magnetic property parameters:

- `--Ms`: {1200, 1000, 800, 600, 400}
- `--Ku`: {1e5, 2e5, 3e5, 4e5}
- `--Ax`: {0.7e-6, 0.6e-6, 0.4e-6, 0.3e-6}

The `--mask` parameter is designated for determining the shape of the film, inclusive of:

- Triangular films
- Films with a central hole
- Random polygonal films


#### Sample Result Images

Triangular film MH result | Film with a central hole MH result | Random polygonal film MH result
:-------------------------:|:-----------------------------------:|:---------------------------------:
![Triangular film MH result](MH_triangle.png) | ![Film with a central hole MH result](MH_hole.png) | ![Random polygonal film MH result](MH_square.png)




### Running Vortex Simulations

#### Quick Trial

To experiment with the iterative process based on the LLG (Landau-Lifshitz-Gilbert) dynamical equation starting from a random initial state, execute the script:

```bash
./egs/demo/vortex/run.sh
```

This will initiate a simulation comparing FFT and Unet for 10 sets of default material magnetic parameters. The `--pre_core` option specifies the initial number of vortices predicted by Unet.

#### Replication of Published Results

For a detailed replication of the vortex simulation results presented in the manuscript, refer to the following script:

```bash
./egs/NMI/vortex_evaluate/run.sh
```

This script offers a comprehensive comparison of the Unet model's prediction accuracy across different initial vortex quantities. For each set of test conditions, 100 dynamical experiments are compared.

#### Sample Result Images

Vortex simulation with varying parameters | Vortex simulation with different shapes | Vortex simulation on a square lattice
:-----------------------------------------:|:---------------------------------------:|:--------------------------------------:
![Vortex simulation with varying parameters](vortex_rdparam.png) | ![Vortex simulation with different shapes](vortex_rdshape.png) | ![Vortex simulation on a square lattice](vortex_square.png)




### Running Speed Evaluation

To compare the computational costs of micromagnetic simulations across different frameworks, you can execute the script:

```bash
./egs/NMI/speed_evaluate/run.sh
```

This allows for the configuration of comparative experiments for various film sizes via the `--width` option. The script conducts a comparative analysis of computational overheads across three frameworks:

- The conventional framework based on FFT
- The deep neural network-based Unet framework
- The Unet model accelerated by TensorRT

This evaluation provides insights into the efficiency and performance improvements offered by integrating deep learning and acceleration technologies into micromagnetic simulation processes.



### Data Generation and Model Training

#### Data Generation

The training data used in the manuscript can be generated with:

```bash
./utils/run.sh
```

This script automatically prepares data in four sizes: 32, 64, 96, and 128. Data for sizes 32, 64, and 96 are utilized for cross-scale model training, while size 128 data is employed for evaluating cross-scale generalization. In addition to the default square shape, random shape masks are generated for data augmentation. Furthermore, data with random masks are subjected to two different magnitudes of external magnetic fields, meaning each size of data is under three different conditions, with 100 samples per condition.

A data visualization tool is also provided:

```bash
./utils/visualize_data.py
```

Running this will yield visualizations of the generated data, including magnetic vector maps, RGB images, and numerical statistics histograms.

Data Visualization Samples |  | 
:-------------------------:|:-------------------------:
![Magnetic vector map](Hds_vector.png) | ![Magnetic field histogram](Hd_hist.png)
![RGB image of spins](Spins_rgb-1.png) | ![Spin vector map](Spins_vector-1.png)

#### Model Training

Once the data is prepared, you can commence training your own Unet model by executing:

```bash
./egs/demo/train/run_train.sh
```

Adjust the training dataset size with `--ntrain`, representing the data volume for each size. `--ntest` specifies the size of the test dataset. Training hyperparameters such as `--batch-size`, `--lr`, and `--epochs` can be adjusted, with the default settings capable of reproducing the results presented in the manuscript. The `--kc` and `--inch` parameters define the network structure of the model; they should not be altered to maintain the model size. Information on intermediate models, convergence, and evaluations will be automatically saved during training.

To utilize your trained model for the micromagnetic simulation tasks mentioned above, simply replace the model file in `./egs/NMI/ckpt/k16/model.pt` with your trained model.

Model Training Visualizations |  | 
:-----------------------------:|:-----------------------------:
![Training loss example](loss_ex1.0-1.png) | ![Layer 1 RGB output after training](epoch820_L1_rgb.png)
![Layer 2 Vector output after training](epoch820_L2_vec.png) | ![Layer 1 Vector output after training](epoch820_L1_vec.png)



### standproblem



## Documentation

- [技术文档](LINK_TO_YOUR_DOCUMENTATION)
- [API参考](LINK_TO_YOUR_DOCUMENTATION)


## License

本项目在Apache License (Version 2.0)下分发。

详情请见 [Apache License](LINK_TO_YOUR_LICENSE).


### Report an Issue

- 技术或文档问题，请在 [GitHub Issues](YOUR_GITHUB_ISSUES_LINK) 提交。
- 安全问题，请发送邮件至 <a href =

