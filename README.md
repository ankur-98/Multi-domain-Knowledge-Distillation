# Knowledge Distillation of Multi Domain Classification
#### A PyTorch implementation
## Teaching a student from multiple domain expert teachers

## Assignment

__Time Limit:__ 1 week.

### [Assignment](https://github.com/ankur-98/Multi-domain-Knowledge-Distillation/blob/master/MiniProject.pdf) | [Report](https://github.com/ankur-98/Multi-domain-Knowledge-Distillation/blob/master/Report.pdf)

There are 3 primary tasks of the assignment.
* Train a model on the mixed domain
* Train domain expert teachers and using the distillation loss train the student.
* Improve results by suggesting some techniques.

The following improvement techniques have been adopted:
* Early stopping teacher and student model based on least validation loss.
* Alpha hyperparameter tunning.
* Knowledge distillation annealing.

A few basic suggested methods not implemented due to hardware limitations and tuning time required:
* Adding batchnormalization to the fully connected layers before ReLU activation.
* Changing Adam optimizer to AdamW.
* Using cyclic learning rate schedulers for super convergence.

The metric of evaluation is accuracy. The validation accuracy is used as a measure of performance. The experiments are run using 5 given seeds [0, 10, 1234, 99, 2021] to make it reproducible and to have a statistical performance measure of the model.

## Dataset

The assigment is worked on a binary classification data of 3 different domains. There are 4 different variations of the data based on how close the centroid of the point cloud clusters are for the different domains. The data pickle files are placed in the data directory.

The python notebooks of the assignment experiments on the 4 datasets are here:
* [Data with complete seperation](https://github.com/ankur-98/Multi-domain-Knowledge-Distillation/blob/master/MiniProject_d1.00.ipynb)
* [Data with 75% of complete seperation](https://github.com/ankur-98/Multi-domain-Knowledge-Distillation/blob/master/MiniProject_d0.75.ipynb)
* [Data with 50% of complete seperation](https://github.com/ankur-98/Multi-domain-Knowledge-Distillation/blob/master/MiniProject_d0.5.ipynb)
* [Data with 25% of complete seperation](https://github.com/ankur-98/Multi-domain-Knowledge-Distillation/blob/master/MiniProject_d0.25.ipynb)

## Setting up environment
### Option 1: Using pip

In a new `conda` or `virtualenv` environment, run

```bash
pip install -r requirements.txt
```

### Option 2: Using conda

Use the provided `environment.yml` file to install the dependencies into an environment.

```bash
conda env create
conda activate knowledge_distil
```

## References
1. [Hinton, G., Vinyals, O. and Dean, J., 2015. Distilling the knowledge in a neural network. arXiv preprint arXiv:1503.02531.](https://arxiv.org/pdf/1503.02531.pdf)
2. [Jafari, A., Rezagholizadeh, M., Sharma, P. and Ghodsi, A., 2021. Annealing Knowledge Distillation. arXiv preprint arXiv:2104.07163.](https://aclanthology.org/2021.eacl-main.212.pdf)
3. [AdamW and Super-convergence is now the fastest way to train neural nets by Sylvain Gugger and Jeremy Howard.](https://www.fast.ai/2018/07/02/adam-weight-decay/)
