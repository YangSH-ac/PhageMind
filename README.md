# PhageMind
PhageMind is designed to predict interactions between phages and bacterial strains within target genera. The general inputs include genomes from bacteria and phages (fasta files), with some known interactions between them. 
## Prepare the environment
We suggest to install most of packages using conda (https://anaconda.org/) if available.

The main model is built under `torch-geometric`, but first `pytorch` is needed for further installation:
```
# If only CPU is available
conda create -n phagemind -c pytorch -c conda-forge pytorch torchvision torchaudio 
# OR if a GPU is available, check its CUDA version and install the corresponding pytorch‑cuda package
conda create -n phagemind -c pytorch -c conda-forge -c nvidia pytorch torchvision torchaudio pytorch-cuda=xx.x
# For example, if CUDA version is 12.1, then pytorch-cuda=12.1
conda create -n phagemind -c pytorch -c conda-forge -c nvidia pytorch torchvision torchaudio pytorch-cuda=12.1
```
Then for feature generation and input processing, please also run below:
```
conda install -n phagemind -c conda-forge biopython pandas scikit-learn
```
Finally, for installation of package `torch-geometric`, please refer to [PyG Documentation](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html) for details. All the additional libraries are recommended to install too.

To ensure all packages are installed, please run below
```
$CONDA_PREFIX/envs/phagemind/bin/python -c "import torch,torch_geometric,sklearn"
# This should finish without any ERROR message
# OR try this
conda activate phagemind;python -c "import torch,torch_geometric,sklearn"
```
Then download the scripts of PhageMind:
```
git clone https://github.com/YangSH-ac/PhageMind.git
```
To execute the code, navigate to the directory and activate the environment:
```
cd PhageMind/code
conda activate phagemind
```
## Usage
For detailed instruction, refer to `code/STEPS.md`.

For dataset, refer to `data/DATA.md`. Please download all the data before trying to run the examples inside the `code/STEPS.md`.
## References
The arXiv version can be found via: [arXiv version](https://arxiv.org/abs/2601.15886)
