# PhageMind
PhageMind is designed to predict interactions between phages and bacterial strains within target genera. The general inputs include genomes from bacteria and phages (fasta files), with some known interactions between them. 
## Prepare the environment
*Note*: we suggest you to install all the package using conda (https://anaconda.org/).

The main model is built under `torch-geometric`:
```
# If only CPU is available
conda create -n phagemind -c pytorch -c conda-forge pytorch torchvision torchaudio 
# OR if a GPU is available, check its CUDA version and install the corresponding pytorch‑cuda package
conda create -n phagemind -c pytorch -c conda-forge -c nvidia pytorch torchvision torchaudio pytorch-cuda=xx.x
```
For feature generation and input processing, please also run below:
```
conda install -n phagemind -c conda-forge biopython pandas scikit-learn
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
## References
The arXiv version can be found via: [arXiv version](https://arxiv.org/abs/2601.15886)
