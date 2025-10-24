# FEMPBP (Finetuned ESM-2 model for predicting whether a protein is from bacteria or phages)

### Description

A deep learning model developed by finetuning ESM-2, which is used for predicting whether a protein is from bacteria or phages

Deep learning model for PIDE (https://github.com/chyghy/PIDE)

### Instructions (This model can also be directly used in the PIDE environment)

(1) Create an environment for FEMPBP

```bash
conda create -n FEMPBP
conda activate FEMPBP
```

(2) Install the required Python packages

a, PyTorch:

Run on CPU

```bash
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

If you want to use the GPU version, please go to https://pytorch.org/get-started and get the pip install command according to your device and demand.

⚠️ The GPU version of FEMPBP does not currently support macOS systems with Apple Silicon (M-series) chips by default. It is recommended to use the CPU version directly, or modify the relevant code to enable GPU support on M-series Macs.

b, fair-esm

``` 
pip install fair-esm
```

c, biopython
```
conda install -c bioconda biopython
```

(3) Download the model

```bash
wget https://zenodo.org/records/12759619/files/PIDE.model.tar.gz
tar xzvf PIDE.model.tar.gz
```

(4) Download the source code of PIDE from github

```
git clone https://github.com/li-bw18/FEMPBP.git
```

### Usage

To get the HELP information

```bash
python FEMPBP/run.py -h
```

```
python FEMPBP/run.py [-o OUTPUT] [-g GPU] [-b BATCHSIZE] input model
```

Explanation

```
positional arguments:
  input                 Path of the input protein fasta file
  model                 Path of the model parameter file

options:
  -h, --help            show this help message and exit
  -o OUTPUT, --output OUTPUT
                        Path of the output file, default is ./results.txt
  -g GPU, --GPU GPU     Determine which GPU(s) to use, see README for more information
  -b BATCHSIZE, --BatchSize BATCHSIZE
                        Define the batch size used in the prediction
```

### Output file (Default is ./results.txt)

   This txt file lists the predicted results of all input proteins (seperated by TAB'\t')

   **seq_id**: The ID/name of the input proteins

   **phage_prob**: The probability for a protein to be a phage protein

   **predict_result**: The predicted result ("Bacterium"/"Phage" proteins)
