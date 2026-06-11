# STEPS for model
It's better to unpack the data file and move them all into `data` directory for convenience when running examples below.
# 1. Input processing
## 1.1 Bacteria
For bacterial genomes, we recommend to use [RAST](https://rast.nmpdr.org/) for protein translation and annotation.
### 1.1.1 Single version (for preliminary exploration)
After obtaining the annotated protein sequences and the corresponding DNA sequences, the continuous O‑antigen biosynthesis gene cluster can be identified based on the housekeeping genes listed in `HousekeepingGenes.tsv`. For example, in Escherichia, for single bacterial strain
```
# For protein sequence
python FastaExtract.py -i ../data/Escherichia/bacteria/RAST/protein/BE.fasta -o ../data/Escherichia/bacteria/proc/protein/BE.fasta \
                       -s "UTP--glucose-1-phosphate uridylyltransferase" \
                       -e "6-phosphogluconate dehydrogenase" 
# Corresponding For DNA sequence
python FastaExtract.py -i ../data/Escherichia/bacteria/RAST/DNA/BE.fasta -o ../data/Escherichia/bacteria/proc/DNA/BE.fasta \
                       -s "UTP--glucose-1-phosphate uridylyltransferase" \
                       -e "6-phosphogluconate dehydrogenase"
```
If these sequences are in contig form, it is possible that housekeeping genes may be distributed across multiple contigs. To achieve more accurate retrieval, the contigs can be assembled using [RagTag](https://github.com/malonge/RagTag) prior to extraction.
- For more information about this script, run
```
> python FastaExtract.py -h
usage: FastaExtract.py [-h] -i FILE -o FILE [-a] [-s STR] [-e STR] [-k] [-l FILE] [-r]

FASTA annotation extraction tool

optional arguments:
  -h, --help            show this help message and exit
  -i FILE, --input FILE
                        Input FASTA file
  -o FILE, --output FILE
                        Output file path
  -a, --append          Append results to output file

Interval extraction:
  -s STR, --start STR   Start annotation pattern
  -e STR, --end STR     End annotation pattern
  -k, --keep            Keep start and end genes

List extraction:
  -l FILE, --list FILE  Pattern list file
  -r, --reverse         Exclude list patterns
```

### 1.1.2 Advanced version
After obtaining the annotated protein sequences and the corresponding DNA sequences, the continuous O‑antigen biosynthesis gene cluster can be identified based on the housekeeping genes listed in `HousekeepingGenes.tsv` and clustering with database (`OantiDatabase`) by [MMseqs2](https://github.com/soedinglab/mmseqs2). Please install [MMseqs2](https://github.com/soedinglab/mmseqs2) before running the following scripts.

First, run `MMseqs2Cluster.sh` to cluster input fasta sequences against a database fasta, outputting a TSV of clustered input sequence IDs.

Then, run `HousekeepCluster.py` to analyze the distribution of clusters and housekeeping genes on contigs, and extract candidate regions based on defined priority rules (see below for details). For example, in Escherichia, for single bacterial strain
```
# Different database fasta files will differ the results
bash MMseqs2Cluster.sh -i ../data/Escherichia/bacteria/RAST/protein/BE.fasta -d ../data/OantiDatabase/Escherichia.faa \ 
                       -o ../data/Escherichia/bacteria/OantiProc/BE

# To combine the clustering results with housekeeping genes info
# Different housekeeping gene files will differ the results
cat ../data/Escherichia/bacteria/OantiProc/BE/BE_cluster.tsv ../data/HousekeepingGenes.tsv > ../data/Escherichia/bacteria/OantiProc/BE/gene.tsv

# Different housekeeping gene files will differ the results
python HousekeepCluster.py -i ../data/Escherichia/bacteria/RAST/protein/BE.fasta -t ../data/Escherichia/bacteria/OantiProc/BE/gene.tsv \
                           -o ../data/Escherichia/bacteria/OantiProc/BE/prot -s ../data/Escherichia/bacteria/OantiProc/BE/proteins_dist.tsv
python HousekeepCluster.py -i ../data/Escherichia/bacteria/RAST/dna/BE.fasta -t ../data/Escherichia/bacteria/OantiProc/BE/gene.tsv \
                           -o ../data/Escherichia/bacteria/OantiProc/BE/dna -s ../data/Escherichia/bacteria/OantiProc/BE/dna_dist.tsv
```
After running above, more details can be checked in `prot/BE.log`, `dna/BE.log`, `proteins_dist.tsv`, `dna_dist.tsv` under `../data/Escherichia/bacteria/OantiProc/BE`. Then select or re-check the output FASTA file under sub-directory `dna`, `prot` and copy or move to `../data/Escherichia/bacteria/proc/protein` and `../data/Escherichia/bacteria/proc/DNA` for downstream processing.
- For more information about these scripts, run
```
> bash MMseqs2Cluster.sh -h
usage: MMseqs2Cluster.sh -i input -d database -o outdir [-m mmseqs][-p prefix][-c options][-q][-r][-h]

 Run MMseqs2 to cluster input fasta sequences against a database fasta, outputting a TSV of clustered input sequence IDs.
 Output TSV (saved to <outdir>/<input>_cluster.tsv) format: "cluster    input_sequence_id" and can be used for downstream analysis.
 Log file is saved to <outdir>/mmseqs2cluster.log. MMseqs2 log is saved to <outdir>/mmseqs_running.log.

 Options:
  -i  REQUIRED, Input fasta file
  -d  REQUIRED, Database fasta file, considered as reference sequences for clustering
  -o  REQUIRED, Output directory
  -m  Path to mmseqs2 executable (default: mmseqs)
  -p  Database annotation prefix (default: COLLECTED_O_ANTIGEN_CLUSTER_DATABASE)
  -c  Extra options for mmseqs cluster (e.g. "--min-seq-id 0.9 --cov-mode 1 -c 0.8")
  -q  Database already has prefix in headers, skip adding prefix
  -r  Remove mmseqs intermediate files, keep only TSV and final list
  -h  Show this help message and exit
```
```
> python HousekeepCluster.py -h
usage: HousekeepCluster.py [-h] -i FILE -t FILE -o PATH [-g INT] [-s FILE] [-r] [-l FILE]

Analyze the distribution of clusters and housekeeping genes on contigs, and extract candidate regions based on defined priority rules.
The script takes a protein FASTA file and a TSV file that defines cluster IDs and housekeeping gene annotations, then identifies potential cluster
regions while considering the presence of housekeeping genes as boundaries.

The priority rules are:

`P1`: There are multiple different housekeeping genes close to each other in one contig. The candidate region is defined as the segment between
two different housekeeping genes. The gap is the total count of cluster genes in that segment. Candidates with gap > 4 and <= threshold are
considered high confidence, while those with gap <= 3x threshold are considered relaxed.

`P2`: There are housekeeping genes close to an end of single contig and there are multiple housekeeping genes globally. The candidate region
is defined as the combination of segments on either side of the single housekeeping gene across different contigs. The gap is the total count
of cluster genes in the combined segments. Candidates with gap > 4 and <= threshold are considered high confidence, while those with
gap <= 3x threshold are considered relaxed.

`P3`: The housekeeping genes are far from each other or there is only one housekeeping gene globally. The candidate region is defined as the
segment on either side of the single housekeeping gene on the same contig. The gap is the count of cluster genes in that segment. Candidates
with gap > 4 and cluster_count/gap > 0.5 are considered high confidence.

`P4`: Housekeeping genes cannot be found. The candidate region is defined as the segment around the maximum positive count (cluster gene) on
the contig, extended in both directions as long as the gap does not exceed the threshold. Candidates with gap > 4 and cluster_count/gap > 0.5
are considered high confidence, while those that do not meet this ratio but have gap > 4 are considered relaxed.

optional arguments:
  -h, --help            show this help message and exit
  -i FILE, --input FILE
                        Input protein FASTA file
  -t FILE, --tsv FILE   Cluster/housekeeping gene TSV file (first column 'cluster' means mmseqs cluster result, otherwise housekeeping gene
                        abbreviation)
  -o PATH, --outdir PATH
                        Output directory
  -g INT, --gap INT     Max protein count threshold (default 25)
  -s FILE, --tsvout FILE
                        Output TSV file containing gene distribution by housekeeping (abbreviations), clustering (positive counts) and others (negative
                        counts) (optional)
  -r, --relax           Allow `P1` `P2` relaxed candidates with gap up to 3x threshold (`P4` will always output relaxed candidates)
  -l FILE, --log FILE   Log file (default <outdir>/<input>.log)
```

## 1.2 Phages
For phage genomes, we recommend to use [pharokka](https://github.com/gbouras13/pharokka) for protein translation and annotation.

After obtaining the annotated protein sequences and the corresponding DNA sequences, you can manually search for RBP proteins annotated as *tail fiber/fibre*, *fiber/fibre tail*, *tail spike*, or *receptor binding* and extract them. This step can also be done by using [SeekRBP](https://github.com/Saillxl/SeekRBP).

If necessary, you can further use [AlphaFold](https://github.com/google-deepmind/alphafold) for structural prediction and then manually determine which proteins are RBPs for extraction.
# 2. Feature generation
Once the continuous O‑antigen biosynthesis gene clusters of bacteria and the RBPs of phages (including both protein and DNA sequences) have been obtained, feature extraction can be performed. 
```
# For single bacterial strain
python FeatureGenerate.py -d ../data/Escherichia/bacteria/proc/DNA/BE.fasta -p ../data/Escherichia/bacteria/proc/protein/BE.fasta \
                          -o InputDir/bactEscherichia.csv -s mean,std,min,q25,median,q75,max
# For single phage
python FeatureGenerate.py -d ../data/Escherichia/phages/proc/DNA/T4LD.fasta -p ../data/Escherichia/phages/proc/protein/T4LD.fasta \
                          -o InputDir/phageEscherichia.csv -s mean,min,max
```
To collect features from all bacterial or phage sequence pairs into a single CSV for downstream use, run the per‑sample command repeatedly but write into the same output file. Or using `for` loop for processing recursively.

- For more information about this script, run
```
> python FeatureGenerate.py -h
usage: FeatureGenerate.py [-h] -d FILE -p FILE -o FILE [-l FILE] [-s STR,..] [-v] [-rl] [-rd]

Integrated Genomic Feature From DNA and Protein Sequences

optional arguments:
  -h, --help            show this help message and exit
  -d FILE, --dna FILE   Input DNA FASTA file path
  -p FILE, --protein FILE
                        Input protein FASTA file path
  -o FILE, --output FILE
                        Output CSV file path
  -l FILE, --list FILE  Gene filter TSV file (no header, only consider first column)
  -s STR,.., --statistics STR,..
                        Statistics list, separated by "," (valid: mean(default), std, var, min, q5, q10, q25,
                        median, q75, q90, q95, max, ptp, iqr, cv, entropy, skew, kurt)
  -v, --inverse         Inverse filtering mode, will exclude genes
  -rl, --remove-collinear
                        Remove collinear features
  -rd, --remove-codon   Disable codon features
```
## 2.1 Interaction files
To ensure the correctness of the interactions, please make sure that the order of bacteria/phages in the generated feature file matches the order in the interaction file. For the interaction file, convert the TSV file into the required format (for CSV files, use the -c option)
```
python FormatFile.py -i ../data/Escherichia/interaction/Interaction.tsv -o ../data/InputDir/edgeEscherichia.csv
```
- For more information about this script, run
```
> python FormatFile.py -h
usage: FormatFile.py [-h] -i INPUT -o OUTPUT [-c] [-r]

Convert bacteria-phage matrix to long-format CSV

optional arguments:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        Input file (TSV by default)
  -o OUTPUT, --output OUTPUT
                        Output CSV file
  -c, --csv             Input file is CSV instead of TSV
  -r, --reverse         Interpret columns as bacteria and rows as phages
```
# 3. Meta-learning
Once feature generation is complete, place all input files into a single directory (e.g., `InputDir`). Each dataset should contain three files, such as `bactEscherichia.csv`, `phageEscherichia.csv`, and `edgeEscherichia.csv`. Then run the MAML training process to obtain the Meta-learning model.
```
python MAML.py -i ../data/InputDir -o ../data/MAMLDir
```
- For more information about MAML settings, run
```
> python MAML.py -h
usage: MAML.py [-h] -i PATH -o PATH [-s1 INT] [-t INT] [-sn INT,...] [-s2 INT] [-ss INT,INT,INT,INT] [-a INT]
            [-ep INT] [-prop FLOAT] [-sr FLOAT] [-gr FLOAT] [-er FLOAT] [-m {0,1,2,3}] [-ilr FLOAT] [-iep INT]
            [-lr FLOAT] [-g INT] [-p INT]

MAML Meta Learning for Link Prediction

optional arguments:
  -h, --help           show this help message and exit
  -i PATH              Input directory
  -o PATH              Output directory
  -s1 INT              Seed for numpy (default: 86)
  -t INT               Number of task per step (default: 2)
  -sn INT,...          Dataset index for test, seperated by ",", will override -s1 and -t
  -s2 INT              Seed for torch (default: 86)
  -ss INT,INT,INT,INT  Number of nodes (default: 64,32,16,8)
  -a INT               Adaptor dim (default: No adaptor)
  -ep INT              Number of step for training (default: 1000)
  -prop FLOAT          Proportion of nodes to sample (default: 0.75)
  -sr FLOAT            Support set ratio (default: 0.3)
  -gr FLOAT            Training graph set ratio (default: 1=use all)
  -er FLOAT            Training edge set ratio (default: 1=use all)
  -m {0,1,2,3}         Mark for inner update params (default: 0)
  -ilr FLOAT           Inner loop learning rate (default: 0.0001)
  -iep INT             Number of inner update steps (default: 5)
  -lr FLOAT            Learning rate (default: 0.0001)
  -g INT               Gamma for focal loss (default: 4)
  -p INT               Number of patience (default: 50)
```
# 4. Fune-tuning
After completing MAML meta‑learning and obtaining the meta‑learned model, fine‑tuning on individual datasets is required. The first step is to split the dataset.
```
python SplitData.py -i ../data/InputDir -o ../data/SplitDir
```
Then proceed with fine‑tuning
```
python MlFunetun.py -i ../data/SplitDir -o ../data/OutputDir -m ../data/MAMLDir/MAML_best_model.pth
```
If you would like to use a trained model, say Escherichia, you can run:
```
python MlFunetun.py -i ../data/SplitDir -i1 Escherichia -o ../data/OutputDir -m ../data/MAMLDir/Escherichia_best_model.pth
```
- For more information, run
```
> python SplitData.py -h
usage: SplitData.py [-h] -i PATH -o PATH [-s1 INT] [-s2 INT] [-prop FLOAT]

Save data for data spliting

optional arguments:
  -h, --help   show this help message and exit
  -i PATH      Input directory
  -o PATH      Output directory
  -s1 INT      Seed for numpy (default: 86)
  -s2 INT      Seed for torch (default: 86)
  -prop FLOAT  Proportion of nodes to sample (default: 0.75)
```
```
> python MlFunetun.py -h
usage: MlFunetun.py [-h] -ip PATH -o PATH [-i1 STR] [-si INT] [-sd INT] [-s2 INT] [-ss INT,INT,INT,INT] [-a INT]
            [-ep INT] [-prop FLOAT] [-sr FLOAT] [-gr FLOAT] [-er FLOAT] [-m FILE] [-mt INT] [-iep INT]
            [-lr FLOAT] [-g INT] [-p INT]

MAML fine-tuning for link prediction

optional arguments:
  -h, --help           show this help message and exit
  -ip PATH             Input directory
  -o PATH              Output directory
  -i1 STR              Input file prefix
  -si INT              Small index (default: 2)
  -sd INT              Small dimension (default: 756)
  -s2 INT              Seed for torch (default: 86)
  -ss INT,INT,INT,INT  Number of nodes (default: 64,32,16,8)
  -a INT               Adaptor dim (default: No adaptor)
  -ep INT              Number of step for training (default: 1000)
  -prop FLOAT          Proportion of nodes to sample (default: 0.75)
  -sr FLOAT            Support set ratio (default: 1)
  -gr FLOAT            Training graph set ratio (default: 1=use all)
  -er FLOAT            Training edge set ratio (default: 1=use all)
  -m FILE              Pretrained model file
  -mt INT              Model type (1(default): with GCN, 2: without GCN)
  -iep INT             Number of inner update steps (default: 5)
  -lr FLOAT            Learning rate (default: 0.0001)
  -g INT               Gamma for focal loss (default: 4)
  -p INT               Number of patience (default: 50)
```

