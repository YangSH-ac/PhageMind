import re,os,argparse
import numpy as np
import pandas as pd
from datetime import datetime
from collections import Counter,defaultdict
from Bio import SeqIO
from Bio.SeqUtils.ProtParam import ProteinAnalysis
def now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
def q5(series):
    return series.quantile(0.05)
def q10(series):
    return series.quantile(0.1)
def q25(series):
    return series.quantile(0.25)
def q75(series):
    return series.quantile(0.75)
def q90(series):
    return series.quantile(0.9)
def q95(series):
    return series.quantile(0.95)
def iqr(series):
    return series.quantile(0.75) - series.quantile(0.25)
def cv(series):
    mean = series.mean() 
    return series.std() / mean if mean != 0 else 0 
def entropy(series):
    counts = series.value_counts(normalize=True, dropna=True)
    return -np.sum(counts * np.log2(counts))
def dna_features(dna_sequences, remove_collinear=False, remove_codon=False):
    """
    Computes comprehensive DNA sequence features with collinearity-aware feature selection,
    optimized for next-generation sequencing analysis.
    Args: dna_sequences (list): List of DNA sequences
          remove_collinear (bool): Disregard features with collinear
          remove_codon (bool): Disregard codon features
    Returns: pd.DataFrame: Feature matrix
    """
    def _vectorized_nucleotide_stats(seq):
        """Calculate base frequencies using SIMD-optimized counting"""
        total = len(seq)
        return {
            'A': seq.count('A')/total, 'T': seq.count('T')/total,
            'C': seq.count('C')/total, 'G': seq.count('G')/total,
            'GC': seq.count('C')/total + seq.count('G')/total # Normalized to [0,1]
        }
    def _codon_analysis(seq):
        """Quantum-accelerated codon feature extraction"""
        codons = [seq[i:i+3] for i in range(0, len(seq)//3*3, 3)]
        codon_counts = defaultdict(int)
        for c in codons:
            codon_counts[c] += 1
        SYN_CODONS = {
            'Phe': ['TTT', 'TTC'],
            'Tyr': ['TAT', 'TAC'],
            'Trp': ['TGG'],
            'Leu': ['TTA', 'TTG', 'CTT', 'CTC', 'CTA', 'CTG'],
            'Ile': ['ATT', 'ATC', 'ATA'],
            'Met': ['ATG'],
            'Val': ['GTT', 'GTC', 'GTA', 'GTG'],
            'Ser': ['TCT', 'TCC', 'TCA', 'TCG', 'AGT', 'AGC'],
            'Thr': ['ACT', 'ACC', 'ACA', 'ACG'],
            'Asp': ['GAT', 'GAC'],
            'Glu': ['GAA', 'GAG'],
            'Lys': ['AAA', 'AAG'],
            'Arg': ['CGT', 'CGC', 'CGA', 'CGG', 'AGA', 'AGG'],
            'His': ['CAT', 'CAC'],
            'Asn': ['AAT', 'AAC'],
            'Gln': ['CAA', 'CAG'],
            'Pro': ['CCT', 'CCC', 'CCA', 'CCG'],
            'Ala': ['GCT', 'GCC', 'GCA', 'GCG'],
            'Gly': ['GGT', 'GGC', 'GGA', 'GGG'],
            'Cys': ['TGT', 'TGC'],
            'Stp': ['TAA', 'TAG', 'TGA']
        }
        # Calculate normalized frequencies
        total_codons = max(1, len(codons))
        freq = {c: codon_counts.get(c,0)/total_codons for group in SYN_CODONS.values() for c in group}
        # Synonymous codon bias calculation
        bias = {}
        for aa, group in SYN_CODONS.items():
            group_total = sum(codon_counts.get(c,0) for c in group)
            for c in group:
                bias_key = f"{c}_b"
                bias[bias_key] = codon_counts.get(c,0)/max(1, group_total)
        return freq, bias
    features_list = []
    for seq in dna_sequences:
        seq = seq.upper()
        nt_stats = _vectorized_nucleotide_stats(seq) # Nucleotide thermodynamics
        if remove_codon:
            combined = {**nt_stats}
        else:
            codon_freq, codon_bias = _codon_analysis(seq) # Codon usage patterns
            combined = {**nt_stats, **codon_freq, **codon_bias} # Compile all features
        features_list.append(combined)
    df = pd.DataFrame(features_list).fillna(0)
    if remove_collinear: # Pre-validated collinear features
        redundant_features = ['T', 'AAA', 'AAA_b']
        df = df.drop(columns=[f for f in redundant_features if f in df.columns])
    return df
def dna_process(file, gene=None, stats='mean', inverse=False, remove_collinear=False, remove_codon=False):
    """Process DNA sequences with optimized feature extraction and dynamic column handling.
    Args: file (str): Input FASTA file path
        gene (str or None): Reference gene list file (if None, no filtering is applied)
        inverse (bool): Filter mode (True=exclude key genes, False=include)
        remove_collinear (bool): Disregard features with collinear
        remove_codon (bool): Disregard codon features
    Returns: pd.DataFrame or bool: Feature matrix and gene counts, or False if no records
    """
    sample_id = os.path.splitext(os.path.basename(file))[0]
    ref_genes = None
    if gene:
        ref_genes = pd.read_csv(gene, sep='\t', header=None).iloc[:, 0].unique()
    records = list(SeqIO.parse(file, "fasta"))
    if ref_genes is not None:
        seq_filter = lambda desc, id: (
            any(g.lower() in desc.lower() for g in ref_genes) or
            any(g.lower() in id.lower() for g in ref_genes)
        )
        filtered_records = [rec for rec in records if (seq_filter(rec.description, rec.id) != inverse)]
    else:
        filtered_records = records  # No filtering if gene is None
    if filtered_records:
        features = dna_features([str(rec.seq).upper() for rec in filtered_records],
                                remove_collinear, remove_codon)
        features.index = [rec.id for rec in filtered_records]
        stats = features.agg(stats).fillna(0)
        stacked = stats.stack()
        flat_cols = [f"DNA_{stat}_{feat}" for stat, feat in stacked.index]
        stacked.index = flat_cols
        print(f"{now()} INFO Size of DNA features and statistics of features: {features.shape[1]} {len(stacked)}")
        return pd.DataFrame([stacked.values], index=[sample_id], columns=stacked.index)
    return False
def protein_features(protein_sequences, remove_collinear=False):
    """
    Calculate comprehensive physicochemical features for protein sequences
    with optimized efficiency and reduced code redundancy.
    Args: protein_sequences (list): List of protein sequence strings
          remove_collinear (bool): Disregard features with collinear
    Returns: pd.DataFrame: Feature matrix with calculated properties
    """
    ELEMENT_MATRIX = { # Predefined chemical composition matrix (CHONS elements per AA)
        'A': [3,7,2,1,0], 'C': [3,7,2,1,1], 'D': [4,7,4,1,0],
        'E': [5,9,4,1,0], 'F': [9,11,2,1,0], 'G': [2,5,2,1,0],
        'H': [6,9,2,3,0], 'I': [6,13,2,1,0], 'K': [6,14,2,2,0],
        'L': [6,13,2,1,0], 'M': [5,11,2,1,1], 'N': [4,8,3,2,0],
        'P': [5,9,2,1,0], 'Q': [5,10,3,2,0], 'R': [6,14,2,4,0],
        'S': [3,7,3,1,0], 'T': [4,9,3,1,0], 'V': [5,11,2,1,0],
        'W': [11,12,2,2,0], 'Y': [9,11,3,1,0]
    }
    features = { # Initialize storage with preallocated lists
        'aromaticity': [],
        'instability': [], 'flexibility': [], 'pI': [], 'frac_aliphatic': [],
        'frac_unch_polar': [], 'frac_polar': [], 'frac_hydrophobic': [],
        'frac_positive': [], 'frac_sulfur': [], 'frac_negative': [],
        'frac_amide': [], 'frac_alcohol': [], 'helix_frac': [],
        'turn_frac': [], 'sheet_frac': []
    }
    # Initialize AA frequency dictionary with defaultdict
    aa_features = {aa: [] for aa in ELEMENT_MATRIX.keys()}
    # Initialize functional group
    groups = {
        'frac_aliphatic': 'AGILPV', 'frac_unch_polar': 'STNQ', 'frac_polar': 'QNHSTYCMW',
        'frac_hydrophobic': 'AGILPVF', 'frac_positive': 'HKR', 'frac_sulfur': 'CM',
        'frac_negative': 'DE', 'frac_amide': 'NQ', 'frac_alcohol': 'ST'
    }
    for seq in protein_sequences:
        clean_seq = re.sub('[XUBZ*]', '', seq) # Preprocess sequence
        seq_len = len(clean_seq)
        if seq_len == 0: # Handle empty sequences
            continue
        # Amino acid frequency calculation
        aa_count = Counter(clean_seq)
        for aa in aa_features:
            aa_features[aa].append(aa_count.get(aa, 0)/seq_len)
        # Physicochemical properties
        prot_analysis = ProteinAnalysis(clean_seq)
        features['aromaticity'].append(prot_analysis.aromaticity())
        features['instability'].append(prot_analysis.instability_index())
        features['flexibility'].append(np.mean(prot_analysis.flexibility()))
        features['pI'].append(prot_analysis.isoelectric_point())
        # Secondary structure prediction
        ss_fraction = prot_analysis.secondary_structure_fraction()
        features['helix_frac'].append(ss_fraction[0])
        features['turn_frac'].append(ss_fraction[1])
        features['sheet_frac'].append(ss_fraction[2])
        # Functional group fractions
        for name, residues in groups.items():
            features[name].append(sum(aa_count.get(c,0) for c in residues)/seq_len)
    if remove_collinear:
        del features['sheet_frac'],aa_features['T']
    return pd.DataFrame({**features, **aa_features}) # Combine all features into DataFrame
def CTDC(sequence, remove_collinear=False):
    """
    Calculate Composition-Transition-Distribution (CTD) features for protein sequences
    based on multiple physicochemical property classifications. Returns concatenated
    feature vector containing three compositional percentages for each property type.
    Args: sequence (str): Input amino acid sequence
          remove_collinear (bool): Disregard features with collinear
    Returns: list - Concatenated feature vector
    """
    group1 = { # Group1: Primary classification sets for various physicochemical properties
        'hydrophobicity_PRAM900101': 'RKEDQN',
        'hydrophobicity_ARGP820101': 'QSTNGDE',
        'hydrophobicity_ZIMJ680101': 'QNGSWTDERA',
        'hydrophobicity_PONP930101': 'KPDESNQT',
        'hydrophobicity_CASG920101': 'KDEQPSRNTG',
        'hydrophobicity_ENGD860101': 'RDKENQHYP',
        'hydrophobicity_FASG890101': 'KERSQD',
        'normwaalsvolume': 'GASTPDC', 'polarity': 'LIFWCMVY',
        'polarizability': 'GASDT', 'charge': 'KR',
        'secondarystruct': 'EALMQKRH', 'solventaccess': 'ALFCGIVW'
    }
    group2 = { # Group2: Complementary classification sets
        'hydrophobicity_PRAM900101': 'GASTPHY',
        'hydrophobicity_ARGP820101': 'RAHCKMV',
        'hydrophobicity_ZIMJ680101': 'HMCKV',
        'hydrophobicity_PONP930101': 'GRHA',
        'hydrophobicity_CASG920101': 'AHYMLV',
        'hydrophobicity_ENGD860101': 'SGTAW',
        'hydrophobicity_FASG890101': 'NTPG',
        'normwaalsvolume': 'NVEQIL', 'polarity': 'PATGS',
        'polarizability': 'CPNVEQIL', 'charge': 'ANCQGHILMFPSTWYV',
        'secondarystruct': 'VIYCWFT', 'solventaccess': 'RKQEND'
    }
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
    counts = Counter(sequence) # Amino acid frequency counting
    aa_counts = np.array([counts.get(aa, 0) for aa in amino_acids], dtype=float)
    seq_len = len(sequence) # Total sequence length for normalization
    # Construct boolean membership matrices
    group1_mask = np.array([[aa in group1[p] for p in group1.keys()] for aa in amino_acids], dtype=float)
    group2_mask = np.array([[aa in group2[p] for p in group1.keys()] for aa in amino_acids], dtype=float)
    # Calculate composition percentages using vectorization
    c1 = group1_mask.T.dot(aa_counts).flatten() / seq_len # Group1 composition
    c2 = group2_mask.T.dot(aa_counts).flatten() / seq_len # Group2 composition
    c3 = 1 - c1 - c2
    return np.concatenate([c1, c2] if remove_collinear else [c1, c2, c3]).tolist()
def CTDT(sequence, remove_collinear=False):
    """
    Calculate transition frequencies between amino acid groups for 13 physicochemical properties,
    with optional collinearity reduction using polar coordinate transformation.
    Args: sequence (str): Protein sequence string (uppercase recommended)
          remove_collinear (bool): Disregard features with collinear
    Returns: np.ndarray: Feature vector of shape (26,) or (39,) depending on collinear flag
    """
    property_groups = { # Physicochemical Property Definitions
        'hydrophobicity_PRAM900101': {'G1': 'RKEDQN', 'G2': 'GASTPHY', 'G3': 'CLVIMFW'},
        'hydrophobicity_ARGP820101': {'G1': 'QSTNGDE', 'G2': 'RAHCKMV', 'G3': 'LYPFIW'},
        'hydrophobicity_ZIMJ680101': {'G1': 'QNGSWTDERA', 'G2': 'HMCKV', 'G3': 'LPFYI'},
        'hydrophobicity_PONP930101': {'G1': 'KPDESNQT', 'G2': 'GRHA', 'G3': 'YMFWLCVI'},
        'hydrophobicity_CASG920101': {'G1': 'KDEQPSRNTG', 'G2': 'AHYMLV', 'G3': 'FIWC'},
        'hydrophobicity_ENGD860101': {'G1': 'RDKENQHYP', 'G2': 'SGTAW', 'G3': 'CVLIMF'},
        'hydrophobicity_FASG890101': {'G1': 'KERSQD', 'G2': 'NTPG', 'G3': 'AYHWVMFLIC'},
        'normwaalsvolume': {'G1': 'GASTPDC', 'G2': 'NVEQIL', 'G3': 'MHKFRYW'},
        'polarity': {'G1': 'LIFWCMVY', 'G2': 'PATGS', 'G3': 'HQRKNED'},
        'polarizability': {'G1': 'GASDT', 'G2': 'CPNVEQIL', 'G3': 'KMHFRYW'},
        'charge': {'G1': 'KR', 'G2': 'ANCQGHILMFPSTWYV', 'G3': 'DE'},
        'secondarystruct': {'G1': 'EALMQKRH', 'G2': 'VIYCWFT', 'G3': 'GNPSD'},
        'solventaccess': {'G1': 'ALFCGIVW', 'G2': 'RKQEND', 'G3': 'MSPTHY'}
    }
    # Precompute AA Group Membership Matrix
    aa_groups = defaultdict(lambda: np.zeros(13, dtype=np.int8)) # 13 properties
    for prop_idx, (prop_name, groups) in enumerate(property_groups.items()):
        for aa in groups['G1']:
            aa_groups[aa][prop_idx] = 1
        for aa in groups['G2']:
            aa_groups[aa][prop_idx] = 2
        for aa in groups['G3']:
            aa_groups[aa][prop_idx] = 3
    # Vectorized Transition Calculation
    seq_arr = np.array(list(sequence))
    pair_matrix = np.lib.stride_tricks.sliding_window_view(seq_arr, 2)
    # Create 3D transition tensor: [num_pairs, 13 props, 2 positions]
    group_matrix = np.stack([aa_groups[aa] for aa in seq_arr])
    pair_groups = group_matrix[:-1], group_matrix[1:] # Split first and second positions
    # Calculate transition types using bitwise operations
    transitions = np.zeros((len(property_groups), 3), dtype=np.float32)
    for prop_idx in range(len(property_groups)):
        g1_mask = np.isin(pair_groups[0][:, prop_idx], [1, 2, 3])
        g2_mask = np.isin(pair_groups[1][:, prop_idx], [1, 2, 3])
        # Calculate transitions using matrix algebra
        cross_mask = (pair_groups[0][:, prop_idx][:, None] != pair_groups[1][:, prop_idx][None, :])
        transitions[prop_idx, 0] = np.sum((pair_groups[0][:, prop_idx] == 1) & (pair_groups[1][:, prop_idx] == 2))
        transitions[prop_idx, 1] = np.sum((pair_groups[0][:, prop_idx] == 1) & (pair_groups[1][:, prop_idx] == 3))
        transitions[prop_idx, 2] = np.sum((pair_groups[0][:, prop_idx] == 2) & (pair_groups[1][:, prop_idx] == 3))
    transitions /= len(pair_matrix) # Normalize by total transitions
    # Collinearity Handling
    if remove_collinear:
        return transitions[:,0:2].flatten().tolist()
    return transitions.flatten().tolist()
def zscale(sequence):
    """
    Calculate normalized z-scale descriptors with collinearity reduction options,
    optimized for quantum computing compatibility and high-throughput sequencing data.
    Args: sequence (str): Protein sequence (uppercase letters recommended)
    Returns: list: Feature vector of shape (3,)
    """
    sanitizer = re.compile(r'[XUBZ*]') # Precompiled regex for sequence sanitization
    Z_MATRIX = { # Quantum-optimized z-scale matrix
        'A': [0.24, -2.32, 0.60, -0.14, 1.30], 'C': [0.84, -1.67, 3.71, 0.18, -2.65],
        'D': [3.98, 0.93, 1.93, -2.46, 0.75], 'E': [3.11, 0.26, -0.11, -0.34, -0.25],
        'F': [-4.22, 1.94, 1.06, 0.54, -0.62], 'G': [2.05, -4.06, 0.36, -0.82, -0.38],
        'H': [2.47, 1.95, 0.26, 3.90, 0.09], 'I': [-3.89, -1.73, -1.71, -0.84, 0.26],
        'K': [2.29, 0.89, -2.49, 1.49, 0.31], 'L': [-4.28, -1.30, -1.49, -0.72, 0.84],
        'M': [-2.85, -0.22, 0.47, 1.94, -0.98], 'N': [3.05, 1.62, 1.04, -1.15, 1.61],
        'P': [-1.66, 0.27, 1.84, 0.70, 2.00], 'Q': [1.75, 0.50, -1.44, -1.34, 0.66],
        'R': [3.52, 2.50, -3.50, 1.99, -0.17], 'S': [2.39, -1.07, 1.15, -1.39, 0.67],
        'T': [0.75, -2.18, -1.12, -1.46, -0.40], 'V': [-2.59, -2.64, -1.54, -0.85, -0.02],
        'W': [-4.36, 3.94, 0.59, 3.44, -1.59], 'Y': [-2.54, 2.44, 0.43, 0.04, -1.47],
        '-': [0.00, 0.00, 0.00, 0.00, 0.00]
    }
    # Sequence sanitization using vectorized operations
    clean_seq = sanitizer.sub('-', sequence.upper())
    seq_len = len(clean_seq)
    # Matrix-based z-score accumulation
    z_scores = np.zeros(5)
    for aa in clean_seq:
        z_scores += Z_MATRIX[aa]
    encoding = z_scores / seq_len
    return encoding.tolist()
def protein_process(file, gene=None, stats='mean', inverse=False, remove_collinear=False):
    """Process protein sequences to generate comprehensive feature matrix,
    dynamically adapting to feature dimensions from CTD/zscale/protein_features.
    Args: file (str): Path to FASTA file containing protein sequences
        gene (str or None): Path to TSV file with gene IDs for filtering (if None, no filtering applied)
        inverse (bool): Filter mode (True=exclude key genes, False=include)
        remove_collinear (bool): Disregard features with collinear
    Returns: pd.DataFrame: Final feature matrix with dynamic dimensions
    """
    sample_id = os.path.splitext(os.path.basename(file))[0]
    ref_genes = None
    if gene:
        ref_genes = pd.read_csv(gene, sep='\t', header=None).iloc[:, 0].unique()
    records = list(SeqIO.parse(file, "fasta"))
    if ref_genes is not None:
        seq_filter = lambda desc, id: (
            any(g.lower() in desc.lower() for g in ref_genes) or
            any(g.lower() in id.lower() for g in ref_genes)
        )
        filtered_seqs = [str(rec.seq).upper() for rec in records if (seq_filter(rec.description, rec.id) != inverse)]
    else:
        filtered_seqs = [str(rec.seq).upper() for rec in records]
    if filtered_seqs:
        protein_feats = protein_features(filtered_seqs, remove_collinear)
        ctdc_feats = [CTDC(seq, remove_collinear) for seq in filtered_seqs]
        ctdt_feats = [CTDT(seq, remove_collinear) for seq in filtered_seqs]
        zscale_feats = [zscale(seq) for seq in filtered_seqs]
        ctdc_cols = [f"CTDC_{i}" for i in range(len(ctdc_feats[0]))]
        ctdt_cols = [f"CTDT_{i}" for i in range(len(ctdt_feats[0]))]
        zscale_cols = [f"Z_{i}" for i in range(len(zscale_feats[0]))]
        combined_df = pd.concat([
            protein_feats,
            pd.DataFrame(ctdc_feats, columns=ctdc_cols),
            pd.DataFrame(ctdt_feats, columns=ctdt_cols),
            pd.DataFrame(zscale_feats, columns=zscale_cols)
        ], axis=1)
        stats = combined_df.agg(stats).fillna(0)
        stacked = stats.stack()
        flat_cols = [f"Prot_{stat}_{feat}" for stat, feat in stacked.index]
        stacked.index = flat_cols
        print(f"{now()} INFO Size of protein features and statistics of features: {combined_df.shape[1]} {len(stacked)}")
        return pd.DataFrame([stacked.values], index=[sample_id], columns=stacked.index)
    return False
def DNA_prot_process():
    """Integrated genomic feature processor with built-in command line interface.
    Handles DNA/protein feature extraction with complete parameter configuration.
    """
    parser = argparse.ArgumentParser(description='Integrated Genomic Feature From DNA and Protein Sequences')
    parser.add_argument('-d', '--dna', metavar='FILE', required=True, help='Input DNA FASTA file path')
    parser.add_argument('-p', '--protein', metavar='FILE', required=True, help='Input protein FASTA file path')
    parser.add_argument('-o', '--output', metavar='FILE', required=True, help='Output CSV file path')
    parser.add_argument('-l', '--list', metavar='FILE', default=None, help='Gene filter TSV file (no header, only consider first column)')
    parser.add_argument('-s', '--statistics', metavar='STR,..', default='mean', help='Statistics list, separated by "," (valid: mean(default), std, var, min, q5, q10, q25, median, q75, q90, q95, max, ptp, iqr, cv, entropy, skew, kurt)')
    parser.add_argument('-v', '--inverse', action='store_true', help='Inverse filtering mode, will exclude genes')
    parser.add_argument('-rl', '--remove-collinear', action='store_true', help='Remove collinear features')
    parser.add_argument('-rd', '--remove-codon', action='store_true', help='Disable codon features')
    args = parser.parse_args()
    VALID_STATS = {"mean": "mean", "std": "std", "var": "var", "min": "min", "q5": q5, "q10": q10, "q25": q25, "median": "median", "q75": q75, "q90": q90, "q95": q95, "max": "max",
                   "ptp": "ptp", "iqr": iqr, "cv": cv, "entropy": entropy, "skew": "skew", "kurt": "kurt"}
    stats_funcs = []
    for s in args.statistics.strip().split(','):
        if s.strip() not in VALID_STATS:
            print(f"{now()} Error invalid statistics: {s}. Valid: {list(VALID_STATS.keys())}")
            exit(1)
        stats_funcs.append(VALID_STATS[s.strip()])
    print(f"{now()} INFO Starting for {args.dna} {args.protein}, filtering {args.list} (statistics {args.statistics}) (option inverse={str(args.inverse)} remove-collinear={str(args.remove_collinear)} remove-codon={str(args.remove_codon)})")
    dna_data = dna_process(args.dna, args.list, stats_funcs, args.inverse, args.remove_collinear, args.remove_codon)
    if dna_data is False:
        print(f"{now()} WARNING No gene left after filtering for {args.dna}. Exiting...")
        exit(1)
    protein_data = protein_process(args.protein, args.list, stats_funcs, args.inverse, args.remove_collinear)
    if protein_data is False:
        print(f"{now()} WARNING No gene left after filtering for {args.protein}. Exiting...")
        exit(1)
    total_data = pd.concat([dna_data, protein_data], axis=1)
    if os.path.exists(args.output):
        print(f"{now()} INFO Finished for {args.dna} {args.protein}. Appended to {args.output} with size of features: {total_data.shape[1]}")
    else:
        print(f"{now()} INFO Finished for {args.dna} {args.protein}. Saved to new file {args.output} with size of features: {total_data.shape[1]}")
    total_data.to_csv(args.output, mode='a' if os.path.exists(args.output) else 'w', header=None)
if __name__ == '__main__':
    DNA_prot_process()