import argparse
import pandas as pd
def main():
    parser = argparse.ArgumentParser(description="Convert bacteria-phage matrix to long-format CSV")
    parser.add_argument("-i", "--input", required=True, help="Input file (TSV by default)")
    parser.add_argument("-o", "--output", required=True, help="Output CSV file")
    parser.add_argument("-c", "--csv", action="store_true", help="Input file is CSV instead of TSV")
    parser.add_argument("-r", "--reverse", action="store_true", help="Interpret columns as bacteria and rows as phages")
    args = parser.parse_args()
    sep = "," if args.csv else "\t"
    df = pd.read_csv(args.input, sep=sep, index_col=0)
    if args.reverse:
        df = df.T
    rows = []
    for i, bact in enumerate(df.index):       # bacteria rows
        for j, phage in enumerate(df.columns): # phage columns
            val = df.loc[bact, phage]
            if pd.isna(val):
                continue
            try:
                num = float(val)
            except Exception:
                continue  # skip non-numeric
            new_val = 1 if num > 0 else 0
            rows.append([bact, phage, num, new_val, i, j])
    out_df = pd.DataFrame(rows, columns=["Bacteria", "Phage", "OriginalValue", "BinaryValue", "BacteriaIndex", "PhageIndex"])
    out_df.to_csv(args.output, index=False, header=False)
if __name__ == "__main__":
    main()