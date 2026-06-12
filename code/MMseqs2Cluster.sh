#!/bin/bash

usage() {
  [[ -n "$1" ]] && echo "Error: $1" >&2
  cat >&2 << USAGE0
 Usage: $0 -i input -d database -o outdir [-m mmseqs][-p prefix][-c options][-q][-r][-h]

 Run MMseqs2 to cluster input fasta sequences against a database fasta, outputting a TSV of clustered input sequence IDs. 
 Output TSV (saved to <outdir>/<input>_cluster.tsv) format: "cluster	input_sequence_id" and can be used for downstream analysis.
 Log file is saved to <outdir>/mmseqs2cluster.log. MMseqs2 log is saved to <outdir>/mmseqs_running.log.
 
 Options:
  -i  REQUIRED, Input fasta file
  -d  REQUIRED, Database fasta file
  -o  REQUIRED, Output directory
  -m  Path to mmseqs2 executable (default: mmseqs)
  -p  Database annotation prefix (default: COLLECTED_O_ANTIGEN_CLUSTER_DATABASE)
  -c  Extra options for mmseqs cluster
  -q  Database already has prefix
  -r  Remove mmseqs intermediate files, keep only TSV and final list
  -h  Show this help message and exit
USAGE0
  exit 1
}
optc() { export $1="$OPTARG"; }
optd() { export $1="${OPTARG%/}"; }
opte() { export $1="Yes"; }
declare -A optfun=(["i"]="optc input" ["d"]="optc database" ["o"]="optd outdir" ["m"]="optc mmseqs" ["c"]="optc options" ["p"]="optc prefix" ["q"]="opte q" ["r"]="opte r" ["h"]="usage")
which mmseqs &>/dev/null && mmseqs="mmseqs"
prefix="COLLECTED_O_ANTIGEN_CLUSTER_DATABASE"
while getopts ":i:d:o:m:c:p:qrh" opt; do
  [[ -n "${optfun[$opt]}" ]] && ${optfun[$opt]} || usage "Option -$OPTARG invalid or requires argument."
done
[[ -z "$input" || -z "$database" || -z "$outdir" ]] && usage "Options -i, -d, -o are required."
[[ -s "$input" ]] || usage "Input fasta file (-i) $input does not exist or is empty."
[[ -s "$database" ]] || usage "Database fasta file (-d) $database does not exist or is empty."
[[ -z "$mmseqs" || ! -x "$mmseqs" ]] && usage "MMseqs2 executable (-m) $mmseqs not found or not executable."

mmseqsdir="$outdir/mmseqs"
mkdir -p "$mmseqsdir"
log="$outdir/mmseqs2cluster.log"
exec > >(while IFS= read -r line; do echo "[$(date '+%F %T')]$line" | tee -a "$log"; done) 2>&1
echo "INFO: Starting MMseqs2 clustering for $input against $database with prefix $prefix${options:+and options $options}."
echo "INFO: Output directory: $outdir, Log: $log, MMseqs: $mmseqs."
echo "INFO: Database already has prefix: ${q:-No}, Remove intermediate files: ${r:-No}"
dbfile="$mmseqsdir/database.fa"
[[ -n "$q" ]] && cp "$database" "$dbfile" || sed "s/^>/>${prefix}|/" "$database" > "$dbfile"
inname=$(basename "${input%.*}")
tmpfile="$mmseqsdir/combined.fa"
cat "$input" "$dbfile" > "$tmpfile"

mmseqs_log="$outdir/mmseqs_running.log"
echo "INFO: Combined input and database into $tmpfile, starting MMseqs2 clustering. MMseqs2 log saved to $mmseqs_log"
$mmseqs createdb "$tmpfile" "$mmseqsdir/db" &> "$mmseqs_log"
$mmseqs cluster "$mmseqsdir/db" "$mmseqsdir/clu" "$mmseqsdir/pref" --threads 1 $options &>> "$mmseqs_log"
tsvfile="$mmseqsdir/${inname}_cluster.tsv" &>> "$mmseqs_log"
$mmseqs createtsv "$mmseqsdir/db" "$mmseqsdir/db" "$mmseqsdir/clu" "$tsvfile" &>> "$mmseqs_log"

echo "INFO: MMseqs2 clustering completed, processing results in $tsvfile to extract input sequence IDs clustered with database sequences."
tsvfilefinal="$outdir/${inname}_cluster.tsv"
awk -v pref="$prefix" '
{
  has_db=0; has_in=0;
  for(i=1;i<=NF;i++){
    if($i ~ "^"pref) has_db=1; else has_in=1;
  }
  if(has_db && has_in){
    for(i=1;i<=NF;i++){
      if($i !~ "^"pref) print "cluster\t"$i;
    }
  }
}' "$tsvfile" | sort -u > "$tsvfilefinal"
echo "INFO: Final list written to $tsvfilefinal, total $(wc -l < "$tsvfilefinal") unique input sequences clustered with database sequences."

[[ -n "$r" ]] && rm -rf $mmseqsdir/db* $mmseqsdir/clu* "$mmseqsdir/pref" "$mmseqsdir/combined.fa" "$mmseqsdir/database.fa"
