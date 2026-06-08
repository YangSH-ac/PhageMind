#!/home/yshen86/python
import argparse, os, logging
def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, description="Analyze the distribution of clusters and housekeeping genes on contigs, and extract candidate regions based on defined priority rules." \
    "\nThe script takes a protein FASTA file and a TSV file that defines cluster IDs and housekeeping gene annotations, then identifies potential cluster regions while considering the presence of housekeeping genes as boundaries." \
    "\n\nThe priority rules are: " \
    "\n`P1`: There are multiple different housekeeping genes close to each other in one contig. The candidate region is defined as the segment between two different housekeeping genes. The gap is the total count of cluster genes in that segment. Candidates with gap > 4 and <= threshold are considered high confidence, while those with gap <= 3x threshold are considered relaxed." \
    "\n`P2`: There are housekeeping genes close to an end of single contig and there are multiple housekeeping genes globally. The candidate region is defined as the combination of segments on either side of the single housekeeping gene across different contigs. The gap is the total count of cluster genes in the combined segments. Candidates with gap > 4 and <= threshold are considered high confidence, while those with gap <= 3x threshold are considered relaxed." \
    "\n`P3`: The housekeeping genes are far from each other or there is only one housekeeping gene globally. The candidate region is defined as the segment on either side of the single housekeeping gene on the same contig. The gap is the count of cluster genes in that segment. Candidates with gap > 4 and cluster_count/gap > 0.5 are considered high confidence." \
    "\n`P4`: Housekeeping genes cannot be found. The candidate region is defined as the segment around the maximum positive count (cluster gene) on the contig, extended in both directions as long as the gap does not exceed the threshold. Candidates with gap > 4 and cluster_count/gap > 0.5 are considered high confidence, while those that do not meet this ratio but have gap > 4 are considered relaxed.")
    parser.add_argument("-i", "--input", metavar="FILE", required=True, help="Input protein FASTA file")
    parser.add_argument("-t", "--tsv", metavar="FILE", required=True, help="Cluster/housekeeping gene TSV file (first column 'cluster' means mmseqs cluster result, otherwise housekeeping gene abbreviation)")
    parser.add_argument("-o", "--outdir", metavar="PATH", required=True, help="Output directory")
    parser.add_argument("-g", "--gap", metavar="INT", type=int, default=25, help="Max protein count threshold (default 25)")
    parser.add_argument("-s", "--tsvout", metavar="FILE", help="Output TSV file containing gene distribution by housekeeping (abbreviations), clustering (positive counts) and others (negative counts) (optional)")
    parser.add_argument("-r", "--relax", action="store_true", help="Allow `P1` `P2` relaxed candidates with gap up to 3x threshold (`P4` will always output relaxed candidates)")
    parser.add_argument("-l", "--log", metavar="FILE", help="Log file (default <outdir>/<input>.log)")
    return parser.parse_args()
if __name__ == "__main__":
    args = parse_args()
from Bio import SeqIO
import pandas as pd
from collections import defaultdict
def check_options(args):
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input FASTA file not found: {args.input}")
    if not os.path.exists(args.tsv):
        raise FileNotFoundError(f"TSV file not found: {args.tsv}")
    if args.gap <= 5:
        raise ValueError(f"Protein count threshold must be positive number greater than 5: {args.gap}")
    if not args.log:
        args.log = os.path.join(args.outdir, os.path.splitext(os.path.basename(args.input))[0] + ".log")
    return args
def build_distribution(args):
    # 读取 TSV，区分 cluster 和 housekeeping
    cluster_ids = set()
    housekeeping_map = {}
    df = pd.read_csv(args.tsv, sep="\t", header=None)
    for _, row in df.iterrows():
        if row[0].lower() == "cluster":
            cluster_ids.add(row[1].lower())
        else:
            housekeeping_map[row[1].lower()] = row[0]
    contigs = defaultdict(list)
    contig_offsets = {}          # 记录每个 contig 第一个基因的全局索引
    acc = 0 
    current_contig = None
    gene_idx = 0                 # 全局基因计数器，从 0 开始
    for record in SeqIO.parse(args.input, "fasta"):
        coords, annotation = record.description.split("|", 1)
        parts = coords.split("_")
        contig = "_".join(parts[:-2])
        gene_id = record.id
        # 检测 contig 切换，记录偏移量
        if current_contig is None:
            current_contig = contig
            contig_offsets[contig] = gene_idx   # 记录该 contig 的起始基因编号
        elif contig != current_contig:
            if acc != 0:
                contigs[current_contig].append(acc)
                acc = 0
            current_contig = contig
            contig_offsets[contig] = gene_idx   # 新 contig 的起始基因编号
        hk_gene_found = False
        for substr, abbr in housekeeping_map.items():
            if substr.lower() in annotation.lower():
                if acc != 0:
                    contigs[contig].append(acc)
                    acc = 0
                contigs[contig].append(abbr)
                logging.debug(f"Gene {gene_id} on contig {contig} is a housekeeping gene ({abbr}), appended to contigs[{contig}]")
                hk_gene_found = True
                break
        if not hk_gene_found:
            if gene_id.lower() in cluster_ids:
                if acc >= 0:
                    acc += 1
                else:
                    contigs[contig].append(acc)
                    acc = 1
            else:
                if acc <= 0:
                    acc -= 1
                else:
                    contigs[contig].append(acc)
                    acc = -1
        gene_idx += 1            # 每处理一个基因，全局计数 +1
    # 遍历结束后，处理最后一个 contig 的残留
    if acc != 0 and current_contig is not None:
        contigs[current_contig].append(acc)
    return contigs, contig_offsets 
def analyze_priorities(contigs, args):
    """
    根据 contig 的分布，按照优先级规则判断潜在的 cluster 区域。
    返回 (priority_level, results)，其中 results 是列表，每个元素包含 contig、片段索引、gap、cluster_count 等信息。
    """
    results = []
    # 判断全局 housekeeping 数量
    all_hk = set()
    for rep in contigs.values():
        all_hk.update([x for x in rep if isinstance(x, str)])
    has_multi_hk = len(all_hk) >= 2
    # 优先级 1：同一 contig 上有不同 housekeeping
    if has_multi_hk:
        for contig, rep in contigs.items():
            hk = [x for x in rep if isinstance(x, str)]
            if len(set(hk)) >= 2:
                indices = [i for i, x in enumerate(rep) if isinstance(x, str)]
                for i in range(len(indices)-1):
                    start, end = indices[i], indices[i+1]
                    segment = rep[start:end+1]
                    gap = sum(abs(x) for x in segment if isinstance(x, int))
                    cluster_count = sum(x for x in segment if isinstance(x, int) and x > 0)
                    if gap >4 and gap <= args.gap:
                        results.append(("P1", contig, start + 1, end - 1, gap, cluster_count, segment))
                        logging.info(f"Priority 1 candidate on {contig}: segment {segment}, gap={gap}, cluster_count={cluster_count}, proportion={cluster_count/gap*100:.2f}%")
                    elif gap <= 3*args.gap and args.relax:
                        results.append(("P1-relaxed", contig, start + 1, end - 1, gap, cluster_count, segment))
                        logging.info(f"Priority 1 relaxed candidate on {contig}: segment {segment}, gap={gap}, cluster_count={cluster_count}, proportion={cluster_count/gap*100:.2f}%")
        if results:
            return 1, results
    # 优先级 2：同一 contig 上只有 1 个 housekeeping，但全局有多个
        kept_contigs = {c: rep for c, rep in contigs.items() if any(isinstance(x, str) for x in rep)}
        # 预先为每条 contig 计算有效片段（左、右）
        contig_fragments = {}   # { contig_name: [ (side, start, end, gap, cluster_count, segment), ... ] }
        for contig, rep in kept_contigs.items():
            hk_indices = [i for i, x in enumerate(rep) if isinstance(x, str)]   # 所有 housekeeping 的位置
            first_hk = hk_indices[0]
            last_hk = hk_indices[-1]
            fragments = []
            # 左片段：从开头到第一个 housekeeping 之前 
            if first_hk > 0:
                left_seg = rep[0:first_hk + 1]
                gap_left = sum(abs(x) for x in left_seg if isinstance(x, int))
                cluster_left = sum(x for x in left_seg if isinstance(x, int) and x > 0)
                if cluster_left > 0 and gap_left <= 3 * args.gap:
                    fragments.append(("left", 0, first_hk - 1, gap_left, cluster_left, left_seg))
            # 右片段：从最后一个 housekeeping 之后到结尾
            if last_hk < len(rep) - 1:
                right_seg = rep[last_hk:]
                gap_right = sum(abs(x) for x in right_seg if isinstance(x, int))
                cluster_right = sum(x for x in right_seg if isinstance(x, int) and x > 0)
                if cluster_right > 0 and gap_right <= 3 * args.gap:
                    fragments.append(("right", last_hk + 1, len(rep) - 1, gap_right, cluster_right, right_seg))
            if fragments:   # 只有至少保留了一个片段的 contig 才参与配对
                contig_fragments[contig] = fragments
        # 对任意两条 contig 的片段进行组合 
        contig_names = list(contig_fragments.keys())
        for i in range(len(contig_names)):
            for j in range(i + 1, len(contig_names)):
                c1 = contig_names[i]
                c2 = contig_names[j]
                for frag1 in contig_fragments[c1]:
                    side1, start1, end1, gap1, cluster1, seg1 = frag1
                    for frag2 in contig_fragments[c2]:
                        side2, start2, end2, gap2, cluster2, seg2 = frag2
                        total_gap = gap1 + gap2 
                        total_cluster = cluster1 + cluster2
                        if total_gap > 4 and total_gap <= args.gap:
                            results.append(("P2", (c1, c2), (start1, start2), (end1, end2), total_gap, total_cluster, (seg1, seg2)))
                            logging.info(f"Priority 2 candidate: {c1}({side1}) + {c2}({side2}), gap={total_gap}, cluster_count={total_cluster}, proportion={total_cluster/total_gap*100:.2f}%")
                        elif total_gap <= 3 * args.gap and total_cluster > 0 and args.relax:
                            results.append(("P2-relaxed", (c1, c2), (start1, start2), (end1, end2), total_gap, total_cluster, (seg1, seg2)))
                            logging.info(f"Priority 2-relaxed candidate: {c1}({side1}) + {c2}({side2}), gap={total_gap}, cluster_count={total_cluster}, proportion={total_cluster/total_gap*100:.2f}%")
        if results:
            return 2, results
    kept_contigs = {c: rep for c, rep in contigs.items() if any(isinstance(x, str) or (isinstance(x, int) and x > 0) for x in rep)}
    # 优先级 3：housekeeping 独立或只有 1 个
    for contig, rep in kept_contigs.items():
        hk = [i for i, x in enumerate(rep) if isinstance(x, str)]
        for idx in hk:
            # ---- 左侧扩展 ----
            seg_left = [rep[idx]]
            gap_left = 0 
            start_left = idx
            end_left = idx - 1
            start_right = idx + 1
            end_right = idx
            p = idx - 1 
            while p >= 0:
                x = rep[p]
                if isinstance(x, str):          # 遇到 housekeeping，停止 
                    break 
                add_gap = abs(x)
                if gap_left + add_gap <= args.gap:
                    seg_left.insert(0, x)       # 左侧插入，保持从左到右顺序 
                    gap_left += add_gap 
                    start_left = p
                    p -= 1 
                else:
                    if x > 0:                   # 正数，强行包含 
                        seg_left.insert(0, x)
                        gap_left += add_gap 
                        end_left = p
                    break                       # 无论是否包含都停止 
            # ---- 右侧扩展 ----
            seg_right = [rep[idx]]
            gap_right = 0 
            p = idx + 1 
            while p < len(rep):
                x = rep[p]
                if isinstance(x, str):
                    break 
                add_gap = abs(x)
                if gap_right + add_gap <= args.gap:
                    seg_right.append(x)
                    gap_right += add_gap 
                    end_right = p
                    p += 1 
                else:
                    if x > 0:
                        seg_right.append(x)
                        gap_right += add_gap 
                        end_right = p
                    break 
            # ---- 对左右片段分别处理 ----
            for seg, side, gap, start, end in [(seg_left, "left", gap_left, start_left, end_left), (seg_right, "right", gap_right, start_right, end_right)]:
                cluster = sum(v for v in seg if isinstance(v, int) and v > 0)
                # 尝试修剪直到满足条件 
                cur_seg = seg[:]
                cur_gap = gap 
                cur_cluster = cluster 
                if start >= end:  # 只有 housekeeping 没有其他基因，跳过修剪直接判断
                    continue
                while cur_seg:
                    if cur_gap < 5:
                        break
                    if cur_cluster > 0 and cur_gap < 1.5 * cur_cluster:
                        logging.info(f"Priority 3 candidate on {contig} {side} segment {cur_seg}, gap={cur_gap}, cluster_count={cur_cluster}, proportion={cur_cluster/cur_gap*100:.2f}%")
                        results.append(("P3", contig, start, end, cur_gap, cur_cluster, cur_seg))
                        break  # 找到满足条件的片段，跳出修剪循环 
                    # 尝试移除末尾的一正一负对 
                    if len(cur_seg) >= 2:
                        # 取最后两个元素，判断是否含有一个正数和一个负数 
                        if side == "left":
                            vals = [cur_seg[0], cur_seg[1]]
                        else:
                            vals = [cur_seg[-1], cur_seg[-2]]
                        pos_val = None 
                        neg_val = None 
                        for v in vals:
                            if isinstance(v, int):
                                if v > 0:
                                    pos_val = v 
                                elif v < 0:
                                    neg_val = v 
                        if pos_val is not None and neg_val is not None:
                            # 移除这两个元素 
                            if side == "left":
                                cur_seg = cur_seg[2:]
                                start += 2
                            else:
                                cur_seg = cur_seg[:-2]
                                end -= 2
                            cur_gap -= (abs(pos_val) + abs(neg_val))
                            cur_cluster -= pos_val 
                            continue 
                    # 无法再移除一对，退出 
                    break 
    if results:
        return 3, results 
    # 优先级 4：完全没有 housekeeping
    for contig, rep in contigs.items():
        # 找到最大的正整数及其索引（取第一个最大值）
        max_val = 0 
        max_idx = -1
        for i, x in enumerate(rep):
            if isinstance(x, int) and x > 0 and x > max_val:
                max_val = x
                max_idx = i
        if max_idx == -1:
            continue
        # 初始化：起点为最大正整数
        seg = [max_val]
        gap = abs(max_val)          # 起点计入 gap
        cluster = max_val 
        left_ptr = max_idx - 1 
        right_ptr = max_idx + 1
        start = end = max_idx
        # 双向成对扩展，每次选择比值更优的一端 
        while True:
            # ---- 评估左侧候选对 ----
            left_available = False
            left_ratio = 0
            left_neg = left_pos = 0 
            if left_ptr >= 1:
                x1, x2 = rep[left_ptr - 1], rep[left_ptr]
                if not isinstance(x1, str) and not isinstance(x2, str):
                    if x1 > 0 and x2 < 0:
                        pos, neg = x1, x2
                    elif x1 < 0 and x2 > 0:
                        pos, neg = x2, x1
                    else:
                        pos, neg = None, None 
                    if pos is not None and neg is not None:
                        if gap + abs(neg) <= args.gap:
                            left_available = True
                            left_ratio = pos / abs(neg)
                            left_neg, left_pos = neg, pos 
            # ---- 评估右侧候选对 ----
            right_available = False
            right_ratio = 0 
            right_neg = right_pos = 0
            if right_ptr < len(rep) - 1:
                x1, x2 = rep[right_ptr], rep[right_ptr + 1]
                if not isinstance(x1, str) and not isinstance(x2, str):
                    if x1 > 0 and x2 < 0:
                        pos, neg = x1, x2
                    elif x1 < 0 and x2 > 0:
                        pos, neg = x2, x1
                    else:
                        pos, neg = None, None
                    if pos is not None and neg is not None:
                        if gap + abs(neg) <= args.gap:
                            right_available = True
                            right_ratio = pos / abs(neg)
                            right_neg, right_pos = neg, pos
            # ---- 选择一端扩展 ----
            if not left_available and not right_available:
                break 
            if left_available and right_available:
                if left_ratio >= right_ratio:
                    choose_left = True
                else:
                    choose_left = False 
            elif left_available:
                choose_left = True
            else:
                choose_left = False
            if choose_left:
                # 向左扩展：将一对插入片段头部
                seg = [rep[left_ptr - 1], rep[left_ptr]] + seg
                gap += abs(left_neg) + abs(left_pos)
                cluster += left_pos 
                left_ptr -= 2
                start = left_ptr + 1
            else:
                # 向右扩展：将一对追加到片段尾部
                seg.extend([rep[right_ptr], rep[right_ptr + 1]])
                gap += abs(right_neg) + abs(right_pos)
                cluster += right_pos
                right_ptr += 2
                end = right_ptr - 1
        # 过滤：gap < 5 不输出
        if gap < 5:
            continue
        cur_seg = seg[:]
        cur_gap = gap
        cur_cluster = cluster
        if cur_cluster > 0 and cur_gap < 1.5 * cur_cluster:
            logging.info(f"Priority 4 candidate on {contig}: max_val index {max_idx}, segment {cur_seg}, gap={cur_gap}, cluster_count={cur_cluster}, proportion={cur_cluster/cur_gap*100:.2f}%")
            results.append(("P4", contig, start, end, cur_gap, cur_cluster, cur_seg))
        else:
            logging.info(f"Priority 4 relaxed candidate on {contig}: max_val index {max_idx}, segment {cur_seg}, gap={cur_gap}, cluster_count={cur_cluster}, proportion={cur_cluster/cur_gap*100:.2f}%")
            results.append(("P4_relaxed", contig, start, end, cur_gap, cur_cluster, cur_seg))
    if results:
        return 4, results
    return None, []
def extract_region_fasta(contig, start, end, contigs, contig_offsets, fasta_path):
    """
    利用 contig_offsets 快速定位区间，从 FASTA 文件中提取序列。
    contig: contig 名称
    start: rep 中片段起始索引（包含）
    end:   rep 中片段结束索引（包含）
    """
    rep = contigs[contig]
    gene_counts = [abs(x) if isinstance(x, int) else 1 for x in rep]
    offset = sum(gene_counts[:start])                # contig 内部偏移 
    length = sum(gene_counts[start:end+1])           # 要提取的基因数
    if length == 0:
        return ""
    abs_start = contig_offsets[contig] + offset      # 全局起始基因编号
    abs_end = abs_start + length - 1                 # 全局结束基因编号
    fasta_lines = []
    for i, record in enumerate(SeqIO.parse(fasta_path, "fasta")):
        if i < abs_start:
            continue
        if i > abs_end:
            break
        fasta_lines.append(f">{record.description}\n{str(record.seq)}")
    return "\n".join(fasta_lines)
def main(args):
    os.makedirs(args.outdir, exist_ok=True)
    if os.path.exists(args.log): os.remove(args.log)
    logging.basicConfig(filename=args.log, level=logging.DEBUG if args.debug else logging.INFO, format="[%(asctime)s]%(levelname)s %(message)s")
    filename, sufname = os.path.splitext(os.path.basename(args.input))
    logging.info(f"Starting analysis for input: {args.input}, TSV: {args.tsv}, Output directory: {args.outdir}, Gap threshold: {args.gap}")
    logging.info(f"Log file: {args.log}, TSV output: {args.tsvout if args.tsvout else 'None'}")
    contigs, contig_offsets = build_distribution(args)
    # 输出 TSV
    if args.tsvout:
        with open(args.tsvout, "w") as f:
            for contig, rep in contigs.items():
                f.write(contig + "\t" + "\t".join(map(str, rep)) + "\n")
    # 分析优先级
    priority, results = analyze_priorities(contigs, args)
    if priority is None:
        logging.info("No candidate regions found.")
        return
    logging.info(f"Found {len(results)} candidate regions.")
    # 输出结果
    for idx, res in enumerate(results):
        level, contig_info, start, end, gap, cluster_count, segment = res
        # 提取 FASTA
        faafile = os.path.join(args.outdir, f"{filename}_{level}_{idx+1}{sufname}")
        if level.startswith("P2"):
            contig1, contig2 = contig_info
            seg1, seg2 = segment
            fasta1 = extract_region_fasta(contig_info[0], start[0], end[0], contigs, contig_offsets, args.input)
            fasta2 = extract_region_fasta(contig_info[1], start[1], end[1], contigs, contig_offsets, args.input)
            with open(faafile, "w") as f:
                f.write(fasta1)
                f.write("\n")
                f.write(fasta2)
            logging.info(f"{level} candidate {idx+1}: {contig1}({start[0]}-{end[0]}), {contig2}({start[1]}-{end[1]}), gap={gap}, cluster_count={cluster_count}, proportion={cluster_count/gap*100:.2f}%, seg1={seg1}, seg2={seg2} saved to {faafile}")
        else:
            fasta = extract_region_fasta(contig_info, start, end, contigs, contig_offsets, args.input)
            with open(faafile, "w") as f:
                f.write(fasta)
            logging.info(f"{level} candidate {idx+1}: {contig_info}({start}-{end}), gap={gap}, cluster_count={cluster_count}, proportion={cluster_count/gap*100:.2f}%, segment={segment} saved to {faafile}")
if __name__ == "__main__":
    args = check_options(args)
    main(args)