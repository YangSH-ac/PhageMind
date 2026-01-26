import argparse, time
def fuzzy_match(header, pattern):
    return pattern.lstrip('>').strip().lower() in header.lstrip('>').strip().lower()
def curtime():
    return time.strftime('[%Y-%m-%d %H:%M:%S]', time.localtime(time.time()))
def process_fasta(input_file, output_file, pattern1, pattern2, keep, pattern_list, reverse, append):
    blocks = []
    current_block = None
    with open(input_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            if line.startswith('>'):
                if current_block:
                    blocks.append(current_block)
                current_block = {'header': line.strip(), 'start_line': line_num, 'lines': [line]}
            else:
                if current_block:
                    current_block['lines'].append(line)
        if current_block:
            blocks.append(current_block)
    if pattern_list:
        print(f"{curtime()} INFO Start processing {input_file}, {'append' if append else 'write'} to {output_file}, from list {pattern_list} {'exclude' if reverse else 'extract'}")
        with open(pattern_list, 'r') as f_list:
            patterns = [p.strip() for p in f_list.readlines() if p.strip()]
        if reverse:
            matched_blocks = [b for b in blocks if not any(fuzzy_match(b['header'], pattern) for pattern in patterns)]
        else:
            matched_blocks = [b for b in blocks if any(fuzzy_match(b['header'], pattern) for pattern in patterns)]
        if not matched_blocks:
            print(f"{curtime()} WARNING {input_file} no matching annotation lines found from {pattern_list}")
            return False 
        with open(output_file, 'a' if append else 'w') as f_out:
            for b in matched_blocks:
                f_out.writelines(b['lines']) 
        print(f"{curtime()} INFO {input_file} successfully extracted {len(matched_blocks)} blocks {'appended' if append else 'written'} to {output_file}")
        return True
    print(f"{curtime()} INFO Start processing {input_file}, {'append' if append else 'write'} to {output_file}, extract from {pattern1} to {pattern2} {'including boundaries' if keep else ''}")
    matches1 = [b for b in blocks if fuzzy_match(b['header'], pattern1)]
    matches2 = [b for b in blocks if fuzzy_match(b['header'], pattern2)]
    errors = [] # Error detection
    if not matches1:
        errors.append(f"{input_file} no matching annotation line found for {pattern1}")
    elif len(matches1) > 1:
        lines = ', '.join(str(b['start_line']) for b in matches1)
        print(f"{curtime()} WARNING {input_file} multiple matches found for {pattern1} (lines: {lines}), will attempt to extract minimal interval")
    if not matches2:
        errors.append(f"{input_file} no matching annotation line found for {pattern2}")
    elif len(matches2) > 1:
        lines = ', '.join(str(b['start_line']) for b in matches2)
        print(f"{curtime()} WARNING {input_file} multiple matches found for {pattern2} (lines: {lines}), will attempt to extract minimal interval")
    if errors:
        print(f"{curtime()} ERROR Errors occurred while processing {input_file}:")
        print('\n'.join(errors))
        return False
    best_pair = min(((s, e) for s in matches1 for e in matches2), key=lambda x: abs(blocks.index(x[0]) - blocks.index(x[1])))
    idx1, idx2 = sorted([blocks.index(best_pair[0]), blocks.index(best_pair[1])])
    if blocks.index(best_pair[0]) > blocks.index(best_pair[1]):
        print(f"{curtime()} INFO {input_file} order reversed: {pattern1} comes after {pattern2}")
    if keep:
        idx1, idx2 = max(-1, idx1-1), min(len(blocks), idx2+1)
    with open(output_file, 'a' if append else 'w') as f_out: # Extract content and output
        for block in blocks[idx1+1:idx2]:
            f_out.writelines(block['lines'])
    print(f"{curtime()} INFO {input_file} successfully extracted {idx1}-{idx2}: {idx2-idx1-1} blocks {'appended' if append else 'written'} to {output_file}")
    return True
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='FASTA annotation extraction tool')
    parser.add_argument('-i', '--input', metavar='FILE', required=True, help='Input FASTA file')
    parser.add_argument('-o', '--output', metavar='FILE', required=True, help='Output file path')
    parser.add_argument('-a', '--append', action='store_true', help='Append results to output file')
    group1 = parser.add_argument_group(title='Interval extraction')
    group1.add_argument('-s', '--start', metavar='STR', help='Start annotation pattern')
    group1.add_argument('-e', '--end', metavar='STR', help='End annotation pattern')
    group1.add_argument('-k', '--keep', action='store_true', help='Keep start and end genes')
    group2 = parser.add_argument_group(title='List extraction')
    group2.add_argument('-l', '--list', metavar='FILE', help='Pattern list file')
    group2.add_argument('-r', '--reverse', action='store_true', help='Exclude list patterns')
    args = parser.parse_args()
    if args.list and (args.start or args.end):
        parser.error("List extraction -l cannot be used together with interval extraction -s/-e")
    if not args.list and not (args.start and args.end):
        parser.error("Interval extraction -s/-e requires both parameters")
    if not args.list and not args.start and not args.end:
        parser.error("Either interval extraction or list extraction must be provided")
    success = process_fasta(args.input, args.output, args.start, args.end, args.keep, args.list, args.reverse, args.append)
    if not success:
        print(f"{curtime()} ERROR Processing failed, please check input parameters")