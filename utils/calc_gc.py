in_fn='data/yeast_promoters.txt'
with open(in_fn,'r') as f:

	count=dict()

	nucleotides = ['A','T','G','C']
	for k in nucleotides:
		count[k]=0

	n_lines=0
	for line in f:
		n_lines+=1
		split_line = line.strip().split()
		for char in split_line[1]:
			if char=='A':
				count['A']+=1
			elif char=='T':
				count['T']+=1
			elif char=='G':
				count['G']+=1
			else:
				count['C']+=1

	n_bases=sum(count.values())
	assert n_lines*1000 == n_bases

	freq=dict()
	for k in nucleotides:
		freq[k]=count[k]/float(n_bases)

	print freq

# {'A': 0.3099365410397736, 'C': 0.19293430656934307, 'T': 0.30830716520184714, 'G': 0.1888219871890362}