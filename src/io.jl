using FastaIO

"""
    read_fasta(file)

Reads a fasta file, returning two lists of matching lengths:
    * headers, containing all headers
    * sequences, containing sequences
"""
function read_fasta(file::String)
    headers = String[]
    sequences = String[]
    multiline_sequence = false
    for line in eachline(file)
        if startswith(line, '>') # new header
            push!(headers, strip(line))
            multiline_sequence = false
        elseif multiline_sequence
            sequences[end] *= strip(line)
        else
            push!(sequences, strip(line))
            multiline_sequence = true
        end
    end
    return headers, sequences
end

# amino acid letter to integer (alphabetical order, gap is last)
const _aa2int = Dict('A' => 1, 'C' => 2, 'D' => 3, 'E' => 4, 'F' => 5,
                     'G' => 6, 'H' => 7, 'I' => 8, 'K' => 9, 'L' => 10,
                     'M' => 11, 'N' => 12, 'P' => 13, 'Q' => 14, 'R' => 15,
                     'S' => 16, 'T' => 17, 'V' => 18, 'W' => 19, 'Y' => 20,
                     '*' => 21, '-' => 21, 'X' => 21)
# nucleotide letter to integer (alphabetical order)
const _nt2int = Dict('A' => 1, 'C' => 2, 'G' => 3, 'T' => 4, 'N' => 5)
@assert allunique(keys(_aa2int))
@assert allunique(keys(_nt2int))
@assert sort(unique(collect(values(_aa2int)))) == collect(1:21)
@assert sort(unique(collect(values(_nt2int)))) == collect(1:5)
# integer to amino acid letter
const _int2aa = Dict(k => aa for (aa, k) in _aa2int)
# integer to nucleotide letter
const _int2nt = Dict(k => aa for (aa, k) in _nt2int)

"convert amino acid sequence string (one letter codes) to an integer vector"
aa2int(s::String) = getindex.(Ref(_aa2int), collect(s))
"convert nucleotide sequence string (one letter codes) to an integer vector"
nt2int(s::String) = getindex.(Ref(_nt2int), collect(s))
"convert integer sequence to amino acid string (one letter codes)"
int2aa(sequence) = join(getindex.(Ref(_int2aa), sequence))
"convert integer sequence to DNA string (one letter codes)"
int2nt(sequence) = join(getindex.(Ref(_int2nt), sequence))

const _amino_acids = "ACDEFGHIKLMNPQRSTVWY" # other symbols "*-X"
const _amino_acids_index = Dict(a => i for (i, a) in enumerate(_amino_acids))

"""
    potts(sequence)

Returns a Potts matrix `A x L` for sequence.
"""
potts(sequence::String) = [get(_amino_acids_index, a, 0) for a in sequence]
potts(sequences::Vector{String}) = hcat(potts.(sequences)...)
