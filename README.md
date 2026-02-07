# HMM POS Tagger - AID-829 Assignment 1

Implementation of a Part-of-Speech tagger using Hidden Markov Models and the Viterbi algorithm.

## Project Structure

```
hmm_project/
├── src/
│   ├── data_parser.py       # Parse CoNLL-U format files
│   ├── hmm_trainer.py       # Train HMM (estimate probabilities)
│   ├── viterbi_decoder.py   # Viterbi algorithm with explicit matrices
│   ├── evaluator.py         # Evaluation metrics
│   └── main.py              # Main pipeline script
├── data/
│   ├── en_ewt-ud-train.conllu
│   └── en_ewt-ud-test.conllu
├── models/
│   └── hmm_model.pkl        # Saved model
├── output/
│   ├── results.txt          # Evaluation metrics
│   └── predictions.txt      # Detailed predictions
└── README.md
```

## Requirements

- Python 3.7+
- numpy
- No external NLP libraries (implemented from scratch)

```bash
pip install numpy
```

## Dataset

Download the Universal Dependencies English-EWT dataset:

```bash
cd data
wget https://raw.githubusercontent.com/UniversalDependencies/UD_English-EWT/master/en_ewt-ud-train.conllu
wget https://raw.githubusercontent.com/UniversalDependencies/UD_English-EWT/master/en_ewt-ud-test.conllu
```

Or clone the repository:
```bash
git clone https://github.com/UniversalDependencies/UD_English-EWT
cp UD_English-EWT/*.conllu data/
```

## Usage

### Run Complete Pipeline

```bash
cd src
python main.py --train ../data/en_ewt-ud-train.conllu --test ../data/en_ewt-ud-test.conllu
```

### Run Individual Components

**1. Parse Data**
```bash
python data_parser.py ../data/en_ewt-ud-train.conllu
```

**2. Train HMM**
```bash
python hmm_trainer.py ../data/en_ewt-ud-train.conllu
```

**3. Decode with Viterbi**
```bash
python viterbi_decoder.py ../models/hmm_model.pkl ../data/en_ewt-ud-test.conllu
```

**4. Evaluate**
```bash
python evaluator.py ../models/hmm_model.pkl ../data/en_ewt-ud-test.conllu
```

## Implementation Details

### HMM Parameters

**Initial Probabilities** - P(tag)
```
P(tag) = Count(tag at sentence start) / Total sentences
```

**Transition Probabilities** - P(tag_j | tag_i)
```
P(tag_j | tag_i) = Count(tag_i → tag_j) / Count(tag_i)
```

**Emission Probabilities** - P(word | tag)
```
P(word | tag) = Count(tag emits word) / Count(tag)
```

### Viterbi Algorithm

**Matrices:**
- `V[tag][t]` - Viterbi probability matrix (log probabilities)
- `B[tag][t]` - Backpointer matrix (stores previous tag)

**Steps:**
1. **Initialization**: `V[tag][0] = log P(tag) + log P(word_0 | tag)`
2. **Recursion**: For each position t, compute max probability
3. **Termination**: Find tag with max probability at final position
4. **Backtracking**: Follow backpointers to recover best sequence

### Unknown Word Handling

- Words appearing ≤2 times are treated as rare
- Replaced with `<UNK>` token during training
- Laplace smoothing applied to all probabilities

## Expected Results

**Accuracy**: ~90-94% on UD-EWT test set

## Output Files

**results.txt** - Summary statistics
```
Total tokens:     25,096
Correct tokens:   23,147
Accuracy:         0.9223 (92.23%)
```

**predictions.txt** - Detailed predictions for each sentence
```
Sentence 1:
from    ADP     ADP     ✓
the     DET     DET     ✓
ap      PROPN   PROPN   ✓
...
```

## Assignment Compliance

✓ No use of pre-built POS taggers or HMM libraries  
✓ Explicit Viterbi matrix construction  
✓ Explicit backpointer matrix  
✓ From-scratch implementation  
✓ Accuracy evaluation  

## Libraries Used (Allowed)

- `numpy` - For matrix operations
- `collections` - For Counter, defaultdict
- `pickle` - For model serialization

## Author

[Your Name]  
AID-829 Assignment 1  
February 2026
