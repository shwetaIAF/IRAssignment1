# 📚 Information Retrieval System using Streamlit

## 📌 Project Overview

This project is a comprehensive **Information Retrieval (IR) System** developed using **Python** and **Streamlit**.  
The application demonstrates the implementation of several fundamental and advanced Information Retrieval concepts through an interactive and visually rich user interface.

The system allows users to:

- Upload and analyze document collections
- Perform text preprocessing
- Generate inverted indexes
- Compare stemming and lemmatization
- Execute phrase queries
- Compare BST and B-Tree search structures
- Perform tolerant retrieval using spelling correction and wildcard matching
- Visualize experimental analytics through interactive dashboards

The project was designed with:
- Modular architecture
- Production-style coding standards
- Descriptive docstrings
- Strong experimental analysis
- Interactive visualizations

---

# 🎯 Objectives

The primary objectives of this project are:

1. Understand Information Retrieval fundamentals
2. Implement indexing and retrieval mechanisms
3. Compare retrieval optimization techniques
4. Demonstrate phrase query processing
5. Analyze dictionary search structures
6. Implement tolerant retrieval mechanisms
7. Visualize experimental performance metrics
8. Build a scalable and modular IR application

---

# 🏗️ System Architecture

## Project Structure

```text
IR/
│
├── app.py
├── requirements.txt
├── README.md
│
├── data/
│   └── sample_dataset.csv
│
├── modules/
│   ├── bst.py
│   ├── btree.py
│   ├── constants.py
│   ├── data_loader.py
│   ├── indexing.py
│   ├── phrase_query.py
│   ├── preprocessing.py
│   ├── retrieval.py
│   ├── tolerant_retrieval.py
│   └── utils.py
```

---

# 🧠 Core Concepts Implemented

| Concept | Description |
|---|---|
| Text Preprocessing | Tokenization, stopword removal, normalization |
| Inverted Index | Efficient term-to-document mapping |
| Stemming | Reduces words to root forms |
| Lemmatization | Converts words into dictionary forms |
| Phrase Query Processing | Exact phrase retrieval |
| Biword Index | Stores adjacent term pairs |
| Positional Index | Stores term positions |
| BST Search | Binary Search Tree-based dictionary search |
| B-Tree Search | Balanced multi-level dictionary search |
| Tolerant Retrieval | Handles spelling and wildcard queries |
| K-Gram Index | Supports approximate matching |
| Edit Distance | Computes spelling similarity |
| Experimental Analytics | Visualization of retrieval performance |

---

# ⚙️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Streamlit | Interactive web application |
| Pandas | Data processing |
| NLTK | Natural Language Processing |
| Plotly | Interactive charts |
| Typing Module | Static type annotations |

---

# 📄 Dataset Handling

The application supports:

- Default sample datasets
- User-uploaded CSV datasets

## Expected CSV Format

```csv
doc_id,text
1,"Machine learning improves search systems."
2,"Information retrieval uses indexing."
```

---

# ⚙️ Text Preprocessing

## Overview

Text preprocessing is the foundational stage of any Information Retrieval system.  
The objective is to normalize raw textual content into machine-processable tokens.

---

## Implemented Techniques

### 1. Tokenization

Tokenization splits text into individual terms or tokens.

### Example

Input:
```text
Machine learning improves retrieval.
```

Output:
```python
["Machine", "learning", "improves", "retrieval"]
```

---

### 2. Lowercasing

Converts all tokens into lowercase format.

### Example

```python
["machine", "learning"]
```

---

### 3. Stopword Removal

Common words such as:
- is
- the
- are
- and

are removed because they contribute little semantic meaning.

---

### 4. Stemming

Stemming aggressively reduces words to root forms.

| Original Word | Stemmed |
|---|---|
| retrieval | retriev |
| learning | learn |
| studies | studi |

### Advantages
- Faster indexing
- Reduced vocabulary size

### Limitations
- Can generate non-dictionary words

---

### 5. Lemmatization

Lemmatization converts words into meaningful dictionary forms.

| Original Word | Lemmatized |
|---|---|
| studies | study |
| running | run |
| retrievals | retrieval |

### Advantages
- Better semantic accuracy
- Cleaner normalization

### Limitations
- Slightly slower than stemming

---

# 📚 Inverted Index

## Overview

An inverted index maps terms to the documents containing them.

### Example

```python
{
    "machine": [1, 6, 8],
    "retrieval": [2, 7, 9]
}
```

---

## Importance

Inverted indexes:
- Improve retrieval speed
- Reduce sequential document scanning
- Form the backbone of search engines

---

# 🔍 Retrieval Comparison

## Objective

Compare:
- Stemming-based retrieval
- Lemmatization-based retrieval

---

## Experimental Workflow

1. Preprocess query
2. Normalize terms
3. Perform retrieval
4. Compare results
5. Measure execution time

---

## Observations

### Stemming
- Broader retrieval coverage
- Faster normalization
- Less semantic precision

### Lemmatization
- Better semantic quality
- Cleaner vocabulary
- More meaningful retrieval

---

# 🧠 Phrase Query Processing

## Overview

Phrase queries retrieve documents containing exact word sequences.

### Example Query

```text
machine learning
```

---

# 📌 Biword Index

## Concept

Biword indexing stores adjacent term pairs.

### Example

Document:
```text
machine learning techniques
```

Generated biwords:
```text
machine learning
learning techniques
```

---

## Structure

```python
{
    "machine learning": [1, 6],
    "learning techniques": [6]
}
```

---

## Advantages

- Faster phrase retrieval
- Simpler phrase matching

---

## Limitations

- False positives for long phrases
- Limited adjacency validation

---

# 📍 Positional Index

## Concept

Stores term positions within documents.

### Structure

```python
{
    "machine": {
        1: [0],
        6: [0]
    }
}
```

---

## Advantages

- Accurate phrase matching
- Exact adjacency verification
- Supports proximity search

---

## Comparison

| Feature | Biword Index | Positional Index |
|---|---|---|
| Speed | Faster | Moderate |
| Accuracy | Lower | Higher |
| Phrase Validation | Partial | Exact |

---

# 🌳 BST vs B-Tree Search

## Objective

Compare dictionary search performance between:
- Binary Search Tree
- B-Tree

---

# 🌲 Binary Search Tree (BST)

## Characteristics

- Hierarchical structure
- Left child < parent
- Right child > parent

---

## Advantages

- Simple implementation
- Efficient small-scale search

---

## Limitations

- Can become unbalanced
- Performance degradation possible

---

# 🌴 B-Tree

## Characteristics

- Balanced multi-level structure
- Optimized for disk-based retrieval
- Stores multiple keys per node

---

## Advantages

- Better scalability
- Balanced search paths
- Improved search consistency

---

## Experimental Observation

B-Trees generally provide:
- More stable search performance
- Better scalability for larger vocabularies

---

# 🛠️ Tolerant Retrieval

## Overview

Tolerant retrieval handles:
- Misspellings
- Partial terms
- Approximate queries
- Wildcard searches

---

# 🔤 K-Gram Index

## Concept

A K-Gram index breaks terms into substrings of length K.

### Example

Term:
```text
retrieval
```

3-Grams:
```text
$ret
ret
etr
tri
rie
iev
eva
val
al$
```

---

## Applications

- Wildcard matching
- Spelling correction
- Approximate search

---

# ✏️ Edit Distance

## Overview

Edit distance computes similarity between terms using:
- Insertion
- Deletion
- Substitution

---

## Example

```text
retrival → retrieval
```

Edit Distance:
```text
1
```

---

# 🔎 Wildcard Search

## Example

Input:
```text
retr*
```

Matches:
```text
retrieval
retrieve
retrieved
retrieves
```

---

# 📊 Experimental Analytics Dashboard

The application includes interactive visualizations using Plotly.

---

## Visualizations Included

### 1. Retrieval Comparison

Compares:
- Stemming
- Lemmatization

---

### 2. Phrase Query Comparison

Compares:
- Biword Index
- Positional Index

---

### 3. Dictionary Search Performance

Compares:
- BST Search Time
- B-Tree Search Time

---

# 🎨 User Interface Features

## Implemented UI Enhancements

- Multi-tab navigation
- Colored section headers
- Interactive charts
- Metrics visualization
- Responsive layouts
- Experimental dashboards

---

# 🚀 Installation Guide

## Step 1 — Create Virtual Environment

### macOS / Linux

```zsh
python3 -m venv venv_ir
```

---

## Step 2 — Activate Virtual Environment

```zsh
source venv_ir/bin/activate
```

---

## Step 3 — Install Dependencies

```zsh
pip install -r requirements.txt
```

---

## Step 4 — Download NLTK Resources

```python
import nltk

nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")
```

---

# ▶️ Running the Application

```zsh
streamlit run app.py
```

---

# 📦 requirements.txt

```text
streamlit
pandas
nltk
plotly
```

---

# 📸 Screenshots

## Suggested Screenshots

1. Main Dashboard
2. Dataset Viewer
3. Text Preprocessing
4. Inverted Index
5. Retrieval Comparison
6. Phrase Query Processing
7. BST vs B-Tree
8. Tolerant Retrieval
9. Analytics Dashboard

---

# 🧪 Experimental Analysis Summary

| Experiment | Observation |
|---|---|
| Stemming vs Lemmatization | Stemming retrieves broader matches |
| Phrase Query Processing | Positional indexing provides higher accuracy |
| BST vs B-Tree | B-Tree provides more balanced search |
| Tolerant Retrieval | Improves usability for imperfect queries |

---

# 🔮 Future Enhancements

Potential future improvements include:

- TF-IDF ranking
- BM25 ranking
- Vector databases
- Semantic search
- Transformer embeddings
- Hybrid retrieval
- Elasticsearch integration
- RAG-based retrieval systems

---

# 📖 References

1. Manning, Christopher D. — *Introduction to Information Retrieval*
2. NLTK Documentation
3. Streamlit Documentation
4. Plotly Documentation
5. Python Official Documentation

---

# 👨‍💻 Authors

- **KAMESWARA RAO K.**
- **BHARATH M**
- **SHWETA GAUR**

M.Tech — Information Retrieval Assignment

---

# ✅ Conclusion

This project successfully demonstrates the implementation of several foundational and advanced Information Retrieval concepts through a modular, interactive, and experimentally driven application.

The system combines:
- preprocessing techniques
- indexing structures
- retrieval optimization
- phrase querying
- tolerant retrieval
- analytics visualization

to provide a comprehensive educational Information Retrieval platform.
