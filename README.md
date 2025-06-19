[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15570827.svg)](https://doi.org/10.5281/zenodo.15570827)
# Conjunction Analysis Framework

This repository provides the custom resources and parsing tools developed for the study:

**“Dual-Level Modeling of Conjunction in L1 and L2 Writing: A Corpus-Based Computational Approach”**  
*José Lema-Alarcón (2025)*

## 🔍 Overview

The project implements a novel extension of Halliday’s Systemic Functional Grammar by incorporating a **positional dimension** into the classification of conjunctions. The framework distinguishes conjunctions by:
- **Function**: elaboration, extension, enhancement
- **Position**: intra-sentential, inter-sentential, and inter-paragraph

These scripts and dictionaries allow researchers to parse text segments and extract conjunction indices aligned to this dual-layered model.

---

## 📁 Repository Structure

```
.
├── dicts/
│   ├── dict_intrasent.py          # Clause-level dictionary
│   ├── dict_intersent.py          # Sentence-level dictionary
│   └── dict_interpara.py          # Paragraph-level dictionary
├── parsers/
│   ├── parser_intrasent.py        # Applies intra-sentential tagging
│   ├── parser_intersent.py        # Applies inter-sentential tagging
│   └── parser_interpara.ipynb     # Notebook for paragraph-initial tagging
├── sample_outputs/
│   ├── sample_output_intrasent.csv
│   ├── sample_output_intersent.csv
│   └── sample_output_interpara.csv
├── requirements.txt
└── LICENSE
```

---

## 🚀 How to Use

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run a parser (e.g., intra-sentential tagging)
```bash
python parsers/parser_intrasent.py --input your_text_file.txt --output output.csv
```

*(Modify the script if needed. Paths and flags can be customized.)*

---

## 🧪 Sample Output Format

Each CSV in `sample_outputs/` shows:
- Original sentence
- Detected conjunction(s)
- Function tag (e.g., enhancement_causal)
- Position (e.g., inter-sentential)

---

## 📄 Citation

If you use or adapt this code or dictionaries, please cite the paper:

> Lema-Alarcón, J. (2025). Dual-Level Modeling of Conjunction in L1 and L2 Writing: A Corpus-Based Computational Approach. 

---

## 📜 License

This project is shared under the MIT License.  
For academic and non-commercial use only. Contact the author for other uses.
