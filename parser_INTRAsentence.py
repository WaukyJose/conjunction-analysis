##############################
### INITIALIZATION SECTION ###
##############################
import os
import time
from datetime import datetime
import psutil  # For memory monitoring
from tqdm import tqdm  # Progress bars
import polars as pl  # Faster alternative to pandas
import spacy
from nltk.tokenize import sent_tokenize, word_tokenize
from xml.etree import ElementTree as ET
from joblib import Parallel, delayed
from joblib import parallel_backend
from itertools import islice
import nltk
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
import logging
import re
from halliday_intrasent_dic import INTRA_CLAUSE_CONJUNCTIONS

# Configuration constants (adjust based on your system)
CONFIG = {
    "input_dir": "/Users/joselema/LocalProjects/LocalProjects/ConjuntionArticle/Code_1_Intra_Ielts/essays_xml",
    "output_dir": "output_batches",
    "batch_size": 1000,
    "n_jobs": -1,  # -1 means use all available cores
    "max_ram_usage": 0.85,
    "spacy_model": "en_core_web_sm",
    "report_interval": 50,
    "log_file": "processing_log.txt",
}

# Set up logging
logging.basicConfig(
    filename=CONFIG["log_file"],
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# Initialize NLP components
nltk.download("punkt", quiet=True)
try:
    nlp = spacy.load(CONFIG["spacy_model"], disable=["ner", "lemmatizer"])
except OSError:
    logging.error(f"Spacy model {CONFIG['spacy_model']} not found. Installing...")
    import spacy.cli

    spacy.cli.download(CONFIG["spacy_model"])
    nlp = spacy.load(CONFIG["spacy_model"], disable=["ner", "lemmatizer"])

os.makedirs(CONFIG["output_dir"], exist_ok=True)

# Linguistic processing constants
PARA_DEPS = {"cc", "conj"}
HYPO_DEPS = {"mark", "advcl", "relcl"}
PUNCTUATION = {",", ";", ":", "-", "—", "–"}

# Metadata structure
METADATA_FIELDS = ["band", "prompt"]


def flatten_intra_conj_dict(
    d: Dict, path: Optional[List[str]] = None, result: Optional[Dict] = None
) -> Dict:
    """Flatten the nested conjunction dictionary into a flat structure."""
    if result is None:
        result = {}
    if path is None:
        path = []
    if isinstance(d, dict):
        for k, v in d.items():
            flatten_intra_conj_dict(v, path + [k], result)
    elif isinstance(d, list):
        for conj in d:
            normalized_conj = conj.lower().strip()
            if normalized_conj:
                result[normalized_conj] = " → ".join(path)
    return result


flat_intra_conj_dict = flatten_intra_conj_dict(INTRA_CLAUSE_CONJUNCTIONS)


def log_message(msg: str) -> None:
    """Log messages with timestamp and memory usage."""
    mem = psutil.virtual_memory()
    log_str = (
        f"[{datetime.now().strftime('%H:%M:%S')}] "
        f"[RAM: {mem.used/1024**3:.1f}/{mem.total/1024**3:.1f}GB] "
        f"{msg}"
    )
    print(log_str)
    logging.info(log_str)


def extract_clauses_and_taxis(sentence: str) -> Tuple[List[str], List[str]]:
    """Extract clauses and their taxis relations from a sentence."""
    if not sentence or not isinstance(sentence, str):
        return [], []

    doc = nlp(sentence)
    clauses, taxis_labels = [], []
    current_clause = []

    for token in doc:
        current_clause.append(token.text)
        if token.dep_ in PARA_DEPS.union(HYPO_DEPS) or token.text in PUNCTUATION:
            clause_text = " ".join(current_clause).strip()
            if clause_text:
                clauses.append(clause_text)
                taxis_labels.append(token.dep_)
            current_clause = []

    if current_clause:
        final_clause = " ".join(current_clause).strip()
        if final_clause:
            clauses.append(final_clause)

    return clauses, taxis_labels


def detect_intra_conjunctions(text: str) -> List[Dict[str, Any]]:
    """Detect intra-sentential conjunctions in text."""
    if not text or not isinstance(text, str):
        return []

    results = []
    sentences = sent_tokenize(text)

    for sent in sentences:
        if not sent.strip():
            continue

        doc = nlp(sent.lower())
        words = [token.text for token in doc]
        pos_tags = [token.pos_ for token in doc]

        for i, token in enumerate(doc):
            conj = token.text.lower().strip()
            normalized_conj = conj.strip(".,!?;:\"'()[]{}")

            if normalized_conj in flat_intra_conj_dict:
                if not normalized_conj or len(normalized_conj) < 2:
                    continue

                position = (
                    "clause-initial"
                    if i <= 2
                    else "clause-medial" if i < len(words) - 2 else "clause-final"
                )

                left_context = pos_tags[max(0, i - 3) : i]
                right_context = pos_tags[i + 1 : min(len(pos_tags), i + 4)]

                results.append(
                    {
                        "sentence": sent,
                        "conjunction": normalized_conj,
                        "category": flat_intra_conj_dict[normalized_conj],
                        "position": position,
                        "colligation_left": left_context,
                        "colligation_right": right_context,
                    }
                )

    return results


def extract_metadata(xml_content: str) -> Dict[str, str]:
    """Extract metadata like band and prompt from structured XML."""
    metadata = {}

    try:
        tree = ET.fromstring(xml_content)
        band = tree.findtext("band")
        prompt = tree.findtext("prompt")

        metadata["band"] = band if band else ""
        metadata["prompt"] = prompt if prompt else ""
    except ET.ParseError:
        pass  # Fall back to empty metadata if broken XML

    return metadata


def process_text_content(xml_content: str) -> Dict[str, Any]:
    """Process the text content and extract linguistic features."""
    try:
        tree = ET.fromstring(xml_content)
    except ET.ParseError:
        tree = ET.fromstring(f"<root>{xml_content}</root>")

    paragraphs = []
    for elem in tree.iter():
        if elem.tag.lower() in ["text", "writing", "p", "paragraph"]:
            text = " ".join(elem.itertext()).strip()
            if text:
                paragraphs.append(text)

    if not paragraphs:
        paragraphs = [" ".join(tree.itertext()).strip()]

    paragraphs = [p for p in paragraphs if p.strip()]
    full_text = " ".join(paragraphs)
    total_words = len(word_tokenize(full_text))

    # Initialize counters
    counts = {
        "total_clauses": 0,
        "total_clause_words": 0,
        "parataxis_count": 0,
        "hypotaxis_count": 0,
        "matches": [],
    }

    for para in paragraphs:
        for sent in sent_tokenize(para):
            clauses, taxis = extract_clauses_and_taxis(sent)
            counts["total_clauses"] += len(clauses)
            counts["total_clause_words"] += sum(len(c.split()) for c in clauses)
            counts["parataxis_count"] += sum(1 for t in taxis if t in PARA_DEPS)
            counts["hypotaxis_count"] += sum(1 for t in taxis if t in HYPO_DEPS)
            counts["matches"].extend(detect_intra_conjunctions(sent))

    return {
        "paragraphs": paragraphs,
        "full_text": full_text,
        "total_words": total_words,
        **counts,
    }


def process_file(file: str, folder_path: str) -> Optional[Dict[str, Any]]:
    """Process a single XML file and extract all features."""
    if not file.endswith(".xml"):
        return None

    file_path = os.path.join(folder_path, file)
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            xml_content = f.read()

        # Extract metadata
        metadata = extract_metadata(xml_content)

        # Process text content
        text_data = process_text_content(xml_content)

        # Calculate metrics
        total_clauses = text_data["total_clauses"]
        total_words = text_data["total_words"]
        matches = text_data["matches"]

        results = {
            "file_name": file,
            "writing_id": int(re.sub(r"[^\d]", "", file)),
            "total_paragraphs": len(text_data["paragraphs"]),
            "total_sentences": sum(
                len(sent_tokenize(p)) for p in text_data["paragraphs"]
            ),
            "total_words": total_words,
            "total_clauses": total_clauses,
            "mean_words_per_clause": (
                round(text_data["total_clause_words"] / total_clauses, 2)
                if total_clauses
                else 0
            ),
            "intra_conj_total": len(matches),
            "intra_conj_per_clause": (
                round(len(matches) / total_clauses, 4) if total_clauses else 0
            ),
            "intra_conj_per_100_words": (
                round(len(matches) / total_words * 100, 4) if total_words else 0
            ),
            "parataxis_count": text_data["parataxis_count"],
            "hypotaxis_count": text_data["hypotaxis_count"],
            "parataxis_per_clause": (
                round(text_data["parataxis_count"] / total_clauses, 4)
                if total_clauses
                else 0
            ),
            "hypotaxis_per_clause": (
                round(text_data["hypotaxis_count"] / total_clauses, 4)
                if total_clauses
                else 0
            ),
            **metadata,  # Include all metadata fields
        }

        # Add conjunction category counts
        for category_path in flat_intra_conj_dict.values():
            cat_key = category_path + "_Intra"
            results[cat_key] = 0

        for m in matches:
            cat = m["category"] + "_Intra"
            results[cat] = results.get(cat, 0) + 1

        # Add per-100-words metrics for each category
        for key in list(results.keys()):
            if key.endswith("_Intra"):
                results[f"{key}_per_100w"] = (
                    round(results[key] / total_words * 100, 4) if total_words else 0
                )

        return results

    except Exception as e:
        error_msg = f"Failed to process {file}: {str(e)}"
        log_message(f"❌ {error_msg}")
        logging.error(error_msg, exc_info=True)
        return {"file_name": file, "error": str(e)}


def process_batch(batch: List[str], input_dir: str, batch_num: int) -> pl.DataFrame:
    """Process a batch of files and return a Polars DataFrame."""
    with parallel_backend("threading", n_jobs=CONFIG["n_jobs"]):
        results = Parallel()(delayed(process_file)(f, input_dir) for f in batch)

    valid_results = [r for r in results if isinstance(r, dict) and "error" not in r]
    return pl.DataFrame(valid_results)


def main():
    """Main processing pipeline."""
    start_time = time.time()
    log_message(f"Starting processing in {CONFIG['input_dir']}")

    all_files = sorted(f for f in os.listdir(CONFIG["input_dir"]) if f.endswith(".xml"))
    total_batches = (len(all_files) + CONFIG["batch_size"] - 1) // CONFIG["batch_size"]

    log_message(f"Found {len(all_files)} files to process in {total_batches} batches")

    for i in range(total_batches):
        batch_start = time.time()
        batch = list(
            islice(all_files, i * CONFIG["batch_size"], (i + 1) * CONFIG["batch_size"])
        )

        log_message(f"Processing batch {i+1}/{total_batches} ({len(batch)} files)")

        df = process_batch(batch, CONFIG["input_dir"], i + 1)
        output_path = f"{CONFIG['output_dir']}/batch_{i+1}.csv"
        df.write_csv(output_path)

        batch_time = time.time() - batch_start
        log_message(
            f"Completed batch {i+1} ({batch_time:.1f}s, {len(batch)/batch_time:.1f} files/sec)"
        )

    total_time = time.time() - start_time
    log_message(
        f"All batches processed in {total_time/3600:.2f} hours "
        f"({len(all_files)/total_time:.1f} files/sec overall)"
    )


if __name__ == "__main__":
    # Warm up the NLP pipeline
    _ = nlp("This is a warm-up sentence to initialize the pipeline.")

    # Run main processing
    try:
        main()
    except Exception as e:
        logging.critical("Fatal error in main execution", exc_info=True)
        raise
