##############################
### INITIALIZATION SECTION ###
##############################
import os
import sys
import time
from datetime import datetime
import psutil
import polars as pl
import spacy
from nltk.tokenize import sent_tokenize, word_tokenize
from xml.etree import ElementTree as ET
from tqdm import tqdm
import nltk
from halliday_dict import HALLIDAY_CONJUNCTIONS

# Initialize NLP
nltk.download("punkt", quiet=True)
try:
    nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])
    if "sentencizer" not in nlp.pipe_names:
        nlp.add_pipe("sentencizer")
except OSError:
    print("❌ Error: spaCy model 'en_core_web_sm' not found")
    sys.exit(1)


# Flatten conjunction dictionary
def flatten_conj_dict(d, path=None, result=None):
    if result is None:
        result = {}
    if path is None:
        path = []
    if isinstance(d, dict):
        for k, v in d.items():
            flatten_conj_dict(v, path + [k], result)
    elif isinstance(d, list):
        for conj in d:
            normalized_conj = conj.lower().strip()
            if normalized_conj:
                result[normalized_conj] = " → ".join(path)
    return result


flat_conj_dict = flatten_conj_dict(HALLIDAY_CONJUNCTIONS)


##############################
### CORE ANALYSIS FUNCTIONS ###
##############################
def process_file(file_path, flat_conj_dict):
    """Process single EFCAM file for inter-sentence conjunctions"""
    try:
        # Parse XML
        tree = ET.parse(file_path)
        root = tree.getroot()

        # Extract metadata from direct XML tags
        band = root.findtext("band", default="")
        prompt = root.findtext("prompt", default="")

        metadata = {
            "file_name": os.path.basename(file_path),
            "text_id": root.attrib.get("id", ""),  # 'essay' tag with id
            "band": band.strip(),
            "prompt": prompt.strip(),
        }
        # Also extract metadata from comments (below </text>)
        with open(file_path, "r", encoding="utf-8") as f:
            xml_text = f.read()

        def extract_comment_metadata(xml_string):
            meta = {}
            for line in xml_string.splitlines():
                if line.strip().startswith("<!--") and ":" in line:
                    try:
                        key, val = line.strip().strip("<!-->").split(":", 1)
                        meta[key.strip()] = val.strip()
                    except:
                        continue
            return meta

        comment_meta = extract_comment_metadata(xml_text)
        metadata.update(comment_meta)

        # Extract text
        paragraphs = []
        for elem in root.iter():
            if elem.tag.lower() in ["text", "writing", "p", "paragraph"]:
                text = " ".join(elem.itertext()).strip()
                if text:
                    paragraphs.append(text)

        if not paragraphs:
            paragraphs = [" ".join(root.itertext()).strip()]

        # Process text
        matches = []
        total_sentences = 0
        total_words = 0

        for para in paragraphs:
            sentences = [s.strip() for s in sent_tokenize(para) if s.strip()]
            total_sentences += len(sentences)
            total_words += len(word_tokenize(para))

            # Detect inter-sentence conjunctions
            for i in range(1, len(sentences)):
                doc = nlp(sentences[i])
                if len(doc) > 0:
                    first_token = doc[0]
                    conj = first_token.text.lower().strip()
                    if conj in flat_conj_dict and first_token.pos_ in [
                        "CCONJ",
                        "SCONJ",
                    ]:
                        matches.append(
                            {
                                **metadata,
                                "conjunction": conj,
                                "category": flat_conj_dict[conj],
                                "position": "sentence-initial",
                                "current_sentence": sentences[i],
                                "previous_sentence": sentences[i - 1],
                                "sentence_index": i,
                            }
                        )

        return matches, total_sentences, total_words, metadata

    except Exception as e:
        print(f"❌ Error processing {file_path}: {str(e)[:200]}")
        dummy_metadata = {
            "file_name": os.path.basename(file_path),
            "text_id": "",
            "band": "",
            "prompt": "",
        }
        return [], 0, 0, dummy_metadata


def batch_process_inter_conjunctions(folder_path, flat_conj_dict):
    """Process files in batches with progress tracking"""
    all_files = [f for f in os.listdir(folder_path) if f.endswith(".xml")]
    total_files = len(all_files)
    all_matches = []
    summary_rows = []
    global_stats = {
        "total_files": 0,
        "total_sentences": 0,
        "total_words": 0,
        "total_conjunctions": 0,
    }

    print(f"🔍 Found {total_files} XML files to process")

    # Process in batches
    for batch_start in range(0, total_files, BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, total_files)
        batch_files = all_files[batch_start:batch_end]

        print(
            f"\n🔄 Processing batch {batch_start//BATCH_SIZE + 1}/{(total_files-1)//BATCH_SIZE + 1}"
        )
        print(f"Files {batch_start + 1}-{batch_end} of {total_files}")

        batch_matches = []
        batch_summaries = []

        for file in tqdm(batch_files, desc="Processing files"):
            file_path = os.path.join(folder_path, file)
            matches, sent_count, word_count = process_file(file_path, flat_conj_dict)

            # Update global stats
            global_stats["total_files"] += 1
            global_stats["total_sentences"] += sent_count
            global_stats["total_words"] += word_count
            global_stats["total_conjunctions"] += len(matches)

            # Prepare summary
            if matches:
                all_matches.extend(matches)
                cat_counts = {}
                for m in matches:
                    cat = m["category"] + "_Inter"
                    cat_counts[cat] = cat_counts.get(cat, 0) + 1

                summary = {
                    "file_name": file,
                    "text_id": matches[0].get("text_id", ""),
                    "band": matches[0].get("band", ""),
                    "prompt": matches[0].get("prompt", ""),
                    "total_sentences": sent_count,
                    "total_words": word_count,
                    "inter_conj_total": len(matches),
                }
                batch_summaries.append(summary)

        print(f"✅ Batch completed - found {len(batch_matches)} conjunctions")

    # Prepare final outputs
    detailed_df = pl.DataFrame(all_matches) if all_matches else pl.DataFrame()
    summary_df = pl.DataFrame(batch_summaries) if batch_summaries else pl.DataFrame()

    # Calculate global summary
    global_summary = pl.DataFrame(
        [
            {
                **global_stats,
                "conjunctions_per_sentence": (
                    global_stats["total_conjunctions"] / global_stats["total_sentences"]
                    if global_stats["total_sentences"]
                    else 0
                ),
            }
        ]
    )

    return detailed_df, summary_df, global_summary


def export_results(detailed_df, summary_df, global_summary):
    """Save results to CSV files"""
    try:
        os.makedirs("results", exist_ok=True)
        detailed_df.write_csv("results/inter_sentence_detailed.csv")
        summary_df.write_csv("results/inter_sentence_summary.csv")
        global_summary.write_csv("results/inter_sentence_global.csv")
        return True
    except Exception as e:
        print(f"❌ Export failed: {e}")
        return False


##############################
### MAIN EXECUTION BLOCK ###
##############################
if __name__ == "__main__":
    folder_path = "/Users/joselema/LocalProjects/LocalProjects/ConjuntionArticle/Code_2_Inter_Ielts/essays_xml"
    BATCH_SIZE = 1000

    if not os.path.exists(folder_path):
        print(f"❌ Error: Folder not found at {folder_path}")
        sys.exit(1)

    print("🚀 Starting inter-sentence conjunction analysis...")
    start_time = time.time()

    # ✅ Ensure output folders exist
    os.makedirs("results", exist_ok=True)
    os.makedirs("output_batches", exist_ok=True)

    all_files = [f for f in os.listdir(folder_path) if f.endswith(".xml")]
    total_files = len(all_files)
    all_matches = []
    all_summaries = []
    global_stats = {
        "total_files": 0,
        "total_sentences": 0,
        "total_words": 0,
        "total_conjunctions": 0,
    }

    for batch_start in range(0, total_files, BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, total_files)
        batch_files = all_files[batch_start:batch_end]
        batch_id = batch_start // BATCH_SIZE + 1

        print(
            f"\n🔄 Batch {batch_id}: Files {batch_start + 1}-{batch_end} of {total_files}"
        )

        batch_matches = []
        batch_summaries = []

        for file in tqdm(batch_files, desc=f"Processing Batch {batch_id}"):
            file_path = os.path.join(folder_path, file)
            matches, sent_count, word_count, metadata = process_file(
                file_path, flat_conj_dict
            )

            global_stats["total_files"] += 1
            global_stats["total_sentences"] += sent_count
            global_stats["total_words"] += word_count
            global_stats["total_conjunctions"] += len(matches)

            summary = {
                "file_name": file,
                "text_id": metadata.get("text_id", ""),
                "band": metadata.get("band", ""),
                "prompt": metadata.get("prompt", ""),
                "total_sentences": sent_count,
                "total_words": word_count,
                "inter_conj_total": len(matches),
            }
            if matches:
                cat_counts = {}
                for m in matches:
                    cat = m["category"] + "_Inter"
                    cat_counts[cat] = cat_counts.get(cat, 0) + 1
                summary.update({f"cat_{k}": v for k, v in cat_counts.items()})

            all_summaries.append(summary)  # 👈 always append, even if no matches
            batch_summaries.append(summary)

        # ✅ Save each batch inside "output_batches"
        if batch_matches:
            pl.DataFrame(batch_matches).write_csv(
                f"output_batches/batch_{batch_id:03d}_detailed.csv"
            )
        if batch_summaries:
            pl.DataFrame(batch_summaries).write_csv(
                f"output_batches/batch_{batch_id:03d}_summary.csv"
            )

        print(f"✅ Batch {batch_id} saved with {len(batch_matches)} conjunctions")

    # ✅ Final combined exports to "results/"
    detailed_df = pl.DataFrame(all_matches) if all_matches else pl.DataFrame()
    summary_df = pl.DataFrame(all_summaries) if all_summaries else pl.DataFrame()
    global_summary = pl.DataFrame(
        [
            {
                **global_stats,
                "conjunctions_per_sentence": (
                    global_stats["total_conjunctions"] / global_stats["total_sentences"]
                    if global_stats["total_sentences"]
                    else 0
                ),
            }
        ]
    )

    if export_results(detailed_df, summary_df, global_summary):
        print("\n📊 Analysis Summary:")
        print(f"- Processed {global_summary['total_files'][0]} files")
        print(f"- Analyzed {global_summary['total_sentences'][0]} sentences")
        print(
            f"- Found {global_summary['total_conjunctions'][0]} inter-sentence conjunctions"
        )
        print(
            f"- Average {global_summary['conjunctions_per_sentence'][0]:.4f} conjunctions per sentence"
        )

    print(f"\n⏱️ Total processing time: {time.time() - start_time:.2f} seconds")
