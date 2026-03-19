import os
import re
import json
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm

tqdm.pandas()

# 0. Path configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)
    print(f"✅ Data folder created at: {DATA_DIR}")
else:
    print(f"✅ Data folder confirmed at: {DATA_DIR}")


# 1. Load data and save original
print("\nStep 1: Loading dataset from Hugging Face...")
raw_dataset = load_dataset("rntc/mimic-icd-reformulations-medgemma-27b-text-it-2", split="train")
df_raw = pd.DataFrame(raw_dataset)
print("✅ Raw data download completed")

raw_csv_path = os.path.join(DATA_DIR, f"raw_data_{len(df_raw)}.csv")
df_raw.to_csv(raw_csv_path, index=False, encoding="utf-8-sig")
print(f"✅ Original raw data saved: {raw_csv_path}")


# 2. Column selection
print("\nStep 2: Selecting core columns (reformulation, icd_code)...")
df = df_raw[["reformulation", "icd_code"]].copy()
print("✅ Selection completed")


# 3. Create filtered_text
print("\nStep 3: Extracting reason/history and course/treatment sections...")

MAX_COURSE_CHARS = 1000

REASON_LABELS = [
    "Chief Complaint",
    "Chief Complaint/Reason for Visit",
    "Reason for Visit",
    "Reason for Admission",
    "Reasoning for Admission",
]

HISTORY_LABELS = [
    "History of Present Illness",
    "History",
]

COURSE_LABELS = [
    "Hospital Course",
    "Brief Hospital Course",
    "Hospital Course Summary",
    "Post-Operative Course",
    "ICU Course",
    "Treatment/Hospital Course",
    "Treatment",
]


def normalize_label(label):
    return re.sub(r"\s+", " ", str(label)).strip().lower().rstrip(":")


LABEL_MAP = {
    normalize_label(label): label
    for label in REASON_LABELS + HISTORY_LABELS + COURSE_LABELS
}


def clean_text(text):
    if not isinstance(text, str):
        return ""

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def parse_sections(text):
    # Parse markdown-style note sections line by line.
    sections = []
    current_label = None
    current_lines = []

    def flush_section():
        nonlocal current_label, current_lines
        if current_label is not None:
            content = "\n".join(current_lines).strip()
            sections.append((current_label, content))
        current_label = None
        current_lines = []

    for raw_line in clean_text(text).split("\n"):
        line = raw_line.strip()

        if not line:
            if current_label is not None:
                current_lines.append("")
            continue

        label = None
        content = ""

        # Match markdown headings such as ## Hospital Course Note
        m = re.match(r"^#+\s*\**([^:#*\n][^:\n]{0,100}?)\**\s*:?\s*$", line)
        if m:
            label = m.group(1).strip()
        else:
            # Match bold headings such as **Chief Complaint:** Chest pain
            m = re.match(r"^\*\*([^*\n]{1,120}?)\*\*\s*:?\s*(.*)$", line)
            if m:
                label = m.group(1).strip()
                content = m.group(2).strip()
            else:
                # Match plain headings such as Chief Complaint: Chest pain
                m = re.match(r"^([A-Z][A-Za-z0-9 /&(),\-]{1,100})\s*:\s*(.*)$", line)
                if m and len(m.group(1).split()) <= 8:
                    label = m.group(1).strip()
                    content = m.group(2).strip()

        if label is not None:
            flush_section()
            current_label = label.rstrip(":").strip()
            if content:
                current_lines.append(content)
        else:
            if current_label is None:
                current_label = "__preface__"
            current_lines.append(line)

    flush_section()
    return sections


def get_first_section(section_dict, labels, max_chars=None):
    # Return the first matched section from a priority list.
    for label in labels:
        key = normalize_label(label)
        if key in section_dict:
            value = clean_text(section_dict[key])
            if value:
                if max_chars is not None:
                    value = value[:max_chars].strip()
                return value
    return ""


def extract_clinical_info(text):
    if not isinstance(text, str):
        return ""

    parsed_sections = parse_sections(text)

    # Keep the first occurrence for each section label.
    section_dict = {}
    for label, content in parsed_sections:
        key = normalize_label(label)
        if key not in section_dict and content:
            section_dict[key] = content

    reason_text = get_first_section(section_dict, REASON_LABELS)
    history_text = get_first_section(section_dict, HISTORY_LABELS)
    course_text = get_first_section(section_dict, COURSE_LABELS, max_chars=MAX_COURSE_CHARS)

    parts = []
    if reason_text:
        parts.append(reason_text)
    if history_text and history_text != reason_text:
        parts.append(history_text)
    if course_text:
        parts.append(course_text)

    return "\n\n".join(parts).strip()


# Extract note text and remove short samples.
df["filtered_text"] = df["reformulation"].progress_apply(extract_clinical_info)
df = df[df["filtered_text"].str.len() > 50].copy()
print("✅ Extraction completed")


# 4. Create major_code (3-character ICD-10 Code)
print("\nStep 4: Categorizing by 3-character ICD-10 code (major_code)...")
df["major_code"] = df["icd_code"].progress_apply(
    lambda x: x[0][:3].upper() if x and len(x) > 0 else None
)
df = df.dropna(subset=["major_code"])
print("✅ Categorizing completed")


# 5. Save preprocessed data
preprocessed_csv_path = os.path.join(DATA_DIR, f"preprocessed_data_{len(df)}.csv")
df.to_csv(preprocessed_csv_path, index=False, encoding="utf-8-sig")
print(f"✅ Preprocessed full dataset saved: {preprocessed_csv_path}")


# 6. Filter top 30 categories and sample proportionally
print("\nStep 6: Filtering Top 30 categories and sampling proportionally to 10,000...")

TOP_K = 30
TOTAL_SAMPLES = 10000

# Select the top 30 classes by frequency.
top_labels = df["major_code"].value_counts().nlargest(TOP_K).index.tolist()
df_filtered = df[df["major_code"].isin(top_labels)]

# Compute class proportions within the Top 30.
proportions = df_filtered["major_code"].value_counts(normalize=True)
target_counts = (proportions * TOTAL_SAMPLES).astype(int)

# Sample each class while keeping the original imbalance pattern.
print("Applying proportional sampling...")
proportional_df = df_filtered.groupby("major_code", group_keys=False).progress_apply(
    lambda x: x.sample(n=min(len(x), target_counts[x.name]), random_state=359)
).reset_index(drop=True)

print(f"✅ Sampled {len(proportional_df)} records across {TOP_K} categories.")


# 7. Create label mapping and add label column
print("\nStep 7: Creating fixed label mapping and adding label column...")

sorted_top_labels = sorted(top_labels)
label2id = {label: idx for idx, label in enumerate(sorted_top_labels)}
id2label = {idx: label for label, idx in label2id.items()}

proportional_df["labels"] = proportional_df["major_code"].map(label2id)

assert proportional_df["labels"].isna().sum() == 0, "Some labels were not mapped."
proportional_df["labels"] = proportional_df["labels"].astype(int)

label_mapping_df = pd.DataFrame({
    "major_code": sorted_top_labels,
    "labels": [label2id[label] for label in sorted_top_labels],
})

label_mapping_csv_path = os.path.join(DATA_DIR, f"label_mapping_top{TOP_K}.csv")
label_mapping_df.to_csv(label_mapping_csv_path, index=False, encoding="utf-8-sig")

label2id_json_path = os.path.join(DATA_DIR, f"label2id_top{TOP_K}.json")
id2label_json_path = os.path.join(DATA_DIR, f"id2label_top{TOP_K}.json")

with open(label2id_json_path, "w", encoding="utf-8") as f:
    json.dump(label2id, f, indent=2, ensure_ascii=False)

with open(id2label_json_path, "w", encoding="utf-8") as f:
    json.dump(id2label, f, indent=2, ensure_ascii=False)

print(f"✅ Label mapping CSV saved: {label_mapping_csv_path}")
print(f"✅ label2id JSON saved: {label2id_json_path}")
print(f"✅ id2label JSON saved: {id2label_json_path}")


# 7. Save total sampled data
print("\nStep 8: Saving the total sampled dataset before splitting...")
total_csv_path = os.path.join(DATA_DIR, f"total_{TOP_K}_{len(proportional_df)}.csv")
proportional_df.to_csv(total_csv_path, index=False, encoding="utf-8-sig")
print(f"✅ Total sampled data saved: {total_csv_path}")


# 9. Split Train / Validation / Test (8:1:1)
print("\nStep 9: Splitting dataset into Train, Val, and Test (80:10:10)...")

train_df, temp_df = train_test_split(
    proportional_df,
    test_size=0.2,
    stratify=proportional_df["major_code"],
    random_state=359,
)
print("✅ Train data split completed")

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.5,
    stratify=temp_df["major_code"],
    random_state=359,
)
print("✅ Val/Test split completed")

assert train_df["labels"].isna().sum() == 0, "Train split contains unmapped labels."
assert val_df["labels"].isna().sum() == 0, "Validation split contains unmapped labels."
assert test_df["labels"].isna().sum() == 0, "Test split contains unmapped labels."

train_df["labels"] = train_df["labels"].astype(int)
val_df["labels"] = val_df["labels"].astype(int)
test_df["labels"] = test_df["labels"].astype(int)

train_path = os.path.join(DATA_DIR, f"train_{TOP_K}_{len(train_df)}.csv")
val_path = os.path.join(DATA_DIR, f"val_{TOP_K}_{len(val_df)}.csv")
test_path = os.path.join(DATA_DIR, f"test_{TOP_K}_{len(test_df)}.csv")

train_df.to_csv(train_path, index=False, encoding="utf-8-sig")
val_df.to_csv(val_path, index=False, encoding="utf-8-sig")
test_df.to_csv(test_path, index=False, encoding="utf-8-sig")
print(f"✅ Final splits saved: \n - {train_path} \n - {val_path} \n - {test_path}")


# Final summary
print("\n" + "=" * 50)
print("🚀 Project Data Preparation Complete!")
print(f"- Target Directory:\n   {DATA_DIR}")
print(f"- Selected Categories (Top {TOP_K}, frequency order):\n   {top_labels}")
print(f"- Label Mapping Order (alphanumeric):\n   {sorted_top_labels}")
print(f"- Total  : {len(proportional_df)}")
print(f"- Train  : {len(train_df)}")
print(f"- Val    : {len(val_df)}")
print(f"- Test   : {len(test_df)}")
print("=" * 50)
