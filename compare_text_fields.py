import streamlit as st
import json
import pandas as pd
from pathlib import Path

st.set_page_config(layout="wide")



# -------------------------
# Paths (CHANGE THESE)
# -------------------------
JSONL_PATH = "PlacesHindi100k_train-processed-max7s_dev+english.json"
VOCAB_PATH = "dev_vocab-max7s.txt"
CSV_PATH = "streamlit_caption_precision.csv"  # replace with your CSV file path
CSV_PATH_RECALL = "dataset_multiling/final/streamlit_caption_recall.csv"

# ---------------------------------------------------
# LOAD DATA
# ---------------------------------------------------
@st.cache_data
def load_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data

@st.cache_data
def load_vocab(path):
    vocab = {}
    df = pd.read_csv(path, sep="\t", header=None)
    for _, row in df.iterrows():
        eng_word = str(row[0]).strip().lower()
        romanized_synonyms = str(row[2]).split(",")
        romanized_synonyms = [w.strip().lower() for w in romanized_synonyms]
        vocab[eng_word] = romanized_synonyms
    return vocab

data = load_jsonl(JSONL_PATH)
vocab = load_vocab(VOCAB_PATH)

# ---------------------------------------------------
# NORMALIZATION HELPERS
# ---------------------------------------------------
def normalize_to_list(value):
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        return value.strip().split()
    return []

def highlight_words(word_list, target_word, synonyms=None):
    word_list = normalize_to_list(word_list)
    highlighted = []
    for w in word_list:
        lw = w.lower()
        if lw == target_word.lower() or (synonyms and lw in synonyms):
            highlighted.append(f"<span style='color:green; font-weight:bold'>{w}</span>")
        else:
            highlighted.append(w)
    return " | ".join(highlighted)

def check_match(word_list, target_word, synonyms=None):
    word_list = normalize_to_list(word_list)
    for w in word_list:
        lw = w.lower()
        if lw == target_word.lower() or (synonyms and lw in synonyms):
            return True
    return False

# ---------------------------------------------------
# SIDEBAR
# ---------------------------------------------------
st.sidebar.title("Search Panel")

selected_word = st.sidebar.selectbox(
    "Choose vocabulary word",
    sorted(vocab.keys())
)

include_fields = st.sidebar.multiselect(
    "Include in fields",
    [
        "english_text_lemma",
        "tag2text+blip2+coco",
        "translation_lemma",
        "romanized_text_lemma",
    ],
    default=["english_text_lemma"]
)

exclude_fields = st.sidebar.multiselect(
    "Exclude from fields",
    [
        "english_text_lemma",
        "tag2text+blip2+coco",
        "translation_lemma",
        "romanized_text_lemma",
    ]
)

synonyms = vocab[selected_word]

# ---------------------------------------------------
# FILTERING
# ---------------------------------------------------
def field_include(entry, field):
    """Check if the word/synonym is in the field"""
    values = normalize_to_list(entry.get(field))
    if field == "romanized_text_lemma":
        return any(s in values for s in synonyms)
    else:
        return selected_word in values

def field_exclude(entry, field):
    """Check if the word/synonym is in the field (for exclusion)"""
    values = normalize_to_list(entry.get(field))
    if field == "romanized_text_lemma":
        return any(s in values for s in synonyms)
    else:
        return selected_word in values

filtered_entries = []
for e in data:
    # Include filter: must satisfy all include fields
    if include_fields and not all(field_include(e, f) for f in include_fields):
        continue
    # Exclude filter: must NOT be present in any exclude field
    if exclude_fields and any(field_exclude(e, f) for f in exclude_fields):
        continue
    filtered_entries.append(e)

# Count samples containing synonyms in romanized_text_lemma
samples_in_romanized = sum(
    1 for e in filtered_entries
    if check_match(e.get("romanized_text_lemma"), selected_word, synonyms)
)

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Matching samples for filter criteria:** {len(filtered_entries)}")
st.sidebar.markdown(f"**Actual samples in Hindi transcripts:** {samples_in_romanized}")

# -------------------------
# Pagination
# -------------------------
SAMPLES_PER_PAGE = 10
total_pages = (len(filtered_entries) - 1) // SAMPLES_PER_PAGE + 1
if 'page' not in st.session_state:
    st.session_state.page = 1

# Prev / Next buttons
col1, col2 = st.sidebar.columns([1,1])
if col1.button("Prev") and st.session_state.page > 1:
    st.session_state.page -= 1
if col2.button("Next") and st.session_state.page < total_pages:
    st.session_state.page += 1

page = st.session_state.page
start_idx = (page - 1) * SAMPLES_PER_PAGE
end_idx = start_idx + SAMPLES_PER_PAGE
paginated_entries = filtered_entries[start_idx:end_idx]

st.sidebar.markdown(f"Page {page} of {total_pages}")

# -------------------------
# Sidebar: Load CSV & Display Table
# -------------------------

st.sidebar.markdown("---")
st.sidebar.markdown("### Word Performance Table")


@st.cache_data
def load_metrics_csv(path):
    with open(path, "r") as f:
        lines = [line.strip().split(",") for line in f.readlines()]

    # Ensure we have at least 4 rows (header + 3 data rows)
    if len(lines) < 4:
        raise ValueError("CSV file must have at least 4 rows (header + 3 data rows)")

    # Header starts from the 3rd column
    header = lines[0][2:]

    # Function to convert values to float safely, pad/truncate to header length
    def parse_row(row):
        vals = row[2:]
        # convert to float, fill missing with NaN
        vals_float = []
        for i in range(len(header)):
            if i < len(vals):
                try:
                    vals_float.append(float(vals[i]))
                except:
                    vals_float.append(float("nan"))
            else:
                vals_float.append(float("nan"))
        return vals_float

    tags_en = parse_row(lines[1])
    tags_hi = parse_row(lines[2])
    en_hi   = parse_row(lines[3])

    df = pd.DataFrame({
        "word": header,
        "tags-en": tags_en,
        "tags-hi": tags_hi,
        "en-hi": en_hi
    })

    df["diff_en-hi_tags-en"] = df["en-hi"] - df["tags-en"]
    df[["tags-en","tags-hi","en-hi","diff_en-hi_tags-en"]] = df[["tags-en","tags-hi","en-hi","diff_en-hi_tags-en"]].round(3)
    return df

st.sidebar.markdown("---")
st.sidebar.markdown("### Word Precision Table")

# Usage
metrics_df = load_metrics_csv(CSV_PATH)

st.sidebar.dataframe(metrics_df)

st.sidebar.markdown("### Word Recall Table")
# Usage
metrics_df_recall = load_metrics_csv(CSV_PATH_RECALL)

st.sidebar.dataframe(metrics_df_recall)


# ---------------------------------------------------
# MAIN DISPLAY
# ---------------------------------------------------
st.title("Dataset Viewer")

for idx, selected_entry in enumerate(paginated_entries):
    st.markdown(f"## Sample {start_idx + idx + 1}")

    col1, col2 = st.columns([2,2])
    with col1:
        st.markdown("Should have been image")
        #st.image(selected_entry["image_filepath"], width=400)
    with col2:
        #st.audio(selected_entry["audio_filepath"])
        st.markdown("Should have been audio")

        st.markdown("### Hindi Transcription")
        st.write(selected_entry.get("romanized_text"))

        st.write("---")

        st.markdown("### Hindi->English Translation")
        st.write(selected_entry.get("translation"))

        st.write("---")
        
        st.markdown(f"### Hindi Synonyms for word {selected_word}:")
        st.markdown(
            f"<div style='padding:10px; border-radius:12px; font-size:16px;'>{' | '.join(synonyms)}</div>",
            unsafe_allow_html=True
        )

    st.write('---')

    # 4-column comparison
    header_names = [
        
        "English Transcript",
        "Translation Lemma",
        "Hindi Lemma",
        "Image tags",
    ]
    fields_to_display= [
        
        "english_text_lemma",
        "translation_lemma",
        "romanized_text_lemma",
        "tag2text+blip2+coco",
    ]

    cols = st.columns(4)
    for col, header in zip(cols, header_names):
        col.markdown(f"### {header}")

    cols = st.columns(4)
    for col, field in zip(cols, fields_to_display):
        values = selected_entry.get(field)
        if field == "romanized_text_lemma":
            is_match = check_match(values, selected_word, synonyms)
            display_text = highlight_words(values, selected_word, synonyms)
        else:
            is_match = check_match(values, selected_word)
            display_text = highlight_words(values, selected_word)

        icon = "✅" if is_match else "❌"
        color = "green" if is_match else "red"

        col.markdown(f"<div style='color:{color}; font-size:22px; font-weight:bold'>{icon}</div>", unsafe_allow_html=True)
        col.markdown(f"<div style='padding:12px; border-radius:12px; min-height:120px;'>{display_text}</div>", unsafe_allow_html=True)
    
    st.write("---")
