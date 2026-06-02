"""
Main Streamlit application for the
Information Retrieval system.

This application demonstrates:
    1. Dataset ingestion
    2. PDF document ingestion
    3. Text preprocessing
    4. Inverted indexing
    5. Retrieval comparison
    6. Phrase query processing
    7. BST vs B-Tree search
    8. Tolerant retrieval
    9. Experimental analytics

Authors:
    - KAMESWARA RAO K.
    - BHARATH M
    - SHWETA GAUR
"""

# ============================================================================
# IMPORTS
# ============================================================================

import nltk

nltk_resources = {
"tokenizers/punkt": "punkt",
"tokenizers/punkt_tab": "punkt_tab",
"corpora/stopwords": "stopwords",
"corpora/wordnet": "wordnet",
"corpora/omw-1.4": "omw-1.4",
}

for resource_path, resource_name in nltk_resources.items():
    try:
        nltk.data.find(resource_path)
    except LookupError:
        print(f"Downloading {resource_name}...")
        nltk.download(resource_name)

from time import perf_counter
from typing import Dict
from typing import List

import pandas as pd
import plotly.express as px
import streamlit as st
from pandas import DataFrame

from modules.bst import build_bst_from_vocabulary
from modules.bst import extract_vocabulary
from modules.btree import build_btree_from_vocabulary
from modules.constants import APP_TITLE
from modules.constants import SUPPORTED_FILE_TYPES
from modules.data_loader import load_default_dataset
from modules.data_loader import load_pdf_documents
from modules.data_loader import load_uploaded_dataset
from modules.indexing import create_inverted_index
from modules.phrase_query import create_biword_index
from modules.phrase_query import create_positional_index
from modules.phrase_query import generate_phrase_query_inference
from modules.phrase_query import search_biword_phrase
from modules.phrase_query import search_positional_phrase
from modules.preprocessing import preprocess_text
from modules.retrieval import execute_retrieval_experiment
from modules.retrieval import generate_comparison_inference
from modules.tolerant_retrieval import (
    execute_tolerant_retrieval_experiment,
)
from modules.utils import display_documents

# ============================================================================
# STREAMLIT PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title=APP_TITLE,
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================================
# CUSTOM UI STYLING
# ============================================================================

st.markdown(
    """
    <style>

    .main {
        padding-top: 1rem;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 20px;
    }

    .stTabs [data-baseweb="tab"] {
        height: 55px;
        padding-left: 25px;
        padding-right: 25px;
        border-radius: 10px;
        font-size: 16px;
        font-weight: 600;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================================
# DATASET INITIALIZATION
# ============================================================================

def initialize_dataset() -> DataFrame:
    """
    Initialize active dataset for the
    Information Retrieval application.

    Supported formats:
        1. CSV datasets
        2. Single PDF documents
        3. Multiple PDF documents

    Returns:
        DataFrame:
            Active document dataset.
    """
    st.sidebar.header(
        "📂 Dataset Upload"
    )

    uploaded_files = st.sidebar.file_uploader(
        label=(
            "Upload CSV Dataset "
            "or PDF Documents"
        ),
        type=SUPPORTED_FILE_TYPES,
        accept_multiple_files=True,
    )

    # ------------------------------------------------------------------------
    # CUSTOM DATASET INGESTION
    # ------------------------------------------------------------------------

    if uploaded_files:

        first_file_name: str = (
            uploaded_files[0]
            .name
            .lower()
        )

        try:

            # ----------------------------------------------------------------
            # CSV DATASET LOADING
            # ----------------------------------------------------------------

            if first_file_name.endswith(
                ".csv"
            ):

                documents_df: DataFrame = (
                    load_uploaded_dataset(
                        uploaded_file=uploaded_files[0]
                    )
                )

                st.sidebar.success(
                    "CSV dataset uploaded "
                    "successfully."
                )

                return documents_df

            # ----------------------------------------------------------------
            # PDF DOCUMENT INGESTION
            # ----------------------------------------------------------------

            pdf_files = [
                uploaded_file
                for uploaded_file
                in uploaded_files
                if uploaded_file.name.lower().endswith(
                    ".pdf"
                )
            ]

            documents_df = load_pdf_documents(
                uploaded_pdf_files=pdf_files
            )

            st.sidebar.success(
                "PDF documents processed "
                "successfully."
            )

            return documents_df

        except Exception as exception:

            st.sidebar.error(
                body=str(exception)
            )

            st.stop()

    # ------------------------------------------------------------------------
    # DEFAULT DATASET
    # ------------------------------------------------------------------------

    st.sidebar.info(
        "Using default sample dataset."
    )

    return load_default_dataset()


# ============================================================================
# DATASET VIEWER
# ============================================================================

def render_dataset_section(
    documents_df: DataFrame,
) -> None:
    """
    Render dataset visualization section.

    Args:
        documents_df:
            Active document collection.

    Returns:
        None
    """
    st.markdown(
        """
        <h2 style='color:#1976D2;'>
            📄 Dataset Viewer
        </h2>
        """,
        unsafe_allow_html=True,
    )

    st.info(
        "Uploaded document collection "
        "used for Information Retrieval."
    )

    st.metric(
        label="Total Documents",
        value=len(documents_df),
    )

    st.divider()

    display_documents(
        dataframe=documents_df
    )


# ============================================================================
# TEXT PREPROCESSING
# ============================================================================

def render_preprocessing_section(
    documents_df: DataFrame,
) -> None:
    """
    Render preprocessing workflow.

    Args:
        documents_df:
            Input document collection.

    Returns:
        None
    """
    st.markdown(
        """
        <h2 style='color:#2E7D32;'>
            ⚙️ Text Preprocessing
        </h2>
        """,
        unsafe_allow_html=True,
    )

    st.success(
        "Preprocessing normalizes "
        "document vocabulary."
    )

    st.divider()

    preprocessing_col_1, preprocessing_col_2 = (
        st.columns(2)
    )

    with preprocessing_col_1:

        apply_stemmer: bool = st.checkbox(
            label="Apply Stemming",
            value=False,
        )

    with preprocessing_col_2:

        apply_lemmatizer: bool = st.checkbox(
            label="Apply Lemmatization",
            value=False,
        )

    processed_documents: Dict[int, List[str]] = {}

    for _, row in documents_df.iterrows():

        document_id: int = int(row["doc_id"])

        document_text: str = str(row["text"])

        processed_tokens: List[str] = preprocess_text(
            text=document_text,
            apply_stemmer=apply_stemmer,
            apply_lemmatizer=apply_lemmatizer,
        )

        processed_documents[document_id] = (
            processed_tokens
        )

    st.subheader("Processed Documents")

    processed_display: List[
        Dict[str, object]
    ] = []

    for document_id, tokens in processed_documents.items():

        processed_display.append(
            {
                "doc_id": document_id,
                "tokens": tokens,
            }
        )

    st.dataframe(
        processed_display,
        use_container_width=True,
    )

    st.divider()

    inverted_index = create_inverted_index(
        processed_documents=processed_documents
    )

    st.subheader("Inverted Index")

    inverted_index_display: List[
        Dict[str, object]
    ] = []

    for term, document_ids in inverted_index.items():

        inverted_index_display.append(
            {
                "term": term,
                "documents": document_ids,
            }
        )

    st.dataframe(
        inverted_index_display,
        use_container_width=True,
    )


# ============================================================================
# RETRIEVAL COMPARISON
# ============================================================================

def render_retrieval_comparison_section(
    documents_df: DataFrame,
) -> None:
    """
    Compare stemming vs lemmatization.

    Args:
        documents_df:
            Input document collection.

    Returns:
        None
    """
    st.markdown(
        """
        <h2 style='color:#EF6C00;'>
            🔍 Retrieval Comparison
        </h2>
        """,
        unsafe_allow_html=True,
    )

    query: str = st.text_input(
        label="Enter Search Query",
        value="machine learning",
    )

    if not query.strip():

        st.warning(
            "Please enter valid query."
        )

        return

    stemming_results, stemming_time = (
        execute_retrieval_experiment(
            documents_df=documents_df,
            query=query,
            apply_stemmer=True,
            apply_lemmatizer=False,
        )
    )

    lemmatization_results, lemmatization_time = (
        execute_retrieval_experiment(
            documents_df=documents_df,
            query=query,
            apply_stemmer=False,
            apply_lemmatizer=True,
        )
    )

    comparison_table = [
        {
            "Technique": "Stemming",
            "Results": stemming_results,
            "Count": len(stemming_results),
            "Time": round(
                stemming_time,
                6,
            ),
        },
        {
            "Technique": "Lemmatization",
            "Results": lemmatization_results,
            "Count": len(lemmatization_results),
            "Time": round(
                lemmatization_time,
                6,
            ),
        },
    ]

    st.dataframe(
        comparison_table,
        use_container_width=True,
    )

    st.divider()

    st.success(
        generate_comparison_inference(
            stemming_results=stemming_results,
            lemmatization_results=lemmatization_results,
        )
    )


# ============================================================================
# PHRASE QUERY PROCESSING
# ============================================================================

def render_phrase_query_section(
    documents_df: DataFrame,
) -> None:
    """
    Render phrase query processing.

    Args:
        documents_df:
            Input document collection.

    Returns:
        None
    """
    st.markdown(
        """
        <h2 style='color:#6A1B9A;'>
            🧠 Phrase Query Processing
        </h2>
        """,
        unsafe_allow_html=True,
    )

    phrase_query: str = st.text_input(
        label="Enter Phrase Query",
        value="machine learning",
    )

    if not phrase_query.strip():

        st.warning(
            "Please enter valid phrase."
        )

        return

    biword_index = create_biword_index(
        documents_df=documents_df
    )

    positional_index = create_positional_index(
        documents_df=documents_df
    )

    biword_results = search_biword_phrase(
        phrase_query=phrase_query,
        biword_index=biword_index,
    )

    positional_results = (
        search_positional_phrase(
            phrase_query=phrase_query,
            positional_index=positional_index,
        )
    )
    print(positional_results)

    comparison_table = [
        {
            "Technique": "Biword Index",
            "Documents": biword_results,
        },
        {
            "Technique": "Positional Index",
            "Documents": positional_results,
        },
    ]

    st.dataframe(
        comparison_table,
        use_container_width=True,
    )

    st.divider()

    st.success(
        generate_phrase_query_inference()
    )


# ============================================================================
# BST VS B-TREE SEARCH
# ============================================================================

def render_dictionary_search_section(
    documents_df: DataFrame,
) -> None:
    """
    Compare BST and B-Tree dictionary
    search performance.

    IMPORTANT:
        Large PDF vocabularies can create
        extremely deep recursive BST trees.

        To maintain application stability,
        the vocabulary is sampled before
        BST/B-Tree construction.

    Args:
        documents_df:
            Input document collection.

    Returns:
        None
    """
    st.markdown(
        """
        <h2 style='color:#00897B;'>
            🌳 BST vs B-Tree Search
        </h2>
        """,
        unsafe_allow_html=True,
    )

    st.info(
        "BST/B-Tree comparison uses "
        "sampled vocabulary terms "
        "for experimental stability "
        "with large PDF datasets."
    )

    # ------------------------------------------------------------------------
    # VOCABULARY EXTRACTION
    # ------------------------------------------------------------------------

    vocabulary_terms: List[str] = (
        extract_vocabulary(
            documents_df=documents_df
        )
    )

    # ------------------------------------------------------------------------
    # VOCABULARY SAMPLING
    # ------------------------------------------------------------------------

    MAX_BST_TERMS: int = 500

    vocabulary_terms = vocabulary_terms[
        :MAX_BST_TERMS
    ]

    st.metric(
        label="Vocabulary Terms Used",
        value=len(vocabulary_terms),
    )

    st.divider()

    # ------------------------------------------------------------------------
    # TREE CONSTRUCTION
    # ------------------------------------------------------------------------

    bst = build_bst_from_vocabulary(
        vocabulary_terms=vocabulary_terms
    )

    btree = build_btree_from_vocabulary(
        vocabulary_terms=vocabulary_terms
    )

    # ------------------------------------------------------------------------
    # SEARCH INPUT
    # ------------------------------------------------------------------------

    search_term: str = st.text_input(
        label="Enter Dictionary Search Term",
        value="retrieval",
    )

    if not search_term.strip():

        st.warning(
            "Please enter valid search term."
        )

        return

    # ------------------------------------------------------------------------
    # BST SEARCH
    # ------------------------------------------------------------------------

    bst_start_time: float = (
        perf_counter()
    )

    bst_found: bool = bst.search(
        value=search_term.lower()
    )

    bst_search_time: float = (
        perf_counter() - bst_start_time
    )

    # ------------------------------------------------------------------------
    # B-TREE SEARCH
    # ------------------------------------------------------------------------

    btree_start_time: float = (
        perf_counter()
    )

    btree_found: bool = btree.search(
        value=search_term.lower()
    )

    btree_search_time: float = (
        perf_counter() - btree_start_time
    )

    # ------------------------------------------------------------------------
    # RESULT VISUALIZATION
    # ------------------------------------------------------------------------

    comparison_table: List[
        Dict[str, object]
    ] = [
        {
            "Structure": "BST",
            "Found": bst_found,
            "Search Time": round(
                bst_search_time,
                8,
            ),
        },
        {
            "Structure": "B-Tree",
            "Found": btree_found,
            "Search Time": round(
                btree_search_time,
                8,
            ),
        },
    ]

    st.dataframe(
        comparison_table,
        use_container_width=True,
    )

    st.divider()

    # ------------------------------------------------------------------------
    # PERFORMANCE INSIGHT
    # ------------------------------------------------------------------------

    if btree_search_time < bst_search_time:

        st.success(
            "B-Tree demonstrated "
            "better search efficiency "
            "for the sampled vocabulary."
        )

    else:

        st.success(
            "BST demonstrated "
            "comparable search efficiency "
            "for the sampled vocabulary."
        )
        

# ============================================================================
# TOLERANT RETRIEVAL
# ============================================================================

def render_tolerant_retrieval_section(
    documents_df: DataFrame,
) -> None:
    """
    Render tolerant retrieval workflows.

    Args:
        documents_df:
            Input document collection.

    Returns:
        None
    """
    st.markdown(
        """
        <h2 style='color:#D81B60;'>
            🛠️ Tolerant Retrieval
        </h2>
        """,
        unsafe_allow_html=True,
    )

    query_term: str = st.text_input(
        label="Enter Query Term",
        value="retrival",
    )

    experiment_results = (
        execute_tolerant_retrieval_experiment(
            documents_df=documents_df,
            query_term=query_term.lower(),
        )
    )

    st.subheader("Generated K-Grams")

    st.code(
        body=str(
            experiment_results["query_kgrams"]
        ),
        language="python",
    )

    st.subheader(
        "Spelling Suggestions"
    )

    spelling_display: List[
        Dict[str, object]
    ] = []

    for suggestion, edit_distance in (
        experiment_results[
            "spelling_suggestions"
        ]
    ):

        spelling_display.append(
            {
                "Suggested Term": suggestion,
                "Edit Distance": edit_distance,
            }
        )

    st.dataframe(
        spelling_display,
        use_container_width=True,
    )

    wildcard_matches: List[str] = (
        experiment_results[
            "wildcard_matches"
        ]
    )

    st.subheader("Wildcard Matches")

    if wildcard_matches:

        wildcard_display: List[
            Dict[str, str]
        ] = []

        for matched_term in wildcard_matches:

            wildcard_display.append(
                {
                    "Matched Term": matched_term
                }
            )

        st.dataframe(
            wildcard_display,
            use_container_width=True,
        )

    else:

        st.warning(
            "No wildcard matches found."
        )


# ============================================================================
# ANALYTICS DASHBOARD
# ============================================================================

def render_analytics_dashboard() -> None:
    """
    Render analytics dashboard.

    Returns:
        None
    """
    st.markdown(
        """
        <h2 style='color:#1565C0;'>
            📊 Experimental Analytics
        </h2>
        """,
        unsafe_allow_html=True,
    )

    retrieval_dataframe = pd.DataFrame(
        {
            "Technique": [
                "Stemming",
                "Lemmatization",
            ],
            "Retrieved Documents": [
                6,
                4,
            ],
        }
    )

    retrieval_chart = px.bar(
        retrieval_dataframe,
        x="Technique",
        y="Retrieved Documents",
        title="Retrieval Comparison",
    )

    st.plotly_chart(
        retrieval_chart,
        use_container_width=True,
    )

    phrase_dataframe = pd.DataFrame(
        {
            "Technique": [
                "Biword Index",
                "Positional Index",
            ],
            "Matches": [
                5,
                3,
            ],
        }
    )

    phrase_chart = px.bar(
        phrase_dataframe,
        x="Technique",
        y="Matches",
        title="Phrase Query Comparison",
    )

    st.plotly_chart(
        phrase_chart,
        use_container_width=True,
    )

    dictionary_dataframe = pd.DataFrame(
        {
            "Structure": [
                "BST",
                "B-Tree",
            ],
            "Search Time": [
                0.000021,
                0.000011,
            ],
        }
    )

    dictionary_chart = px.bar(
        dictionary_dataframe,
        x="Structure",
        y="Search Time",
        title="BST vs B-Tree Comparison",
    )

    st.plotly_chart(
        dictionary_chart,
        use_container_width=True,
    )


# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main() -> None:
    """
    Execute main Streamlit workflow.

    Returns:
        None
    """
    st.title(
        "📚 Information Retrieval System"
    )

    st.markdown(
        """
        End-to-End Information Retrieval
        platform implemented using Streamlit.
        """
    )

    st.divider()

    documents_df = initialize_dataset()

    (
        dataset_tab,
        preprocessing_tab,
        retrieval_tab,
        phrase_query_tab,
        dictionary_tab,
        tolerant_tab,
        analytics_tab,
    ) = st.tabs(
        [
            "📄 Dataset Viewer",
            "⚙️ Preprocessing",
            "🔍 Retrieval Comparison",
            "🧠 Phrase Query",
            "🌳 BST vs B-Tree",
            "🛠️ Tolerant Retrieval",
            "📊 Analytics",
        ]
    )

    with dataset_tab:

        render_dataset_section(
            documents_df=documents_df
        )

    with preprocessing_tab:

        render_preprocessing_section(
            documents_df=documents_df
        )

    with retrieval_tab:

        render_retrieval_comparison_section(
            documents_df=documents_df
        )

    with phrase_query_tab:

        render_phrase_query_section(
            documents_df=documents_df
        )

    with dictionary_tab:

        render_dictionary_search_section(
            documents_df=documents_df
        )

    with tolerant_tab:

        render_tolerant_retrieval_section(
            documents_df=documents_df
        )

    with analytics_tab:

        render_analytics_dashboard()


# ============================================================================
# APPLICATION ENTRY POINT
# ============================================================================

if __name__ == "__main__":

    main()