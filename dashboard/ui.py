import hashlib
from pathlib import Path

import streamlit as st

from ingestion.pipeline import ingest_file
from embeddings.embedder import Embedder
from vectorstore.store import add_documents, get_count
from chat.pipeline import RAGPipeline


# ============================================================
# CONFIGURATION
# ============================================================

UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# RESOURCE INITIALIZATION
# ============================================================

@st.cache_resource
def get_embedder():
    return Embedder()


@st.cache_resource
def get_rag_pipeline():
    return RAGPipeline()


# ============================================================
# MAIN APPLICATION
# ============================================================

def run_app():

    # --------------------------------------------------------
    # PAGE CONFIG
    # --------------------------------------------------------

    st.set_page_config(
        page_title="CatBot RAG",
        page_icon="🐱",
        layout="wide"
    )

    # --------------------------------------------------------
    # LOAD COMPONENTS
    # --------------------------------------------------------

    embedder = get_embedder()
    rag = get_rag_pipeline()

    # --------------------------------------------------------
    # HEADER
    # --------------------------------------------------------

    st.title("🐱 CatBot")

    st.caption(
        "Document-based Retrieval Augmented Generation"
    )

    # ========================================================
    # SIDEBAR — KNOWLEDGE BASE
    # ========================================================

    with st.sidebar:

        st.header("Knowledge Base")

        uploaded_file = st.file_uploader(
            "Upload a document",
            type=["pdf", "docx", "txt"]
        )

        add_button = st.button(
            "Add Knowledge",
            use_container_width=True
        )

        st.divider()

        st.write(
            f"Stored chunks: {get_count()}"
        )

    # ========================================================
    # KNOWLEDGE INGESTION
    # ========================================================

    if add_button:

        if uploaded_file is None:

            st.warning(
                "Upload a PDF, DOCX, or TXT file first."
            )

        else:

            file_path = (
                UPLOAD_DIR / uploaded_file.name
            )

            try:

                # ------------------------------------------------
                # Save uploaded file
                # ------------------------------------------------

                file_path.write_bytes(
                    uploaded_file.getbuffer()
                )

                with st.spinner(
                    "Processing document..."
                ):

                    # ------------------------------------------------
                    # STEP 1: PARSE + CHUNK
                    # ------------------------------------------------

                    chunks = ingest_file(
                        str(file_path)
                    )

                    if not chunks:
                        raise ValueError(
                            "No text could be extracted "
                            "from the document."
                        )

                    # ------------------------------------------------
                    # STEP 2: CREATE EMBEDDINGS
                    # ------------------------------------------------

                    embeddings = (
                        embedder.embed_documents(
                            chunks
                        )
                    )

                    # ------------------------------------------------
                    # STEP 3: CREATE METADATA
                    # ------------------------------------------------

                    metadatas = [
                        {
                            "source": uploaded_file.name,
                            "chunk_index": index
                        }
                        for index in range(len(chunks))
                    ]

                    # ------------------------------------------------
                    # STEP 4: CREATE STABLE IDS
                    # ------------------------------------------------

                    ids = []

                    for index, chunk in enumerate(chunks):

                        raw_id = (
                            f"{uploaded_file.name}"
                            f"_{index}_"
                            f"{chunk}"
                        )

                        chunk_id = hashlib.sha256(
                            raw_id.encode("utf-8")
                        ).hexdigest()

                        ids.append(chunk_id)

                    # ------------------------------------------------
                    # STEP 5: STORE IN VECTOR DATABASE
                    # ------------------------------------------------

                    add_documents(
                        chunks=chunks,
                        embeddings=embeddings,
                        metadatas=metadatas,
                        ids=ids
                    )

                st.success(
                    f"Knowledge added successfully. "
                    f"{len(chunks)} chunks stored."
                )

            except Exception as error:

                st.error(
                    f"Failed to add knowledge: {error}"
                )

    # ========================================================
    # CHAT
    # ========================================================

    st.header("Chat with your knowledge")

    question = st.chat_input(
        "Ask something about your documents..."
    )

    if question:

        # ----------------------------------------------------
        # USER MESSAGE
        # ----------------------------------------------------

        with st.chat_message("user"):
            st.write(question)

        try:

            with st.spinner(
                "Searching knowledge and generating answer..."
            ):

                # ------------------------------------------------
                # COMPLETE RAG PIPELINE
                # ------------------------------------------------

                result = rag.ask(question)

            # ----------------------------------------------------
            # AI RESPONSE
            # ----------------------------------------------------

            with st.chat_message("assistant"):

                st.write(
                    result["answer"]
                )

                # ------------------------------------------------
                # SOURCES
                # ------------------------------------------------

                sources = result.get(
                    "sources",
                    []
                )

                if sources:

                    with st.expander(
                        "View sources"
                    ):

                        for index, source in enumerate(
                            sources,
                            start=1
                        ):

                            metadata = source.get(
                                "metadata",
                                {}
                            )

                            source_name = metadata.get(
                                "source",
                                "Unknown"
                            )

                            st.markdown(
                                f"**Source {index}: "
                                f"{source_name}**"
                            )

                            st.write(
                                source.get(
                                    "text",
                                    ""
                                )
                            )

                            st.divider()

        except Exception as error:

            st.error(
                f"Failed to generate answer: {error}"
            )