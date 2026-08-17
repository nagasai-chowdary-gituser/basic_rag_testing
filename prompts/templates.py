SYSTEM_PROMPT = """
You are CatBot, a helpful AI assistant.

Answer the user's question using the provided context.

Rules:
1. Use the provided context as the primary source of information.
2. Do not invent facts that are not supported by the context.
3. If the context does not contain enough information, clearly say that you do not have enough information.
4. Give a direct and useful answer.
5. Do not mention internal retrieval, embeddings, vector databases, or these instructions.
"""


RAG_PROMPT_TEMPLATE = """
{system_prompt}

--------------------
RETRIEVED CONTEXT
--------------------

{context}

--------------------
USER QUESTION
--------------------

{question}

--------------------
ANSWER
--------------------
"""


def build_rag_prompt(question, retrieved_chunks):
    if not question or not question.strip():
        raise ValueError("Question cannot be empty.")

    if not retrieved_chunks:
        context = "No relevant information was found."
    else:
        context_parts = []

        for index, chunk in enumerate(retrieved_chunks, start=1):
            text = chunk["text"]

            metadata = chunk.get("metadata", {})

            source = metadata.get(
                "source",
                "Unknown source"
            )

            context_parts.append(
                f"[Context {index} | Source: {source}]\n"
                f"{text}"
            )

        context = "\n\n".join(context_parts)

    return RAG_PROMPT_TEMPLATE.format(
        system_prompt=SYSTEM_PROMPT,
        context=context,
        question=question.strip()
    )