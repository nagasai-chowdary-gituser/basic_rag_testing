from retrievel.retriever import Retriever
from prompts.templates import build_rag_prompt
from llm.generator import LLMGenerator


class RAGPipeline:

    def __init__(self):
        self.retriever = Retriever()
        self.generator = LLMGenerator()

    def ask(self, question):

        if not question or not question.strip():
            raise ValueError("Question cannot be empty.")

        # 1. Retrieve relevant chunks
        retrieved_chunks = self.retriever.retrieve(
            question
        )

        # 2. Build RAG prompt
        prompt = build_rag_prompt(
            question=question,
            retrieved_chunks=retrieved_chunks
        )

        # 3. Generate answer
        answer = self.generator.generate(
            prompt
        )

        return {
            "question": question,
            "answer": answer,
            "sources": retrieved_chunks
        }