from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM

from services.bert import search

PROMPT_TEMPLATE = """
Act as a information retrieval expert.
The user will provide you with a query and context.
Dont say "Based on the provided context" or any similar words.
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""


def query_rag(query_text: str, dataset: str, limit=5):
    results = search(query_text, dataset, limit)

    context_text = "\n\n---\n\n".join([result[0].page_content for result in results])

    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    model = OllamaLLM(model="mistral")
    response_text = model.invoke(prompt)

    sources = [result[0].metadata.get("id", None) for result in results]
    return {"content": response_text, "sources": sources}
