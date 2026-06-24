import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from groq import Groq
from config import groq_api_key, groq_model
from retriever import retrieve

# configure Groq once
client = Groq(api_key=groq_api_key)

def build_prompt(query: str, chunks: list[dict]) -> str:
    """
    Builds the prompt that gets sent to Gemini.
    Combines retrieved context chunks with teh student's question.
    """
    context_blocks = []

    for i, chunk in enumerate(chunks):
        block = (
            f"[Source{i+1}: {chunk['source']}, Page {chunk['page']}]\n"
            f"{chunk['text']}"
        )
        context_blocks.append(block)

    context = "\n\n".join(context_blocks)

    prompt = f"""You are a helpful Data Science teaching assistant. 
Answer the student;s question using ONLY the context provided below from their textbooks. 
If the context doesn't contain enough information to answer, say"I couldn't find information in the textbooks to answer this." 
Always mention which source and page your answer is based on. 

CONTEXT FROM TEXTBOOK:
{context}

STUDENT QUESTION:
{query}

ANSWER:"""
    
    return prompt

def ask(query: str) -> dict:
    """
    Main function — takes a question, retrieves context,
    builds prompt, calls Groq, returns answer + sources.
    """

    # step 1 — retrieve relevant chunks
    chunks = retrieve(query)

    # step 2 — build prompt
    prompt = build_prompt(query, chunks)

    # step 3 — call Groq
    response = client.chat.completions.create(
        model    = groq_model,
        messages = [
            {
                "role"   : "system",
                "content": "You are a helpful Data Science teaching assistant who answers questions strictly based on provided textbook context."
            },
            {
                "role"   : "user",
                "content": prompt
            }
        ],
        temperature = 0.3,
        max_tokens  = 1024
    )

    # step 4 — extract answer
    answer = response.choices[0].message.content

    # step 5 — extract sources for citation
    sources = [
        f"{chunk['source']} (Page {chunk['page']})"
        for chunk in chunks
    ]
    # deduplicate sources
    sources = list(dict.fromkeys(sources))

    return {
        "answer" : answer,
        "sources": sources
    }


# Quick TEST
if __name__ == "__main__":
    query = "What is LLM?"
    result = ask(query)

    print(f"\nQuestion: {query}")
    print(f"\nAnswer:\n{result['answer']}")
    print(f"\nSources:")
    for source in result['sources']:
        print(f" - {source}")