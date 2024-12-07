import gradio as gr
from litellm import completion
import glob
import os
from retriever import Retriever
from Reranker import rerank

PROMPT = """\
You are a helpful assistant that can answer questions.
Rules:
- Give answers to all questions asked.
- Reply with the answer only and nothing but the answer.
- Use the provided context.
"""
key = ''


class QuestionAnsweringBot:

    def __init__(self, docs) -> None:
        self.retriever = Retriever(docs)

    def answer_question(self, question: str, api_key: str, methods: list[str]) -> list[str]:
        try:
            key = api_key
            print(api_key)
            os.environ['GROQ_API_KEY'] = key
            if not methods:
                return ["No search method selected. Please select at least one search method.", "-", "-"]

            retr_context = self.retriever.get_docs(question, methods)
            top_k = rerank(retr_context, question).top_k(5)
            context = [result.text for result in top_k]

            messages = [
                {"role": "system", "content": PROMPT},
                {"role": "user", "content": f"Context:\n{context}\nQuestion: {question}"}
            ]
            response = completion(
                model="groq/llama3-8b-8192",
                messages=messages
            )
            return [response.choices[0].message.content, f"{retr_context}", f"{context}"]
        except:
            return ["Error, invalid key", "Error, invalid key", "Error, invalid key"]


docs = []
for path in glob.glob("data/*.txt"):
    with open(path) as f:
        docs.append(f.read())

bot = QuestionAnsweringBot(docs)

demo = gr.Interface(
    fn=bot.answer_question,
    inputs=[
        gr.Textbox(label="Question", placeholder="Ask your question here"),
        gr.Textbox(label="API key", placeholder="Provide API key here"),
        gr.CheckboxGroup(
            ["BM25", "semantic"],
            label="Search Methods",
            info="Choose one search method, or use both."
        ),
    ],
    outputs=[
        gr.Textbox(label="Bot output"),
        gr.Textbox(label="Context from retriever"),
        gr.Textbox(label="Context from retriever after reranker")
    ]
)

demo.launch()
