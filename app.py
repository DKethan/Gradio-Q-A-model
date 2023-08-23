import gradio as gr
from transformers import pipeline

qa_pipeline = pipeline('question-answering', model='bert-large-uncased-whole-word-masking-finetuned-squad')


def answer_question(context, question):
    result = qa_pipeline(context=context, question=question)
    return result['answer']

iface = gr.Interface(fn=answer_question,
                     inputs=[gr.inputs.Textbox(lines=7,label='context'),gr.inputs.Textbox(lines=2,label='question')],
                     outputs=gr.outputs.Textbox(label='answers')
                     )

iface.launch()
