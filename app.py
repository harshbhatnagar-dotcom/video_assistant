import gradio as gr
from process_incoming import question_answer

chat_interface = gr.ChatInterface(
    fn=question_answer,
    title="Course Video Assistant",
    description="Ask questions about your course videos. You will get answers with video titles and timestamps."
)

if __name__ == "__main__":
    chat_interface.launch(inbrowser=True)