import gradio as gr
from retina_model.model import DetectionModel


model = DetectionModel()


demo = gr.Interface(
    fn=model.detect,
    inputs=[gr.Image(type="numpy")],
    outputs=gr.Image(),
    flagging_mode="never"
)


demo.launch()