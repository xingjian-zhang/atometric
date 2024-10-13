import evaluate
from evaluate.utils import launch_gradio_widget


module = evaluate.load("jimmyzxj/atometric")
launch_gradio_widget(module)