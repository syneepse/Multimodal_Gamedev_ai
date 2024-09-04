# import gradio as gr
# import modelLib

# # def greet(name, intensity):
# #     return "Hello, " + name + "!" * int(intensity)



# demo = gr.Interface(
#     fn=modelLib.PromptInput,
#     inputs=gr.MultimodalTextbox( placeholder="prompt or file" ),
#     outputs=["text", "image", gr.Model3D()],
# )

# demo.launch()
import gradio as gr
import modelLib


def greet(name):
    print(name)
    return name

def change_Output(choice):
    Model_name, Image_bool, Output_Image, Output_Model = modelLib.PromptInput(choice)
    print(Model_name, Image_bool, Output_Model, Output_Image)
    if  not Image_bool:
        return {description: gr.Markdown("Model Name: "  ,visible=True), view3D: gr.Model3D(value=Output_Model , visible=True), OutputImage: gr.Image(visible=False)}
    else:
        return {description: gr.Markdown("Model Name: " ,visible=True), view3D: gr.Model3D(visible=False),OutputImage: gr.Image(value= 'Sample-Image.jpg', visible=True)}

with gr.Blocks() as demo:
    user_input = gr.MultimodalTextbox( placeholder="prompt or file" )
    description = gr.Markdown(visible=False)
    OutputImage = gr.Image(visible=False)
    view3D = gr.Model3D(visible=False)
    gr.Button("Submit").click(
        fn=change_Output,
        inputs=user_input,
        outputs=[description,view3D,OutputImage]
    )
   

demo.launch()
