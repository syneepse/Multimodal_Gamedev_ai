from fastapi import FastAPI
import modelLib

app = FastAPI()



@app.post("/get-file")
async def get_file(prompt):
    # Example logic to determine filename based on prompt
    # You should replace this with your actual logic
    modelID, imgbool, outimg,outmodel = modelLib.TextPrompt(prompt)
    
    # Check the file extension to decide where to look for the file
    if imgbool:
        file_path = './'+outimg
        media_type = 'image/png'
    else:
        file_path = './' + outmodel
        media_type = 'model/obj'
    # file_extension = os.path.splitext(filename)[1].lower()

    # if file_extension in ['.png', '.jpg', '.jpeg', '.gif']:
    #     file_path = os.path.join(IMAGE_DIRECTORY, filename)
    #     media_type = 'image/' + file_extension[1:]  # e.g., 'image/png'
    # elif file_extension in ['.obj', '.stl']:  # Add other 3D model formats as needed
    #     file_path = os.path.join(MODEL_DIRECTORY, filename)
    #     media_type = 'model/' + file_extension[1:]  # e.g., 'model/obj'
    # else:
    #     raise HTTPException(status_code=400, detail="Unsupported file type")

    # Check if the file exists
    if not os.path.isfile(file_path):
        raise HTTPException(status_code=404, detail="File not found")

    # Return the file
    return FileResponse(file_path, media_type=media_type)
