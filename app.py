from flask import Flask, request, render_template
import numpy as np 
from PIL import Image
from keras.models import load_model

app = Flask(__name__)

class_names ={0:'Blight', 
             1:'Common_Rust',
             2:'Grey_Leaf_Spot',
             3: 'Healthy'}

model = load_model('model.h5')

model.make_predict_function()

def predict_label(img_path):
    
    image = Image.open(img_path)
    # Resize the image to (255, 255)
    resized_image = image.resize((255, 255))

    # Convert the PIL image to a NumPy array
    image_array = np.array(resized_image)

    # Assuming you have a model that expects input with shape (1, 255, 255, 3)
    input_image = image_array.reshape((1, 255, 255, 3))

    # Now you can use input_image for making predictions
    predictions = model.predict(input_image)

    # Assuming predictions is a NumPy array, you can get the predicted class
    predicted_class = np.argmax(predictions, axis=-1)

    predicted_class_name = class_names[predicted_class[0]]
   
    return predicted_class_name

@app.route("/", methods=['GET', 'POST'])
def main():
	return render_template("index.html")

@app.route("/submit", methods = ['GET', 'POST'])
def get_output():
	if request.method == 'POST':
		img = request.files['my_image']

		img_path = "static/" + img.filename	

		p = predict_label(img_path)
   
	return render_template("index.html", prediction = p, img_path = img_path)

if __name__ == '__main__':
    app.run( debug=True)