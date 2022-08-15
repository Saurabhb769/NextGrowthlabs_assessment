

import pickle
from flask import Flask,request,jsonify,render_template

from tensorflow import keras
import numpy as np


app = Flask(__name__,template_folder="template")
model=keras.models.load_model("booking_hotel_chekedin.h5")

@app.route("/")
def home():
	return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
	"""
	for rendering results on HTML
	"""	
	feature = [float(x) for x in request.form.values()]
	print(feature)
	# re-arranging the list as per data set
	feature_list = ["Age", "DaysSinceCreation", "AverageLeadTime", "LodgingRevenue",
       "OtherRevenue", "PersonsNights", "RoomNights", "DistributionChannel",
       "MarketSegment", "SRHighFloor", "SRLowFloor", "SRAccessibleRoom",
       "SRMediumFloor", "SRBathtub", "SRShower", "SRCrib", "SRKingSizeBed",
       "SRTwinBed", "SRNearElevator", "SRAwayFromElevator",
       "SRNoAlcoholInMiniBar", "SRQuietRoom"]
	
	
	prediction = model.predict(np.array(feature).reshape(1,-1))

	
	print("prediction value: ", prediction)

	result = ""
	if prediction == 0:
		result = "NOT CHEKED IN"
	else:
		result = "CHEKED IN"

	return  result# render_template("index.html", prediction_text = result)


if __name__ == "__main__":
	app.run(debug=True)


