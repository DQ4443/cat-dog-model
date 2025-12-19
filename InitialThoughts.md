You are tasked with building a web application that takes an image from the user's webcam and sends it to a Flask backend.
The backend (tflite file) should determine whether the image corresponds to a dog or a cat.
This decision will be made by a provided tflite file that contains a trained tensorflow classification model that predicts the probability that the image corresponds to either animal.
The backend should store the results in a database, and respond to the user with the resulting score and the class label.

Technologies

- frontend: doesn't matter
- backend: flask (python)
- db: doesn't matter

Functional requirements

- capture an image using the user's webcam (research)
- interface with tflite file
- output to user

non-functional reqs

- low latency
- concurrency?
-

inputs and outputs
input: use the user's webcam to capture a picture --> image
image --> backend (tflite file) --> score, class label - model is intentionally obscure --> need to emperically test model to see how it behaves

output: class label (cat or dog), score (with probability) -> show in frontend
