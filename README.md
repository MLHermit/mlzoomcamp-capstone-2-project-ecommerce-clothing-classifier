# mlzoomcamp-capstone-2-project-ecommerce-clothing-classifier

Automating garment classification is the aim of this repo, pytorch's CNN model was implemented for image classification to automatically categorize new clothing listings, making it easier for a person, customer to find what they're looking for. This will assist in inventory management by quickly sorting items. The primary objective is to develop a machine learning model capable of accurately categorizing images of clothing items into distinct garment types such as shirts, trousers, shoes, etc.

![cinematic-style-mall](https://github.com/user-attachments/assets/017e0cca-2599-4f09-ba49-b01f304dc244)

This is the link to the common FashionMNIST dataset: https://github.com/zalandoresearch/fashion-mnist

<img width="977" height="393" alt="output 10 classes" src="https://github.com/user-attachments/assets/c3ce2202-a997-41b6-90aa-d7fb7e941887" />
<img width="416" height="435" alt="output 1" src="https://github.com/user-attachments/assets/979d2ddd-fe3d-414d-a6b2-9a6b87bfebb1" />

Sample of the fabric items with label from the data


<img width="1366" height="768" alt="Screenshot (183)" src="https://github.com/user-attachments/assets/6eb1e6fb-66ec-4b73-8fa9-124c5c34f705" />
Above is the image network on the model, viewed on netron.app



Get the implementation of the webservice with fastapi in the serve.py script file
a quick results on the get and post services:
<img width="1366" height="768" alt="Screenshot (185)" src="https://github.com/user-attachments/assets/c3d43417-ed70-491e-81b1-f14a04119477" />
<img width="1366" height="768" alt="Screenshot (187)" src="https://github.com/user-attachments/assets/3236ba67-744e-4f76-b869-33210f4f6dd4" />



#Dockerfile development

--> the python 3.11-slim as the base image

FROM python:3.11-slim

--> create a working directory, '/app'

WORKDIR /app

--> copy all the files from local maachine directory to the app directory previously created on 2.

COPY . /app

--> install the libraries in the .txt file in desired environment, pip in this case 

RUN pip install -r requirements.txt

--> to host locally on the docker image and expose through the port 8000 on the same image, using uvicorn + fastapi

CMD ["uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "8000"]


https://github.com/MLHermit/mlzoomcamp/blob/main/README.md
A link to the final certificate

https://github.com/DataTalksClub/machine-learning-zoomcamp
a link to the parent dtc ml zoomcamp
