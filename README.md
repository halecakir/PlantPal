# PlantPal - The AI Botanist

Welcome to the PlantPal documentation. 

Firstly, this guide will walk you through the deployment and testing procedures for the PlantPal runtime services wh.ich is defined under `/app` folder. Then, this repository stores the code generated during our research activities under the `/research` folder. It may appear more disorganized compared to the application code, as we need to implement various models and algorithms throughout the term. The `/app` codes can be considered our final product. Additionally, we store some utility scripts, such as those for crawling and data generation.

## 1. PlantPal Runtime System

### 1.1 Building & Running Service

To build and run both the backend and frontend services, execute the following command in your terminal:

```sh
docker-compose -p plantpal up -d --build
```

To verify that the services are up and running, you can list the running containers using the following command:

```sh
docker ps
```

### 1.2 Testing Runtime Services
* To ensure that the backend services are running correctly, you can execute the following Python code in your favorite integrated development environment (IDE):


```python

import requests
url = 'http://localhost:8000/predict'
data = {
    'text': "How much watering does my snake plant needs ?",
    'command': "intent",
    'info': {}
}
response = requests.post(url, json=data)
prediction = response.json()
print(prediction['prediction'])
```

* To test the frontend services using Streamlit, simply open the following URL in your web browser:


```
http://localhost:8502/
```

**NB.** : If your frontend service is hosted remotely, be sure to handle port forwarding accordingly.


## 2. PlanPal Offline (Research) Codes

This section of the repository stores the code generated during our research activities. It may appear more disorganized compared to the application code, as we need to implement various models and algorithms throughout the term. The `/app` codes can be considered our final product. Additionally, we store some utility scripts, such as those for crawling and data generation.

### 2.1. Training Codes (/research/training)
In this directory, we store the necessary Python and Jupyter files for training intent-based or LLM-based models.

- **JointAlbert** contains the essential scripts for building the Joint Intent Identification and Slot Tagging model.
- **LlamaChat_rqa** represents the chat version of the LLaMa model, which serves as a sentence realizor.
- The **LLama2** script is used for fine-tuning the LLaMa2 - 7B model. We employ this fine-tuned model in disease identification. For more details, please refer to the HOW-TO document available at `/research/training/llama2/README.md`.
- **Other** scripts pertain to deprecated models. Throughout the term, we experimented with various models, some of which did not perform well. Therefore, we have categorized them as "other."

### 2.2. Data (/research/data)
This section contains essential data files for our models. The most significant ones include:

- `data/PlantQA_dataset_8864.csv`: This is a cleaned version of the Reddit data, which we will use for fine-tuning the llama-2 model.
- `data/reddit/plantclinic_***`: These are the raw versions of the Reddit plant clinic dataset.
- `data/intent.csv`: This dataset is used for intent identification.

### 2.3. Scripts (/research/scripts)
This directory contains utility scripts.

