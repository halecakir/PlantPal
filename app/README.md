# PlantPal Runtime System

Welcome to the PlantPal Runtime System documentation. This guide will walk you through the deployment and testing procedures for the PlantPal runtime services.


## Building & Running Service

To build and run both the backend and frontend services, execute the following command in your terminal:

```sh
docker-compose -p plantpal up -d --build
```

To verify that the services are up and running, you can list the running containers using the following command:

```sh
docker ps
```

## Testing Runtime Services
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

