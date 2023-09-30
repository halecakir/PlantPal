# PlanPal Offline (Research) Codes

This section of the repository stores the code generated during our research activities. It may appear more disorganized compared to the application code, as we need to implement various models and algorithms throughout the term. The `/app` codes can be considered our final product. Additionally, we store some utility scripts, such as those for crawling and data generation.

## 1. Training Codes (/research/training)
In this directory, we store the necessary Python and Jupyter files for training intent-based or LLM-based models.

- **JointAlbert** contains the essential scripts for building the Joint Intent Identification and Slot Tagging model.
- **LlamaChat_rqa** represents the chat version of the LLaMa model, which serves as a sentence realizor.
- The **LLama2** script is used for fine-tuning the LLaMa2 - 7B model. We employ this fine-tuned model in disease identification. For more details, please refer to the HOW-TO document available at `/research/training/llama2/README.md`.
- **Other** scripts pertain to deprecated models. Throughout the term, we experimented with various models, some of which did not perform well. Therefore, we have categorized them as "other."

## 2. Data (/research/data)
This section contains essential data files for our models. The most significant ones include:

- `data/PlantQA_dataset_8864.csv`: This is a cleaned version of the Reddit data, which we will use for fine-tuning the llama-2 model.
- `data/reddit/plantclinic_***`: These are the raw versions of the Reddit plant clinic dataset. (!!!This dataset was removed from the repository due to its enormously huge size)
- `data/intent.csv`: This dataset is used for intent identification.

## 3. Scripts (/research/scripts)
This directory contains utility scripts.
