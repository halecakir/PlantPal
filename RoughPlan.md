# PlantPal - Rough Plan


Chat Bot's Capabilities
	-   Provide care instructions for various plants.
            Light requirements
            Watering frequency
            Preferred soil type
            Ideal temperature
            Common diseases and pests
            Pruning tips
	-   Diagnose plant diseases based on user descriptions.
		
	-   Suggest plants suitable for certain conditions (e.g., low light, pet-safe, etc.).
            Light requirements
            Temperature tolerance
            Pet safety
            Difficulty of care

	-   Provide advice on plant propagation, pruning, and repotting.




Design the conversation flow
	According to user's needs from our bot, we are going to define intent and entities.
	We need to create different conversation path for each use case.

	-   Handling Exceptions
		    We need consider what our chat do if it doesn't understand the user's intent. 
	-   Designing the flowchart
		    We need to design the nodes and transitions btw nodes.
		    A node represents a message or action.
		    Flow Builder to create nodes and transitions.
		    We can test it to make sure it works as expected.
	-   Iterate and Improve
		    We can start with simple NLU modes.
		    Then we'll likely find key areas which need improvement.
		    According the pain points, we can refine the conversational paths and NLU model.

Platform
	Botpress  https://github.com/botpress/botpress


Data Collection
    Data for Knowledge Base
		Defining the Scope of the Data
			Common and scientific names
			Light requirements
			Watering frequency
			Preferred soil type and pH
			Ideal temperature range
			Propagation methods
			Pruning advice
			Common diseases and pests
			Safe for pets or not
			Plant images (if your bot will support image recognition)
			Symptoms and treatment for common diseases
		Web Scraping
			We can use scrapy to collect raw data from the web
		External KB
			We need to investigate the existing KBs such as WikiData
		Extracting the Relevant Features from the Raw Data
			In this step, we can use standard NLP flows, also we can leverage the LLMs too.
				Data Cleaning and Preprocessing
				Information Extraction
					Entitit Extraction:
						NER models
						LLM
					e.g. 
					Input : The asparagus plant requires indirect sunlight and moderate watering
					Output : Asparagus [PLANT]
						 indirect sunlight [LIGHT_REQUIREMENT]
						 moderate watering [WATERING_REQUIREMENT]
					Text Classification:
						Classification can help us to identify which part of text is relevant about care instructions and which parts of are about diseases.
				Validation of the Data	 
					Manual Validation	
		Defining Structure of the Data and Implementation
			Databases or KBs
			
	Data Collection for NLU
		We need to collect data for entity recognition
		We may create a few example for each entity then ask the ChatGPT to augment it.
			
					
Train & Test NLU model
	We need to train NLU model based on the LLMs.
	

Implemetation of UI and Deployment
		Botpress has internal UI.
		Other options would be to use existing chat tools such as Telegram
		Custom App ? 



Using large language models (LLMs) to synthesize training data
	https://www.amazon.science/blog/using-large-language-models-llms-to-synthesize-training-data

Research	
	LLMs for intent recognition
		https://txt.cohere.com/llms-for-intent-recognition/
		https://cobusgreyling.medium.com/bootstrapping-a-chatbot-with-a-large-language-model-93fdf5540a1b
	
	
	
Bot Framework
	Botpress : https://github.com/botpress/botpress
	http://localhost:3000/studio/weather/flows/main 
	Example bots from healtcare domain : 
		https://botpress.com/blog/top-health-chatbots-for-2021
		https://botpress.com/blog/chatbots-for-healthcare

	