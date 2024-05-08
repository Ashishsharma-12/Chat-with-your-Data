# Chat-with-your-Data

AI agent for data analysis using LLamma3-8b-Instruct model form Hugging face

**Chat with Your Data**

This repository contains a Streamlit application that enables users to interact with a chatbot model trained on a specified text generation task. The chatbot is capable of processing user queries based on data provided in a CSV file uploaded by the user.

**Functionality**

Chat with Your Data
This repository contains a Streamlit application that enables users to interact with a chatbot model trained on a specified text generation task. The chatbot is capable of processing user queries based on data provided in a CSV file uploaded by the user.

**Environment Setup:** The code initializes necessary libraries and loads environment variables from a .env file.
**Model Initialization:** It initializes a HuggingFaceEndpoint object for text generation and creates a ChatHuggingFace object for chatbot interaction.
**Streamlit Interface:** Users can upload a CSV file containing data to be used by the chatbot.
**Query Processing:** Upon uploading the CSV file and entering a query, users can execute the query, and the chatbot generates a response based on the provided data.
**Output Display:** The response from the chatbot is displayed in the Streamlit interface.

**Getting Started**

Clone this repository to your local machine.
Install the required dependencies listed in the requirements.txt file.
Run the Streamlit application using the command streamlit run app.py.
Upload a CSV file containing your data and start interacting with the chatbot.

**Dependencies**

Python 3.x
Streamlit
PyTorch
Pandas
HuggingFace Transformers
Matplotlib

**Usage**

Upload a CSV file containing your data.
Enter a query in the text area provided.
Click the "Run Query" button to execute the query and interact with the chatbot.

**Contributing**

Contributions are welcome! Feel free to open an issue or submit a pull request for any improvements or new features.

**License**

This project is licensed under the MIT License. The code initializes necessary libraries and loads environment variables from a .env file.
