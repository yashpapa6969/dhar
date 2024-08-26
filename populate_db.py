from flask import Flask, request, jsonify
import weaviate
import weaviate.classes as wvc
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()
# Access the environment variables correctly
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ['LANGCHAIN_TRACING_V2'] = os.getenv("LANGCHAIN_TRACING_V2")
os.environ['LANGCHAIN_ENDPOINT'] = os.getenv("LANGCHAIN_ENDPOINT")
os.environ['LANGCHAIN_API_KEY'] = os.getenv("LANGCHAIN_API_KEY")

# Optionally, retrieve additional environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")
wcd_api_key = os.getenv("WCD_API_KEY")
wcd_url = os.getenv("WCD_URL")
# Initialize your database client and collection name
client = weaviate.connect_to_weaviate_cloud(
    cluster_url=wcd_url,                                    # Replace with your Weaviate Cloud URL
    auth_credentials=wvc.init.Auth.api_key(wcd_api_key),    # Replace with your Weaviate Cloud key
    headers={"X-OpenAI-Api-Key": openai_api_key}            # Replace with appropriate header key/value pair for the required API
)
#auth_config = weaviate.AuthApiKey(api_key=wcd_api_key)
# client = weaviate.Client(
#     url=wcd_url,                                    # Replace with your Weaviate Cloud URL
#     auth_client_secret=auth_config,    # Replace with your Weaviate Cloud key
#     additional_headers={"X-OpenAI-Api-Key": openai_api_key}            # Replace with appropriate header key/value pair for the required API
# )

collection_name = "Bobmail"  # Replace with your actual collection name
class populate_db:
  def __init__(self,client,collection_name):
    # populate_db.create_schema()
    self.client = client
    self.collection_name = collection_name


  @staticmethod
  def create_schema():
    client.collections.delete("Bobmail")
    questions = client.collections.create(
    name="Bobmail",
    vectorizer_config=wvc.config.Configure.Vectorizer.text2vec_openai(model='text-embedding-3-small'),   # Set the vectorizer to "text2vec-openai" to use the OpenAI API for vector-related operations
    properties=[
        wvc.config.Property(
            name="Phone",
            data_type=wvc.config.DataType.TEXT,
            vectorize_property_name=False,  # Include the property name ("question") when vectorizing
            tokenization=wvc.config.Tokenization.LOWERCASE  # Use "lowecase" tokenization
        ),
        wvc.config.Property(
            name="Email",
            data_type=wvc.config.DataType.TEXT,
            vectorize_property_name=False,  # Include the property name ("question") when vectorizing
            tokenization=wvc.config.Tokenization.LOWERCASE  # Use "lowecase" tokenization
        ),
                wvc.config.Property(
            name="Date",
            data_type=wvc.config.DataType.TEXT,
            vectorize_property_name=False,  # Include the property name ("question") when vectorizing
            tokenization=wvc.config.Tokenization.LOWERCASE  # Use "lowecase" tokenization
        ),
        wvc.config.Property(
            name="Query",
            data_type=wvc.config.DataType.TEXT,
            vectorize_property_name=True,  # Include the property name ("question") when vectorizing
            tokenization=wvc.config.Tokenization.LOWERCASE  # Use "lowecase" tokenization
        ),
        wvc.config.Property(
            name="Response",
            data_type=wvc.config.DataType.TEXT,
            vectorize_property_name=True,  # Skip the property name ("answer") when vectorizing
            tokenization=wvc.config.Tokenization.LOWERCASE  # Use "whitespace" tokenization
        )
        ]
        )
    

  def insert_data(self,data):
    question_objs = list()
    question_objs.append({
            "Phone": data["Phone"],
            "Email": data["Email"],
            "Date": data["Date"],
            "Query": data["Query"],
            "Response": data["Helpdesk Response"]
        })
    questions = self.client.collections.get(self.collection_name)
    questions.data.insert_many(question_objs)
app = Flask(__name__)



# Create an instance of the populate_db class
db = populate_db(client, collection_name)
#populate_db.create_schema()

@app.route('/insert_data', methods=['POST'])
def insert_data():
    try:
        # Get JSON data from the request
        data = request.get_json()

        # Ensure data is not empty
        if not data:
            return jsonify({"error": "No data provided"}), 400

        # Call the insert_data function from the populate_db class
        db.insert_data(data)

        return jsonify({"message": "Data inserted successfully"}), 201

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)