from openai import OpenAI
import os
from pydantic import BaseModel

# Initialize OpenAI client using the API key in the OPENAI_API_KEY environment variable
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ---- Basic Usage ----
# Create a simple response asking the model to tell a joke
basic_response = client.responses.create(
    model="gpt-4o-mini",  # any supported model
    input="tell me a joke",
)
print("Basic response:\n", basic_response.output[0].content[0].text)

# Retrieve the response using its ID to show that the API stores state
fetched = client.responses.retrieve(response_id=basic_response.id)
print("Fetched response:\n", fetched.output[0].content[0].text)

# ---- Continue Conversation ----
# Continue from the previous response by specifying previous_response_id
followup = client.responses.create(
    model="gpt-4o-mini",
    input="tell me another",
    previous_response_id=basic_response.id,
)
print("Follow up response:\n", followup.output[0].content[0].text)

# ---- Using Hosted Tools ----
# Example that enables the built-in web search tool
search_response = client.responses.create(
    model="gpt-4o",  # or another supported model
    input="What's the latest news about AI?",
    tools=[{"type": "web_search"}],
)
print("Web search result:\n", search_response.output[0].content[0].text)

# ---- Structured Outputs ----
# Define a Pydantic model for the expected output
class CalendarEvent(BaseModel):
    name: str
    date: str
    participants: list[str]

# Use the response_format parameter to enforce the schema
structured_response = client.responses.create(
    model="gpt-4o-2024-08-06",
    input="Schedule a meeting with Alice and Bob for next Friday at 2pm.",
    response_format=CalendarEvent,
)
print("\nStructured response:\n", structured_response.output[0].content[0].text)

# ---- Vision (Image Input) ----
# Pass an image URL along with text in the input
vision_response = client.responses.create(
    model="gpt-4o",
    input=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What is in this image?"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
                    },
                },
            ],
        }
    ],
)
print("\nVision response:\n", vision_response.output[0].content[0].text)
