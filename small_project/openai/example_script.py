import os
import openai

# Load your API key from an environment variable or secret management service
openai.api_key = os.getenv("OPENAI_API_KEY")

response = openai.Completion.create(model="text-davinci-003", prompt="Say this is a test", temperature=0, max_tokens=7)

response = openai.Completion.create(model="text-davinci-003", prompt="Make an excuses why I was late for the drinks when I have a lot of work too many deadlines",
                                    temperature=1.2, max_tokens=100)
print(response.choices)