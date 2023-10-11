import openai

# Initialize the OpenAI API client
openai.api_key = "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"

def get_gpt4_response(prompt_text):
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt_text}
        ]
    )
    return response.choices[0].message.content

if __name__ == "__main__":
    while True:
        user_input = input("You: ")
        response = get_gpt4_response(user_input)
        print(f"ChatGPT: {response}")