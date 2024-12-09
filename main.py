from langchain_ollama import OllamaLLM
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from textblob import TextBlob

template = """
Answer the question below.
Here is the conversation history: {context}
Question: {question}
Answer:
"""

model = OllamaLLM(model="llama3")
prompt = ChatPromptTemplate.from_template(template)
chain = LLMChain(prompt=prompt, llm=model)


def analyze_sentiment(text):
    """Analyze the sentiment of the text and return positive, negative, or neutral."""
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    if polarity > 0:
        return "positive"
    elif polarity < 0:
        return "negative"
    else:
        return "neutral"


def handle_conversation():
    context = ""
    print("Welcome to the AI ChatBot! Type 'exit' to quit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break

        # Analyze sentiment
        sentiment = analyze_sentiment(user_input)
        print(f"Sentiment: {sentiment}")

        # Incorporate sentiment into context
        context += f"\nUser ({sentiment} sentiment): {user_input}"

        # Get response from the LLM chain using invoke instead of run
        try:
            result = chain.invoke({"context": context, "question": user_input})
        except Exception as e:
            print(f"An error occurred: {e}")
            continue

        print("Bot:", result)

        # Update the context with the AI's response
        context += f"\nAI: {result}"


if __name__ == "__main__":
    handle_conversation()
