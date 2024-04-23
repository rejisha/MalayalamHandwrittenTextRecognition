import base64
from PIL import Image
import openai
import pickle

openai_client = openai.OpenAI(api_key='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')

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def analyze_images(image_path, openai_client):
    # base64_images = [encode_image(image_path) for image_path in image_paths]
    base64_image = encode_image(image_path)

    prompt = "You are an expert in extracting Malayalam texts.Given an image of a document with handwritten Malayalam text, accurately identify and extract the text.Return only the extracted malayalam text."

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": prompt
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{','.join(base64_image)}",
                    },
                },
            ],
        }
    ]

    response = openai_client.chat.completions.create(
        model="gpt-4-vision-preview",
        messages=messages,
        max_tokens=300,
    )

    return [resp.message.content for resp in response.choices]

def calculate_accuracy(predicted_paragraph, label):
    # Split the predicted paragraph into words
    predicted_words = predicted_paragraph[0].split()
    print(predicted_words)

    # Convert both predicted words and label to sets for faster lookup
    predicted_set = set(predicted_words)
    label_set = set(label)

    # Find the intersection of predicted words and label words
    common_words = predicted_set.intersection(label_set)

    # Calculate accuracy
    accuracy = len(common_words) / len(label)
    return accuracy


image_path = 'test_data/sentence_1.jpg'
malayalam = analyze_images(image_path, openai_client)
malayalam


# # Given predicted paragraph (as a single string) and label
predicted_paragraph = 'അമ്മ മലയിൽ പോയി'
label =['അമ്മ', 'മലയിൽ', 'പോയി']
# Calculate accuracy
accuracy = calculate_accuracy(malayalam, label)
print("Accuracy:", accuracy)