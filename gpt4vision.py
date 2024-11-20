import base64
import openai

class GPT4VisionModel:
    def __init__(self, api_key):
        self.openai_client = openai.OpenAI(api_key=api_key)
        self.model = "gpt-4-vision-preview"
        self.prompt = ("You are an expert in extracting Malayalam texts. "
                       "Given an image of a handwritten Malayalam text, "
                       "accurately identify and extract the text. "
                       "Return only the extracted Malayalam text.")
    
    def encode_image(self, image_path):
        '''
        Encoding images to base64.
        '''
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    


    def analyze_image(self, image_path):
        '''
        Analyze image and predict text.
        '''
        base64_image = self.encode_image(image_path)
        image_data_uri = f"data:image/jpeg;base64,{base64_image}"

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": self.prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_data_uri
                        }
                    }
                ]
            }
        ]

        response = self.openai_client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=300
        )

        return [resp.message.content for resp in response.choices]
    

IMAGE_PATH = 'test_data\\sen_5.jpg'
model = GPT4VisionModel(api_key="")
text = model.analyze_image(IMAGE_PATH)
# print(text)

predicted_text = ''.join(text)
with open('predicted_text_gpt.txt', 'w', encoding='utf-8') as f:
    f.write(predicted_text)

