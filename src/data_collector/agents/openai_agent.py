from openai import OpenAI
from data_collector.agents.ai_agent import Agent

class OpenAIAgent(Agent):

    def __init__(self, name, api_key):
        super().__init__(name=name)
        self.name = name
        self.client = OpenAI(api_key=api_key)

    def get_response(self, system_prompt, user_prompt, temperature=1, max_tokens=1024):
        params = {
            "model": self.name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": temperature,
        }
        # Adjust max_tokens based on model name
        if self.name == "o3-mini":
            params["max_completion_tokens"] = max(2048, max_tokens)
        else:
            params["max_tokens"] = max_tokens
            params["top_p"] = 0.6

        completion = self.client.chat.completions.create(**params)
        return completion.choices[0].message.content
