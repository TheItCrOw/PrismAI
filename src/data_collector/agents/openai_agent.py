from openai import OpenAI
from data_collector.agents.ai_agent import Agent


class OpenAIAgent(Agent):
    def __init__(self, name, api_key):
        super().__init__(name=name)
        self.name = name
        self.client = OpenAI(api_key=api_key)

    def get_response(self, system_prompt, user_prompt, temperature=1, max_tokens=1024):
        completion = self.client.chat.completions.create(
            model=self.name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            top_p=0.6,
            max_tokens=max_tokens,
        )
        return completion.choices[0].message.content
