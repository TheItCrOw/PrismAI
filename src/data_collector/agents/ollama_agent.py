import requests

from data_collector.agents.ai_agent import Agent

class OllamaAgent(Agent):

    def __init__(self, 
                 name, 
                 ollama_base_url,
                 context_length):
        super().__init__(name=name, context_length=context_length)
        self.name = name
        self.ollama_base_url = ollama_base_url

    def get_response(self, 
                     system_prompt, 
                     user_prompt, 
                     temperature=1, 
                     max_tokens=1024):
        if self.name.startswith('deepseek-r1'):
            max_tokens = max(2048, max_tokens)
            
        payload = {
            "model": self.name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "stream": False,
            "options": {
                "temperature": temperature,
                "seed": 42,
                "num_predict": max_tokens,
                "num_ctx": self.context_length
            }
        }

        response = requests.post(
            url=f'{self.ollama_base_url}/api/chat',
            json=payload
        )

        response.raise_for_status()
        completion = response.json()
        return completion["message"]["content"]
