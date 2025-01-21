import requests

from data_collector.agents.ai_agent import Agent

class OllamaAgent(Agent):

    def __init__(self, 
                 name, 
                 ollama_base_url):
        super().__init__(name=name)
        self.name = name
        self.ollama_base_url = ollama_base_url

    def get_response(self, 
                     system_prompt, 
                     user_prompt, 
                     temperature=1, 
                     max_tokens=1024):
        
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
                "num_predict": max_tokens
            }
        }

        response = requests.post(
            url=f'{self.ollama_base_url}/api/chat',
            json=payload
        )

        response.raise_for_status()
        completion = response.json()
        return completion["message"]["content"]
