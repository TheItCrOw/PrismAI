import hashlib
import os
import random
import re
from abc import ABC, abstractmethod
from datetime import datetime

from data_collector.collected_item import CollectedItem
from data_collector.collector import Collector


class Agent(ABC):
    def __init__(self, name, context_length=2048):
        self.name = name
        self.information_extract_prompt = self.get_prompt("extract_information.md")
        self.ghostwriting_prompt = self.get_prompt("ghostwrite.md")
        self.context_length = context_length

    @abstractmethod
    def get_response(self, system_prompt, user_prompt, temperature=1, max_tokens=1024):
        pass

    def get_prompt(self, name):
        # Setup the prompt for the information extraction prompt
        prompt_path = os.path.join(os.path.dirname(__file__), "..", "prompts", name)
        if not os.path.exists(prompt_path):
            raise FileNotFoundError(f"Couldn't find prompt file under '{prompt_path}'.")

        with open(prompt_path, "r", encoding="utf-8") as file:
            content = file.read()
        return content

    def truncate_to_full_sentence(self, text):
        """
        Truncates the text to the last complete sentence.
        """
        sentences = re.findall(r"[^.!?]*[.!?]", text)
        if sentences:
            return "".join(sentences)
        else:
            return text

    def synthesize_fulltext(self, item: CollectedItem, collector: Collector):
        synth_obj = {
            "created": str(datetime.now()),
            "type": "fulltext",
            "model": self.name,
        }

        # So, for the fulltext we do the following:
        # 1. Find out what the text is about, topics, opinions and such via the agent
        # 2. Prompt the agent to write a text about these information we extracted
        #   2.1. Adhere to the original text lingo
        #   2.2. Adhere to the original text length
        text_length = min(5000, len(item.text.split()))
        # If the text is too small, it makes no sense really.
        if text_length < 10:
            return None

        synth_obj["extracted_information"] = self.get_response(
            system_prompt=self.information_extract_prompt,
            user_prompt="\n\n### Text:\n" + item.text[:5000],
        )
        if self.name.startswith("deepseek-r1"):
            synth_obj["extracted_information"] = synth_obj[
                "extracted_information"
            ].split("</think>")[1]
        synth_obj["extracted_information"] += (
            f"\n- The text must be around {min(2000, text_length)} words long."
        )

        multiplier = 1.6
        if self.name.startswith("deepseek-r1") or self.name == "o3-mini":
            # In this case, we have a reasoning model. Their response looks differently,
            # since they "<think></think>" before hand. We hence give them more tokens
            multiplier = 15
        response = self.get_response(
            system_prompt=self.ghostwriting_prompt,
            user_prompt="### Text Requirements:\n" + synth_obj["extracted_information"],
            max_tokens=(int)(text_length * multiplier),
        )
        if self.name.startswith("deepseek-r1"):
            response = re.sub(
                r"<think>.*?</think>", "", response, flags=re.DOTALL
            ).strip()

        synth_obj["synth_text"] = self.truncate_to_full_sentence(response)
        synth_obj["og_text_length"] = len(item.text.split())
        synth_obj["synth_text_text_length"] = len(synth_obj["synth_text"].split())
        synth_obj["synth_text_truncated"] = synth_obj["synth_text"] != response

        return synth_obj

    def synthesize_chunks(self, item: CollectedItem, collector: Collector):
        synth_obj = {
            "created": str(datetime.now()),
            "type": "chunk",
            "model": self.name,
        }

        # Generate a seed so that this item would get the same chunks again.
        unique_identifier = f"{item.text}{item.source}{item.date}"
        synth_obj["seed"] = int(
            hashlib.sha256(unique_identifier.encode()).hexdigest(), 16
        ) % (2**32)
        random.seed(synth_obj["seed"])

        # Determine the number of chunks to replace (min 1, max 50%)
        synth_obj["total_chunks"] = len(item.chunks)

        # It makes no sense if there are only 1,2 or 3 chunks.
        if synth_obj["total_chunks"] < 4:
            return None

        synth_obj["amount_chunks_to_replace"] = random.randint(
            1, max(1, synth_obj["total_chunks"] // 2)
        )
        synth_obj["chunks_replaced_percentage"] = round(
            100 / synth_obj["total_chunks"] * synth_obj["amount_chunks_to_replace"], 2
        )

        # Select a random starting index for the consecutive block
        synth_obj["start_idx"] = random.randint(
            0, synth_obj["total_chunks"] - synth_obj["amount_chunks_to_replace"]
        )
        synth_obj["end_idx"] = (
            synth_obj["start_idx"] + synth_obj["amount_chunks_to_replace"]
        )

        before_context = (
            item.chunks[: synth_obj["start_idx"] - 1]
            if synth_obj["start_idx"] > 0
            else ""
        )
        replaced_context = item.chunks[synth_obj["start_idx"] : synth_obj["end_idx"]]
        synth_obj["og_chunk_text_length"] = min(
            5000, len(" ".join(replaced_context).split(" "))
        )
        after_context = (
            item.chunks[synth_obj["end_idx"] + 1 :]
            if synth_obj["end_idx"] < synth_obj["total_chunks"] - 1
            else ""
        )

        system_prompt = collector.get_synthetization_system_prompt(
            {"date": item.date, "length": synth_obj["og_chunk_text_length"]}
        )
        # Sometimes, the before and after contexts are very large (e.g. in cases of paper)
        # Putting the whole thing into the prompt may be too much and costly, so we cutoff
        # some begin and end if needed for those large contexts.
        n_words = 2000
        before = " ".join("".join(before_context).split()[-n_words:])
        after = " ".join("".join(after_context).split()[:n_words])
        user_prompt = f"### START: {before.strip()}\n\n### END: {after.strip()}"

        # Generate the filling chunk gaps now with the AI response.
        multiplier = 1.6
        if self.name.startswith("deepseek-r1") or self.name == "o3-mini":
            # In this case, we have a reasoning model. Their response looks differently,
            # since they "<think></think>" before hand. We hence give them more tokens
            multiplier = 15
        response = self.get_response(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=(int)(synth_obj["og_chunk_text_length"] * multiplier),
        )
        if self.name.startswith("deepseek-r1"):
            response = re.sub(
                r"<think>.*?</think>", "", response, flags=re.DOTALL
            ).strip()

        # Since we want to force a certain length of the response and set max_tokens, the model
        # sometimes doesnt end on a full sentence. In that case, we cut that faulty sentence off.
        cleaned_response = self.truncate_to_full_sentence(response)

        synth_obj["synth_chunk_text_length"] = len(cleaned_response.split())
        synth_obj["synth_text"] = cleaned_response
        synth_obj["synth_text_truncated"] = cleaned_response != response
        return synth_obj

    def synthesize_collected_item(
        self, item: CollectedItem, collector: Collector, force=False
    ):
        if not force and any(
            getattr(synth_item, "agent", None) == self.name
            for synth_item in item.synthetization
        ):
            print(
                "Item already synthesized with agent; skipping it. Use force=True to synthesize again."
            )
            return None

        return {
            "synth_chunks": self.synthesize_chunks(item, collector),
            "synth_fulltext": self.synthesize_fulltext(item, collector),
        }
