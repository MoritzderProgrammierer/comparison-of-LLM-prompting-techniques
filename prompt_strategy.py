import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class PromptStrategy(ABC):
    @abstractmethod
    def translate_to_german(self, model, text_to_translate: str) -> str:
        pass

    @abstractmethod
    def translate_to_english(self, model, text_to_translate: str) -> str:
        pass


class ZeroShotStrategy(PromptStrategy):
    def translate_to_german(self, model, text_to_translate: str) -> str:
        prompt = f"Translate the following text into German. Only return the translation:\n{text_to_translate}"
        logger.debug("ZeroShotStrategy translate_to_german: prompt=%r", prompt)
        response = model(prompt, max_tokens=200)
        # We assume the model returns a dict with response["choices"][0]["text"]
        translation = response["choices"][0]["text"].strip()
        return translation

    def translate_to_english(self, model, text_to_translate: str) -> str:
        prompt = f"Translate the following text into English. Only return the translation:\n{text_to_translate}"
        logger.debug("ZeroShotStrategy translate_to_english: prompt=%r", prompt)
        response = model(prompt, max_tokens=200)
        translation = response["choices"][0]["text"].strip()
        return translation


class InstructionStrategy(PromptStrategy):
    def translate_to_german(self, model, text_to_translate: str) -> str:
        prompt = f"Translate the following text into German. Use natural language and only write the necessary words. Only return the translation:\n{text_to_translate}"
        logger.debug("InstructionStrategy translate_to_german: prompt=%r", prompt)
        response = model(prompt, max_tokens=200)
        # We assume the model returns a dict with response["choices"][0]["text"]
        translation = response["choices"][0]["text"].strip()
        return translation

    def translate_to_english(self, model, text_to_translate: str) -> str:
        prompt = f"Translate the following text into English. Use natural language and only write the necessary words. Only return the translation:\n{text_to_translate}"
        logger.debug("InstructionStrategy translate_to_english: prompt=%r", prompt)
        response = model(prompt, max_tokens=200)
        translation = response["choices"][0]["text"].strip()
        return translation


class PersonaStrategy(PromptStrategy):
    def translate_to_german(self, model, text_to_translate: str) -> str:
        prompt = f"Role: You are an expert in translation. You studied english for 45 years. You are born in germany and live in germany. 
        Your parents raised you bilingual. Translate the following text into German. Only return the translation:\n{text_to_translate}"
        logger.debug("PersonaStrategy translate_to_german: prompt=%r", prompt)
        response = model(prompt, max_tokens=250)
        print("\nPrompt: ", prompt)
        print("\nResponse: ", response)
        # We assume the model returns a dict with response["choices"][0]["text"]
        translation = response["choices"][0]["text"].strip()
        return translation

    def translate_to_english(self, model, text_to_translate: str) -> str:
        prompt = f"Role: You are an expert in translation. You studied english for 45 years. You are born in germany and live in germany. 
        Your parents raised you bilingual. Translate the following text into English. Only return the translation:\n{text_to_translate}"
        logger.debug("PersonaStrategy translate_to_english: prompt=%r", prompt)
        response = model(prompt, max_tokens=250)
        print("\nPrompt: ", prompt)
        print("\nResponse: ", response)
        translation = response["choices"][0]["text"].strip()
        return translation


class FewShotStrategy(PromptStrategy):
    def translate_to_german(self, model, text_to_translate: str) -> str:
        prompt = f"Translate the following text into English. Here are two examples: 
        <example1>The Republicans secured a majority in the election. =  Die Republikaner sicherten sich bei der Wahl eine Mehrheit.<endExample1>
        <example2>During the week, I always go to bed early. = Unter der Woche gehe ich immer fr�h schlafen.<endExample2>
        Only return the translation:\n{text_to_translate}"
        logger.debug("FewShotStrategy translate_to_german: prompt=%r", prompt)
        response = model(prompt, max_tokens=250)
        print("\nPrompt: ", prompt)
        print("\nResponse: ", response)
        # We assume the model returns a dict with response["choices"][0]["text"]
        translation = response["choices"][0]["text"].strip()
        return translation

    def translate_to_english(self, model, text_to_translate: str) -> str:
        prompt = f"Translate the following text into English. Here are two examples: 
        <example1>Die Republikaner sicherten sich bei der Wahl eine Mehrheit. =  The Republicans secured a majority in the election.<endExample1>
        <example2>Unter der Woche gehe ich immer fr�h schlafen. = During the week, I always go to bed early.<endExample2>
        Only return the translation:\n{text_to_translate}"
        logger.debug("FewShotStrategy translate_to_english: prompt=%r", prompt)
        response = model(prompt, max_tokens=250)
        print("\nPrompt: ", prompt)
        print("\nResponse: ", response)
        translation = response["choices"][0]["text"].strip()
        return translation


class ChainOfThoughtStrategy(PromptStrategy):
    def translate_to_german(self, model, text_to_translate: str) -> str:
        # Variation that tries to reason in chain-of-thought
        reasoning_prompt = (
            f"Translate the following text to German and show reasoning step by step:\n{text_to_translate}\n"
            "Let's break down the meaning first..."
        )
        logger.debug("ChainOfThoughtStrategy to_german: reasoning_prompt=%r", reasoning_prompt)
        reasoning_response = model(reasoning_prompt, max_tokens=200)
        reasoning_text = reasoning_response["choices"][0]["text"].strip()

        final_prompt = f"{reasoning_text}\nNow provide the final German translation only:"
        logger.debug("ChainOfThoughtStrategy to_german: final_prompt=%r", final_prompt)
        final_response = model(final_prompt, max_tokens=200)
        translation = final_response["choices"][0]["text"].strip()
        return translation

    def translate_to_english(self, model, text_to_translate: str) -> str:
        reasoning_prompt = (
            f"Translate the following text to English and show reasoning step by step:\n{text_to_translate}\n"
            "Let's break down the meaning first..."
        )
        logger.debug("ChainOfThoughtStrategy to_english: reasoning_prompt=%r", reasoning_prompt)
        reasoning_response = model(reasoning_prompt, max_tokens=200)
        reasoning_text = reasoning_response["choices"][0]["text"].strip()

        final_prompt = f"{reasoning_text}\nNow provide the final English translation only:"
        logger.debug("ChainOfThoughtStrategy to_english: final_prompt=%r", final_prompt)
        final_response = model(final_prompt, max_tokens=200)
        translation = final_response["choices"][0]["text"].strip()
        return translation


