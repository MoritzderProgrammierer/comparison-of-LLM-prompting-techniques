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
        response = model(prompt, max_tokens=250)
        # We assume the model returns a dict with response["choices"][0]["text"]
        translation = response["choices"][0]["text"].strip()
        return translation

    def translate_to_english(self, model, text_to_translate: str) -> str:
        prompt = f"Translate the following text into English. Only return the translation:\n{text_to_translate}"
        logger.debug("ZeroShotStrategy translate_to_english: prompt=%r", prompt)
        response = model(prompt, max_tokens=250)
        translation = response["choices"][0]["text"].strip()
        return translation


class InstructionStrategy(PromptStrategy):
    def translate_to_german(self, model, text_to_translate: str) -> str:
        prompt = f"Translate the following text into German. Use natural language and only write the necessary words. Only return the translation:\n{text_to_translate}"
        logger.debug("InstructionStrategy translate_to_german: prompt=%r", prompt)
        response = model(prompt, max_tokens=250)
        translation = response["choices"][0]["text"].strip()
        return translation

    def translate_to_english(self, model, text_to_translate: str) -> str:
        prompt = f"Translate the following text into English. Use natural language and only write the necessary words. Only return the translation:\n{text_to_translate}"
        logger.debug("InstructionStrategy translate_to_english: prompt=%r", prompt)
        response = model(prompt, max_tokens=250)
        translation = response["choices"][0]["text"].strip()
        return translation


class PersonaStrategy(PromptStrategy):
    def translate_to_german(self, model, text_to_translate: str) -> str:
        prompt = f"Role: You are an expert in translation. You studied english for 45 years. You are born in germany and live in germany. Your parents raised you bilingual. Translate the following text into German. Only return the translation:\n{text_to_translate}"
        logger.debug("PersonaStrategy translate_to_german: prompt=%r", prompt)
        response = model(prompt, max_tokens=250)
        translation = response["choices"][0]["text"].strip()
        return translation

    def translate_to_english(self, model, text_to_translate: str) -> str:
        prompt = f"Role: You are an expert in translation. You studied english for 45 years. You are born in germany and live in germany. Your parents raised you bilingual. Translate the following text into English. Only return the translation:\n{text_to_translate}"
        logger.debug("PersonaStrategy translate_to_english: prompt=%r", prompt)
        response = model(prompt, max_tokens=250)
        translation = response["choices"][0]["text"].strip()
        return translation


class FewShotStrategy(PromptStrategy):
    def translate_to_german(self, model, text_to_translate: str) -> str:
        prompt = f"Translate the following text into German. Here are two examples: <example1>The Republicans secured a majority in the election. =  Die Republikaner sicherten sich bei der Wahl eine Mehrheit.<endExample1> <example2>During the week, I always sleep long. = Unter der Woche schlafe ich immer lange.<endExample2> Only return the translation:\n{text_to_translate}"
        logger.debug("FewShotStrategy translate_to_german: prompt=%r", prompt)
        response = model(prompt, max_tokens=250)
        translation = response["choices"][0]["text"].strip()
        return translation

    def translate_to_english(self, model, text_to_translate: str) -> str:
        prompt = f"Translate the following text into English. Here are two examples: <example1>Die Republikaner sicherten sich bei der Wahl eine Mehrheit. =  The Republicans secured a majority in the election.<endExample1><example2>Unter der Woche schlafe ich immer lange. = During the week, I always sleep long.<endExample2> Only return the translation:\n{text_to_translate}"
        logger.debug("FewShotStrategy translate_to_english: prompt=%r", prompt)
        response = model(prompt, max_tokens=250)
        translation = response["choices"][0]["text"].strip()
        return translation


class AllTogetherStrategy(PromptStrategy):
    def translate_to_german(self, model, text_to_translate: str) -> str:
        prompt = f"Role: You are an expert in translation. You studied english for 45 years. You are born in germany and live in germany. Your parents raised you bilingual. Translate the following text into German. Here are two examples: <example1>The Republicans secured a majority in the election. =  Die Republikaner sicherten sich bei der Wahl eine Mehrheit.<endExample1> <example2>During the week, I always sleep long. = Unter der Woche schlafe ich immer lange.<endExample2> Use natural language and only write the necessary words. Only return the translation:\n{text_to_translate}"
        logger.debug("AllTogetherStrategy translate_to_german: prompt=%r", prompt)
        response = model(prompt, max_tokens=250)
        translation = response["choices"][0]["text"].strip()
        return translation

    def translate_to_english(self, model, text_to_translate: str) -> str:
        prompt = f"Role: You are an expert in translation. You studied english for 45 years. You are born in germany and live in germany. Your parents raised you bilingual. Translate the following text into English. Here are two examples: <example1>Die Republikaner sicherten sich bei der Wahl eine Mehrheit. =  The Republicans secured a majority in the election.<endExample1><example2>Unter der Woche schlafe ich immer lange. = During the week, I always sleep long.<endExample2> Use natural language and only write the necessary words. Only return the translation:\n{text_to_translate}"
        logger.debug("AllTogetherStrategy translate_to_english: prompt=%r", prompt)
        response = model(prompt, max_tokens=250)
        translation = response["choices"][0]["text"].strip()
        return translation


class SelfCorrectStrategy(PromptStrategy):
    def translate_to_german(self, model, text_to_translate: str) -> str:
        first_prompt = f"Translate the following text into German. Only return the translation:\n{text_to_translate}"
        logger.debug("SelfCorrectStrategy to_german: first_prompt=%r", first_prompt)
        reasoning_response = model(first_prompt, max_tokens=250)
        reasoning_text = reasoning_response["choices"][0]["text"].strip()

        final_prompt = f"Reflect on yourself and correct all mistakes so that the text is translated as best as possible.\n{reasoning_text}"
        logger.debug("SelfCorrectStrategy to_german: final_prompt=%r", final_prompt)
        final_response = model(final_prompt, max_tokens=250)
        translation = final_response["choices"][0]["text"].strip()
        return translation

    def translate_to_english(self, model, text_to_translate: str) -> str:
        first_prompt = f"Translate the following text to English. Only return the translation:\n{text_to_translate}\n"
        logger.debug("SelfCorrectStrategy to_english: first_prompt=%r", first_prompt)
        reasoning_response = model(first_prompt, max_tokens=250)
        reasoning_text = reasoning_response["choices"][0]["text"].strip()

        final_prompt = f"Reflect on yourself and correct all mistakes so that the text is translated as best as possible.\n{reasoning_text}"
        logger.debug("SelfCorrectStrategy to_english: final_prompt=%r", final_prompt)
        final_response = model(final_prompt, max_tokens=250)
        translation = final_response["choices"][0]["text"].strip()
        return translation


class ComparisonStrategy(PromptStrategy):
    def translate_to_german(self, model, text_to_translate: str) -> str:
        first_prompt = f"Translate the following text into German. Only return the translation:\n{text_to_translate}"
        logger.debug("comparisonStrategy to_german: first_prompt=%r", first_prompt)
        reasoning_response = model(first_prompt, max_tokens=200)
        first_text = reasoning_response["choices"][0]["text"].strip()

        second_prompt = f"Translate the following text into German. Only return the translation:\n{text_to_translate}"
        logger.debug("comparisonStrategy to_german: second_prompt=%r", second_prompt)
        reasoning_response = model(second_prompt, max_tokens=200)
        second_text = reasoning_response["choices"][0]["text"].strip()

        #third_prompt = f"Translate the following text into German. Only return the translation:\n{text_to_translate}"
        #logger.debug("comparisonStrategy to_german: third_prompt=%r", third_prompt)
        #reasoning_response = model(third_prompt, max_tokens=250)
        #third_text = reasoning_response["choices"][0]["text"].strip()

        final_prompt = f"Out of all the following translations, which one is the best? Only repeat the best one. No yes or no, only repeat the translated text. \n<first>{first_text}<first>\n<second>{second_text}<second>"#\n<third>{third_text}<third>"
        logger.debug("comparisonStrategy to_german: final_prompt=%r", final_prompt)
        final_response = model(final_prompt, max_tokens=250)
        translation = final_response["choices"][0]["text"].strip()
        return translation

    def translate_to_english(self, model, text_to_translate: str) -> str:
        first_prompt = f"Translate the following text to English. Only return the translation:\n{text_to_translate}\n"
        logger.debug("comparisonStrategy to_english: first_prompt=%r", first_prompt)
        reasoning_response = model(first_prompt, max_tokens=200)
        first_text = reasoning_response["choices"][0]["text"].strip()

        second_prompt = f"Translate the following text to English. Only return the translation:\n{text_to_translate}\n"
        logger.debug("comparisonStrategy to_english: second_prompt=%r", second_prompt)
        reasoning_response = model(second_prompt, max_tokens=200)
        second_text = reasoning_response["choices"][0]["text"].strip()

        #third_prompt = f"Translate the following text to English. Only return the translation:\n{text_to_translate}\n"
        #logger.debug("comparisonStrategy to_english: third_prompt=%r", third_prompt)
        #reasoning_response = model(third_prompt, max_tokens=250)
        #third_text = reasoning_response["choices"][0]["text"].strip()

        final_prompt = f"Out of all the following translations, which one is the best? Only repeat the best one. No yes or no, only repeat the translated text. \n<first>{first_text}<first>\n<second>{second_text}<second>"#\n<third>{third_text}<third>"
        logger.debug("comparisonStrategy to_english: final_prompt=%r", final_prompt)
        final_response = model(final_prompt, max_tokens=250)
        translation = final_response["choices"][0]["text"].strip()
        return translation