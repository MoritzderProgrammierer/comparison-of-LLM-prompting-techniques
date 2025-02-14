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


class ZeroShotTechnique(PromptStrategy):
    def translate_to_german(self, model, text_to_translate: str) -> str:
        prompt = f"""Translate this text into German. Only return the translation:
        \n{text_to_translate}"""
        logger.debug("ZeroShotStrategy translate_to_german: prompt=%r", prompt)
        response = model(prompt, max_tokens=250)
        # We assume the model returns a dict with response["choices"][0]["text"]
        translation = response["choices"][0]["text"].strip()
        return translation

    def translate_to_english(self, model, text_to_translate: str) -> str:
        prompt = f"""Translate this text into English. Only return the translation:
        \n{text_to_translate}"""
        logger.debug("ZeroShotStrategy translate_to_english: prompt=%r", prompt)
        response = model(prompt, max_tokens=250)
        translation = response["choices"][0]["text"].strip()
        return translation


class ZeroShotTechniqueJSONOutput(PromptStrategy):
    def translate_to_german(self, model, text_to_translate: str) -> str:
        prompt = f"""Translate this text into German. Output only in this JSON format:
        ("translation": "your_translation_here")\n{text_to_translate}"""
        logger.debug("ZeroShotStrategy translate_to_german: prompt=%r", prompt)
        response = model(prompt, max_tokens=250)
        # We assume the model returns a dict with response["choices"][0]["text"]
        translation = response["choices"][0]["text"].strip()
        return translation

    def translate_to_english(self, model, text_to_translate: str) -> str:
        prompt = f"""Translate this text into English. Output only in this JSON format:
        ("translation": "your_translation_here")\n{text_to_translate}"""
        logger.debug("ZeroShotStrategy translate_to_english: prompt=%r", prompt)
        response = model(prompt, max_tokens=250)
        translation = response["choices"][0]["text"].strip()
        return translation


class ZeroShotTechniqueHighlightOnly(PromptStrategy):
    def translate_to_german(self, model, text_to_translate: str) -> str:
        prompt = f"""Translate this text into German. *Only* return the translation:
        \n{text_to_translate}"""
        logger.debug("ZeroShotStrategy translate_to_german: prompt=%r", prompt)
        response = model(prompt, max_tokens=250)
        # We assume the model returns a dict with response["choices"][0]["text"]
        translation = response["choices"][0]["text"].strip()
        return translation

    def translate_to_english(self, model, text_to_translate: str) -> str:
        prompt = f"""Translate this text into English. *Only* return the translation:
        \n{text_to_translate}"""
        logger.debug("ZeroShotStrategy translate_to_english: prompt=%r", prompt)
        response = model(prompt, max_tokens=250)
        translation = response["choices"][0]["text"].strip()
        return translation


class ZeroShotTechniqueTranslationTextFirst(PromptStrategy):
    def translate_to_german(self, model, text_to_translate: str) -> str:
        prompt = f"""{text_to_translate}\n
        Translate this text into German. Only return the translation."""
        logger.debug("ZeroShotStrategy translate_to_german: prompt=%r", prompt)
        response = model(prompt, max_tokens=250)
        # We assume the model returns a dict with response["choices"][0]["text"]
        translation = response["choices"][0]["text"].strip()
        return translation

    def translate_to_english(self, model, text_to_translate: str) -> str:
        prompt = f"""{text_to_translate}\n
        Translate this text into English. Only return the translation."""
        logger.debug("ZeroShotStrategy translate_to_english: prompt=%r", prompt)
        response = model(prompt, max_tokens=250)
        translation = response["choices"][0]["text"].strip()
        return translation


class ZeroShotTechniqueConcretePromptEnding(PromptStrategy):
    def translate_to_german(self, model, text_to_translate: str) -> str:
        prompt = f"""Translate this text into German. Only return the translation:
        \n{text_to_translate}<|endofprompt|>."""
        logger.debug("ZeroShotStrategy translate_to_german: prompt=%r", prompt)
        response = model(prompt, max_tokens=250)
        # We assume the model returns a dict with response["choices"][0]["text"]
        translation = response["choices"][0]["text"].strip()
        return translation

    def translate_to_english(self, model, text_to_translate: str) -> str:
        prompt = f"""Translate this text into English. Only return the translation:
        \n{text_to_translate}<|endofprompt|>."""
        logger.debug("ZeroShotStrategy translate_to_english: prompt=%r", prompt)
        response = model(prompt, max_tokens=250)
        translation = response["choices"][0]["text"].strip()
        return translation


class ZeroShotTechniqueAllTogether(PromptStrategy):
    def translate_to_german(self, model, text_to_translate: str) -> str:
        prompt = f"""{text_to_translate}\n
        Translate this text into German. Output *only* in this JSON format:
        ("translation": "your_translation_here")<|endofprompt|>."""
        logger.debug("ZeroShotStrategy translate_to_german: prompt=%r", prompt)
        response = model(prompt, max_tokens=250)
        # We assume the model returns a dict with response["choices"][0]["text"]
        translation = response["choices"][0]["text"].strip()
        return translation

    def translate_to_english(self, model, text_to_translate: str) -> str:
        prompt = f"""{text_to_translate}\n
        Translate this text into English. Output *only* in this JSON format:
        ("translation": "your_translation_here")<|endofprompt|>."""
        logger.debug("ZeroShotStrategy translate_to_english: prompt=%r", prompt)
        response = model(prompt, max_tokens=250)
        translation = response["choices"][0]["text"].strip()
        return translation


class InstructionTechnique(PromptStrategy):
    def translate_to_german(self, model, text_to_translate: str) -> str:
        prompt = f"{text_to_translate}\n Translate the text into German. *Only* write the necessary words. Be as precise as possible. *Only* return the translation.<|endofprompt|>"
        logger.debug("InstructionStrategy translate_to_german: prompt=%r", prompt)
        response = model(prompt, max_tokens=250)
        translation = response["choices"][0]["text"].strip()
        return translation

    def translate_to_english(self, model, text_to_translate: str) -> str:
        prompt = f"{text_to_translate}\n Translate the text into English. *Only* write the necessary words. Be as precise as possible. *Only* return the translation.<|endofprompt|>"
        logger.debug("InstructionStrategy translate_to_english: prompt=%r", prompt)
        response = model(prompt, max_tokens=250)
        translation = response["choices"][0]["text"].strip()
        return translation


class PersonaTechnique(PromptStrategy):
    def translate_to_german(self, model, text_to_translate: str) -> str:
        prompt = f"""{text_to_translate}\n
        Role: You are an expert in translation. You studied english for 45 years. 
        You are born in germany and live in germany. Your parents raised you bilingual. 
        Translate the text into German. *Only* return the translation.<|endofprompt|>"""
        logger.debug("PersonaStrategy translate_to_german: prompt=%r", prompt)
        response = model(prompt, max_tokens=250)
        translation = response["choices"][0]["text"].strip()
        return translation

    def translate_to_english(self, model, text_to_translate: str) -> str:
        prompt = f"""{text_to_translate}\n
        Role: You are an expert in translation. You studied english for 45 years. 
        You are born in germany and live in germany. Your parents raised you bilingual. 
        Translate the text into English. *Only* return the translation.<|endofprompt|>"""
        logger.debug("PersonaStrategy translate_to_english: prompt=%r", prompt)
        response = model(prompt, max_tokens=250)
        translation = response["choices"][0]["text"].strip()
        return translation


class FewShotTechniqueWithExampleToken(PromptStrategy):
    def translate_to_german(self, model, text_to_translate: str) -> str:
        prompt = f"""Translate the following text into German. Here are two examples:
        <example1>The Republicans secured a majority in the election. =  Die Republikaner sicherten sich bei der Wahl eine Mehrheit.<endExample1>
        <example2>During the week, I always sleep long. = Unter der Woche schlafe ich immer lange.<endExample2>
        Only return the translation:\n{text_to_translate}"""
        logger.debug("FewShotStrategy translate_to_german: prompt=%r", prompt)
        response = model(prompt, max_tokens=250)
        translation = response["choices"][0]["text"].strip()
        return translation

    def translate_to_english(self, model, text_to_translate: str) -> str:
        prompt = f"""Translate the following text into English. Here are two examples: 
        <example1>Die Republikaner sicherten sich bei der Wahl eine Mehrheit. =  The Republicans secured a majority in the election.<endExample1>
        <example2>Unter der Woche schlafe ich immer lange. = During the week, I always sleep long.<endExample2> Only return the translation:\n{text_to_translate}"""
        logger.debug("FewShotStrategy translate_to_english: prompt=%r", prompt)
        response = model(prompt, max_tokens=250)
        translation = response["choices"][0]["text"].strip()
        return translation


class FewShotTechniqueTranslationTextFirst(PromptStrategy):
    def translate_to_german(self, model, text_to_translate: str) -> str:
        prompt = f"""{text_to_translate}\n
        Translate the text into German. Here are two examples:
        The Republicans secured a majority in the election. =  Die Republikaner sicherten sich bei der Wahl eine Mehrheit.
        During the week, I always sleep long. = Unter der Woche schlafe ich immer lange.
        Only return the translation."""
        logger.debug("FewShotStrategy translate_to_german: prompt=%r", prompt)
        response = model(prompt, max_tokens=250)
        translation = response["choices"][0]["text"].strip()
        return translation

    def translate_to_english(self, model, text_to_translate: str) -> str:
        prompt = f"""{text_to_translate}\n
        Translate the text into English. Here are two examples:
        The Republicans secured a majority in the election. =  Die Republikaner sicherten sich bei der Wahl eine Mehrheit.
        During the week, I always sleep long. = Unter der Woche schlafe ich immer lange.
        Only return the translation."""
        logger.debug("FewShotStrategy translate_to_english: prompt=%r", prompt)
        response = model(prompt, max_tokens=250)
        translation = response["choices"][0]["text"].strip()
        return translation


class FewShotTechnique(PromptStrategy):
    def translate_to_german(self, model, text_to_translate: str) -> str:
        prompt = f"""Translate the following text into German. Here are two examples:
        The Republicans secured a majority in the election. =  Die Republikaner sicherten sich bei der Wahl eine Mehrheit.
        During the week, I always sleep long. = Unter der Woche schlafe ich immer lange.
        Only return the translation:\n{text_to_translate}"""
        logger.debug("FewShotStrategy translate_to_german: prompt=%r", prompt)
        response = model(prompt, max_tokens=250)
        translation = response["choices"][0]["text"].strip()
        return translation

    def translate_to_english(self, model, text_to_translate: str) -> str:
        prompt = f"""Translate the following text into English. Here are two examples: 
        Die Republikaner sicherten sich bei der Wahl eine Mehrheit. =  The Republicans secured a majority in the election.
        Unter der Woche schlafe ich immer lange. = During the week, I always sleep long.
        Only return the translation:\n{text_to_translate}"""
        logger.debug("FewShotStrategy translate_to_english: prompt=%r", prompt)
        response = model(prompt, max_tokens=250)
        translation = response["choices"][0]["text"].strip()
        return translation


class FewShotTechniqueContinueAfterExamples(PromptStrategy):
    def translate_to_german(self, model, text_to_translate: str) -> str:
        prompt = f"""Translate the following text into German. Here are two examples:
        English: "Good night." => German: "Gute Nacht."
        English: "See you later." => German: "Auf Wiedersehen."
        English: "{text_to_translate}" => German:"""
        logger.debug("FewShotStrategy translate_to_german: prompt=%r", prompt)
        response = model(prompt, max_tokens=250)
        translation = response["choices"][0]["text"].strip()
        return translation

    def translate_to_english(self, model, text_to_translate: str) -> str:
        prompt = f"""Translate the following text into English. Here are two examples: 
        German: "Gute Nacht." => English: "Good night."
        German: "Auf Wiedersehen." => English: "See you later."
        German: "{text_to_translate}" => English:"""
        logger.debug("FewShotStrategy translate_to_english: prompt=%r", prompt)
        response = model(prompt, max_tokens=250)
        translation = response["choices"][0]["text"].strip()
        return translation


class AllTogetherTechnique(PromptStrategy):
    def translate_to_german(self, model, text_to_translate: str) -> str:
        prompt = f"""{text_to_translate}\n
        Role: You are an expert in translation. You studied english for 45 years. 
        You are born in germany and live in germany. Your parents raised you bilingual. 
        English: "Good night." => German: "Gute Nacht."
        English: "See you later." => German: "Auf Wiedersehen."
        Translate the text into German. *Only* return the translation.<|endofprompt|>"""
        logger.debug("AllTogetherStrategy translate_to_german: prompt=%r", prompt)
        response = model(prompt, max_tokens=250)
        translation = response["choices"][0]["text"].strip()
        return translation

    def translate_to_english(self, model, text_to_translate: str) -> str:
        prompt = f"""{text_to_translate}\n
        Role: You are an expert in translation. You studied english for 45 years. 
        You are born in germany and live in germany. Your parents raised you bilingual. 
        German: "Gute Nacht." => English: "Good night."
        German: "Auf Wiedersehen." => English: "See you later."
        Translate the text into English. *Only* return the translation.<|endofprompt|>"""
        logger.debug("AllTogetherStrategy translate_to_english: prompt=%r", prompt)
        response = model(prompt, max_tokens=250)
        translation = response["choices"][0]["text"].strip()
        return translation


class SelfCorrectTechnique(PromptStrategy):
    def translate_to_german(self, model, text_to_translate: str) -> str:
        messages = [
            {"role": "user", "content": f"{text_to_translate}\nTranslate this text into German."}
        ]
        logger.debug("SelfCorrectTechnique to_german: initial_message=%r", messages[0])

        reasoning_response = model.create_chat_completion(messages, max_tokens=250)
        reasoning_text = reasoning_response["choices"][0]["message"]["content"].strip()

        messages.append({"role": "assistant", "content": reasoning_text})
        messages.append({"role": "user", "content": "Reflect on yourself and correct all mistakes of the previous translation. *Only* return the final translation:"})

        logger.debug("SelfCorrectTechnique to_german: messages=%r", messages)
        final_response = model.create_chat_completion(messages, max_tokens=250)
        translation = final_response["choices"][0]["message"]["content"].strip()
        return translation

    def translate_to_english(self, model, text_to_translate: str) -> str:
        messages = [
            {"role": "user", "content": f"{text_to_translate}\nTranslate this text into English."}
        ]
        logger.debug("SelfCorrectTechnique to_english: initial_message=%r", messages[0])

        reasoning_response = model.create_chat_completion(messages, max_tokens=250)
        reasoning_text = reasoning_response["choices"][0]["message"]["content"].strip()

        messages.append({"role": "assistant", "content": reasoning_text})
        messages.append({"role": "user", "content": "Reflect on yourself and correct all mistakes of the previous translation. *Only* return the final translation:"})

        logger.debug("SelfCorrectTechnique to_english: messages=%r", messages)
        final_response = model.create_chat_completion(messages, max_tokens=250)
        translation = final_response["choices"][0]["message"]["content"].strip()
        return translation


class ComparisonTechnique(PromptStrategy):
    def translate_to_german(self, model, text_to_translate: str) -> str:
        messages = [
            {"role": "user", "content": f"{text_to_translate}\nProvide *two* distinct german translations of this text."}
        ]
        logger.debug("SelfCorrectTechnique to_german: initial_message=%r", messages[0])

        reasoning_response = model.create_chat_completion(messages, max_tokens=400)
        reasoning_text = reasoning_response["choices"][0]["message"]["content"].strip()

        messages.append({"role": "assistant", "content": reasoning_text})
        messages.append({"role": "user", "content": "Now determine the best translation of the original text. *Only* return the best translation:"})

        logger.debug("SelfCorrectTechnique to_german: messages=%r", messages)
        final_response = model.create_chat_completion(messages, max_tokens=250)
        translation = final_response["choices"][0]["message"]["content"].strip()
        return translation

    def translate_to_english(self, model, text_to_translate: str) -> str:
        messages = [
            {"role": "user", "content": f"{text_to_translate}\nProvide *two* distinct english translations of this text."}
        ]
        logger.debug("SelfCorrectTechnique to_german: initial_message=%r", messages[0])

        reasoning_response = model.create_chat_completion(messages, max_tokens=400)
        reasoning_text = reasoning_response["choices"][0]["message"]["content"].strip()

        messages.append({"role": "assistant", "content": reasoning_text})
        messages.append({"role": "user", "content": "Now determine the best translation of the original text. *Only* return the best translation:"})

        logger.debug("SelfCorrectTechnique to_german: messages=%r", messages)
        final_response = model.create_chat_completion(messages, max_tokens=250)
        translation = final_response["choices"][0]["message"]["content"].strip()
        return translation


