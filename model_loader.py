import logging
from llama_cpp import Llama

logger = logging.getLogger(__name__)

class ModelLoader:
    @staticmethod
    def load_llama_cpp_model(model_path, model_name):
        logger.info("Loading model '%s' from path: %s", model_name, model_path)
        try:
            model = Llama(
                model_path=model_path,
                verbose=False,
                temperature=0.1,
                top_k=40,
                top_p=0.9
            )
            logger.debug("Successfully loaded model '%s'", model_name)
            return {"name": model_name, "model": model}
        except Exception as e:
            logger.error("Error loading model '%s': %s", model_name, e)
            raise
