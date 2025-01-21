import logging
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from bert_score import score as bert_score_lib

logger = logging.getLogger(__name__)

def calculate_bleu_nltk(reference_text: str, translated_text: str) -> float:
    """
    Computes BLEU score using NLTK.
    Returns 0.0 if reference or hypothesis is empty.
    """
    if not reference_text.strip() or not translated_text.strip():
        logger.warning("BLEU calculation: reference or hypothesis is empty, returning 0.0.")
        return 0.0

    references = [reference_text.split()]
    hypothesis = translated_text.split()
    chencherry = SmoothingFunction()
    bleu_val = sentence_bleu(
        references,
        hypothesis,
        smoothing_function=chencherry.method1
    )
    logger.debug("BLEU: reference=%r, hypothesis=%r, score=%f", reference_text, translated_text, bleu_val)
    return bleu_val


def _compute_rouge_scores(reference_text: str, translated_text: str):
    """
    Helper function that uses `rouge_scorer` to compute
    rouge1, rouge2, and rougeL together.
    Returns a dict with 'rouge1', 'rouge2', 'rougeL' F1-scores.
    """
    if not reference_text.strip() or not translated_text.strip():
        logger.warning("ROUGE calculation: reference or hypothesis is empty, returning 0.0.")
        return {
            "rouge1": 0.0,
            "rouge2": 0.0,
            "rougeL": 0.0
        }

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = scorer.score(reference_text, translated_text)
    return {
        "rouge1": scores["rouge1"].fmeasure,
        "rouge2": scores["rouge2"].fmeasure,
        "rougeL": scores["rougeL"].fmeasure
    }


def calculate_rouge1(reference_text: str, translated_text: str) -> float:
    val = _compute_rouge_scores(reference_text, translated_text)["rouge1"]
    logger.debug("ROUGE1: reference=%r, hypothesis=%r, score=%f", reference_text, translated_text, val)
    return val

def calculate_rouge2(reference_text: str, translated_text: str) -> float:
    val = _compute_rouge_scores(reference_text, translated_text)["rouge2"]
    logger.debug("ROUGE2: reference=%r, hypothesis=%r, score=%f", reference_text, translated_text, val)
    return val

def calculate_rougeL(reference_text: str, translated_text: str) -> float:
    val = _compute_rouge_scores(reference_text, translated_text)["rougeL"]
    logger.debug("ROUGEL: reference=%r, hypothesis=%r, score=%f", reference_text, translated_text, val)
    return val


def calculate_bert_score(reference_text: str, translated_text: str,
                         model_type="bert-base-multilingual-cased") -> float:
    """
    Computes BERTScore using the `bert-score` library.
    By default, uses a multilingual model for possible non-English text.
    Returns the F1 score.
    """
    if not reference_text.strip() or not translated_text.strip():
        logger.warning("BERTScore calculation: reference or hypothesis is empty, returning 0.0.")
        return 0.0

    P, R, F1 = bert_score_lib(
        cands=[translated_text],
        refs=[reference_text],
        model_type=model_type
    )
    score_f1 = float(F1[0])
    logger.debug("BERTScore: reference=%r, hypothesis=%r, F1=%f", reference_text, translated_text, score_f1)
    return score_f1
