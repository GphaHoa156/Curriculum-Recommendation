from sentence_transformers import util
def sem_score(sentence_1, sentence_2, model):
    """
        Function to evaluate the semantic score between 2 sentences.
        Args:
            sentence_1, sentence_2 (str): 2 input sentences
            model (sentence_transformer object): the sentence transformer use to encode sentences
    """
    emb1 = model.encode(sentence_1, 
                        convert_to_tensor = True,
                        normalize_embeddings = True,
                        show_progress_bar = False).to(model.device)
    emb2 = model.encode(sentence_2, 
                        convert_to_tensor = True,
                        normalize_embeddings = True,
                        show_progress_bar = False).to(model.device)
    score = util.cos_sim(emb1, emb2)
    return score.item()