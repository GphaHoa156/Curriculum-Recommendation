
def single_f2_components(topic, predicted_content_ids):
    """
        Function to evaluate f2 for a single topic_id with it predictions
        Args:
            topic - Topic object
            predicted_content_ids - list predicted content ids from model predictions
    """
    truth_content_ids = [content.id for content in topic.content]
    truth_set = set(truth_content_ids)
    predict_set = set(predicted_content_ids)

    tp = len(truth_set & predict_set)
    fp = len(predict_set - truth_set)
    fn = len(truth_set - predict_set)

    return tp, fp, fn
    
def macro_f2(topic_ids, retrieval_results):
    """
        Calculate the macro f2 using all the results from model predictions
        and input topic_ids
    """
    predicted_ids = [
    [content_id for content_id, _ in content_list]
    for content_list in retrieval_results.values()
    ]
    f2_scores = []
    list_tp, list_fp, list_fn = [],[],[]
    for topic_id, pred_ids in tqdm(zip(topic_ids, predicted_ids), 
                                   total = len(topic_ids), 
                                   desc = "Macro F2"):
        topic = Topic(topic_id)  # Khởi tạo đối tượng Topic
        tp, fp, fn = single_f2_components(topic, pred_ids)
        try:
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            f2 = 5 * precision * recall / (4 * precision + recall)
        except ZeroDivisionError:
            f2 = 0
        f2_scores.append(f2)
        list_tp.append(tp)
        list_fp.append(fp)
        list_fn.append(fn)
    tp, fp, fn = np.array(list_tp), np.array(list_fp), np.array(list_fn)
    precision = tp/(tp + fp)
    recall =  tp/(tp + fn)
    print("Macro F2:", np.mean(f2_scores))
    print("Recall: ", np.mean(recall))
    print("Precision: ", np.mean(precision))
    
    return tp, fp, fn, np.array(f2_scores)