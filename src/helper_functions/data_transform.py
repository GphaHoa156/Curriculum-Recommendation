import re
import numpy as np
import pandas as pd
from tqdm import tqdm
from IPython.display import display, Markdown
from sentence_transformers import util

def print_markdown(md):
    display(Markdown(md))

"""
    Create 2 class Topic and ContentItem where its attribute can quickly
    be accesed.
"""
class Topic:
    def __init__(self, topic_id):
        self.id = topic_id

    @property
    def parent(self):
        parent_id = topics_df.loc[self.id].parent
        if pd.isna(parent_id):
            return None
        else:
            return Topic(parent_id)

    @property
    def ancestors(self):
        ancestors = []
        parent = self.parent
        while parent is not None:
            ancestors.append(parent)
            parent = parent.parent
        return ancestors

    @property
    def siblings(self):
        if not self.parent:
            return []
        else:
            return [topic for topic in self.parent.children if topic != self]

    @property
    def content(self):
        if self.id in correlations_df.index:
            return [ContentItem(content_id) for content_id in correlations_df.loc[self.id].content_ids.split()]
        else:
            return tuple([]) if self.has_content else []

    def get_breadcrumbs(self, separator=" >> ", include_self=True, include_root=True):
        ancestors = self.ancestors
        if include_self:
            ancestors = [self] + ancestors
        if not include_root:
            ancestors = ancestors[:-1]
        return separator.join(reversed([a.title for a in ancestors]))

    @property
    def children(self):
        return [Topic(child_id) for child_id in topics_df[topics_df.parent == self.id].index]

    def subtree_markdown(self, depth=0):
        markdown = "  " * depth + "- " + self.title + "\n"
        for child in self.children:
            markdown += child.subtree_markdown(depth=depth + 1)
        for content in self.content:
            markdown += ("  " * (depth + 1) + "- " + "[" + content.kind.title() + "] " + content.title) + "\n"
        return markdown

    def __eq__(self, other):
        if not isinstance(other, Topic):
            return False
        return self.id == other.id

    def __getattr__(self, name):
        return topics_df.loc[self.id][name]

    def __str__(self):
        return self.title
    
    def __repr__(self):
        return f"<Topic(id={self.id}, title=\"{self.title}\")>"


class ContentItem:
    def __init__(self, content_id):
        self.id = content_id

    @property
    def topics(self):
        return [Topic(topic_id) for topic_id in topics_df.loc[correlations_df[correlations_df.content_ids.str.contains(self.id)].index].index]

    def __getattr__(self, name):
        return content_df.loc[self.id][name]

    def __str__(self):
        return self.title
    
    def __repr__(self):
        return f"<ContentItem(id={self.id}, title=\"{self.title}\")>"

    def __eq__(self, other):
        if not isinstance(other, ContentItem):
            return False
        return self.id == other.id

    def get_all_breadcrumbs(self, separator=" >> ", include_root=True):
        breadcrumbs = []
        for topic in self.topics:
            new_breadcrumb = topic.get_breadcrumbs(separator=separator, include_root=include_root)
            if new_breadcrumb:
                new_breadcrumb = new_breadcrumb + separator + self.title
            else:
                new_breadcrumb = self.title
            breadcrumbs.append(new_breadcrumb)
        return breadcrumbs
    

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

def remove_prefix(text):
    """
        Remove any unesscessary part before the real content of text
        For example: 2.3 Text -> Text
    """
    # Loại mọi thứ đứng trước và bao gồm chuỗi đánh số dạng 1. / 1.2. / 1.2.3. / v.v.
    pattern = r'^\s*[^a-zA-Z0-9]{0,5}?(?:\w+\s+)*?(\d+(?:\.\d+)*)(\s*[.:])\s*\d*\s*'

    match = re.match(pattern, text)
    if match:
        return text[match.end():].strip()
    return text.strip()

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