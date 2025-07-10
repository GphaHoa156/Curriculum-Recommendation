# Project overview

This project is a solution for Kaggle featured competition Learning Equality - Curriculum Recommendations. You can view more details via this link https://www.kaggle.com/competitions/learning-equality-curriculum-recommendations.
Its goal is to given a lots of **topic** and **content**. We will need to find the correct content for each topic.

# How it's made:
- Programming language: Python
- Tech used: pretrained models from Huggingface, thus you will want to have a GPU from your computer or cloud via Colab or Kaggle.
- Solution approach:
    - The pipeline I used here is **Retrieval and Reranking** pipeline from SentenceTransformer (for more details you can visit the official site https://sbert.net/examples/sentence_transformer/applications/retrieve_rerank/README.html)
            <Diagram-here>
    - Our data are multilingual texts, so here I will have 2 ways to use embedding models:
       - Directly use multingual embedding models
       - First translate all data into English with Google Translate API
    - Through expriments, using second approach give a much better results. But the downside is cost (cause Google API is not free), so this is just test that if a bussiness want to run this system, they can consider both ways
    - The pipeline is devided into 2 steps **retrieval** and **reranking**. We will also finetune 2 model in both steps (I will talk about this later).
       - In retrieval step, we simply encode **topic** and **content** seperately with a bi-encoder (which I used here is **all-mpnet-based-v2** https://huggingface.co/sentence-transformers/all-mpnet-base-v2). Then retrieve top-k with FAISS.
       - In the second step, the retrieved top-k with be put into a cross-encoder (which I used here is **cross-encoder/stsb-distilroberta-base** but I think using **cross-encoder/ms-marco-MiniLM-L6-v2** could result in better result cause I did not have enough resources). This cross-encoder will encode both **topic** and **content** simultaneously, because of this it will gain much more insight of data and give us a better result.
    - Why finetune?
       - The reason is result from using just pretrained model didnt made me satisfy, there are many easy cases where model can't give the better result to a high ranking. Also, after fine-tuning, you can see the model performance increase significantly.
       - For finetuning bi-encoder and cross-encoder, the technique I use is to oversampling.
# Conclusion and my thought of project
- What I tried and didn't work for me, but you can try it yourself:
    - Using summary model to shrink text size.
    - Add margin to the loss for finetuning bi-encoder.
    - Apply hard negative mining for cross-encoder.









