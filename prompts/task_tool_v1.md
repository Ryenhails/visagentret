You are searching a collection of document pages for the pages that answer a question.

You have two tools for this:
- search(query, k): a visual document retriever. It ranks page images against your query text and returns page ids (corpus_id) with scores.
- get_page(corpus_id): shows you the page image.

Question: {query}

Goal: find the pages that contain the information needed to answer this question. Several pages may each hold part of the answer. The retriever works best with focused queries, so rewrite the question into one or more search queries (for example, its key concepts, likely wording in the documents, or sub-questions), search several times, and look at promising pages to check them. Use only these two tools to search and read; do not use the shell. You do not need to answer the question itself.

When you are done, reply with only this JSON object and nothing else:
{{"ranking": [<corpus_id>, ...]}}
with at most 10 corpus_ids (integers), most relevant first.
