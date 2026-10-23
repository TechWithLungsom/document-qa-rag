# Experiment report

These are measured offline lexical-baseline results; semantic and LLM experiments remain to be run after model setup.

| Chunk size | K | Mean precision | Mean recall (answerable) |
|---|---|---|---|
| 160 | 3 | 0.306 | 1.000 |
| 160 | 5 | 0.183 | 1.000 |
| 500 | 3 | 0.306 | 1.000 |
| 500 | 5 | 0.183 | 1.000 |

## Labels and interpretation
Relevance is defined at source-page level: every chunk on the expected page is labelled relevant, and every other chunk is irrelevant. All fixture pages are short and contain the target passage, making this exhaustive label rule inspectable. Precision is relevant retrieved chunks / retrieved chunks; recall is relevant retrieved chunks / all labelled relevant chunks. Unanswerable recall is undefined. These page-level labels are coarser than claim-level relevance.

## Experiments A–D
The fixtures fit within 160 characters per page, so the two chunk sizes produce identical passages. Neither size has demonstrated a quality advantage on this corpus. Longer documents are required to establish a useful chunk-size preference. Increasing K adds unrelated pages once the relevant page has been found, lowering precision without improving completeness for single-page answers. K=3 is the more useful tested setting for concise context.
The cafeteria-menu question has no supporting page, yet nearest-neighbor retrieval still returns passages: similarity ranking cannot establish answerability. The extractive baseline explicitly displays evidence and does not generate a factual menu answer; production generation is prompted to abstain. A reliable semantic relevance threshold requires calibration.
The filter comparison records unrestricted results versus HR-only results. Filtering removes IT pages before selecting Top-K; it improves scope when the category is known but can exclude the correct answer when chosen incorrectly.
No relevant-context/incorrect-LLM-answer case was observed because a live LLM was not run. Such a case must not be fabricated.

## Conceptual questions
1. RAG supplies external evidence that is absent from a model’s prompt or training.
2. Ordinary prompting relies on supplied instructions and model knowledge; RAG searches a document collection first.
3. Embeddings enable similarity search across related meanings. The offline baseline instead measures word overlap.
4. A vector database stores embeddings and supports nearest-neighbor lookup with source metadata.
5. Overlap reduces information loss at chunk boundaries, at the cost of duplicate context.
6. Metadata filters constrain search to a relevant scope, but wrong filters reduce recall.
7. Precision measures retrieved context purity; recall measures coverage of all labelled relevant chunks.
8. Good retrieval can still lead to a wrong answer through misreading, omitted qualifications, or unsupported synthesis.
9. Missing evidence should cause abstention; nearest neighbors alone do not prove evidence exists.
10. The table supports K=3 for precision on these fixtures. Chunk sizes tie; there is no measured basis to prefer one.

## Demonstration and reproduction
results/demo.txt is an actual captured CLI session. Run the README commands for a live demonstration. Full observed passages and extractive answers are in results/evaluation.json. To evaluate generation, run the same questions with --semantic --model, then manually label each answer’s factual claims against the retrieved passages.
