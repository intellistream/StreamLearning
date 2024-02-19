import datasets
# load dataset
qrecc_corpus = datasets.load_dataset("namespace-Pt/qrecc-corpus", split="train")
# save to jsonline format in YOUR data folder
qrecc_corpus.to_json("/home/tongjun/FlagEmbedding/FlagEmbedding/llm_embedder/data/llm-embedder/convsearch/qrecc/corpus.json", force_ascii=False, lines=True, orient="records")