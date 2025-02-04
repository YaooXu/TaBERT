wget -r --no-parent  https://data.dws.informatik.uni-mannheim.de/webtables/2015-07/englishCorpus/compressed/

python \
    -m preprocess.common_crawl \
    --input_file data.dws.informatik.uni-mannheim.de/webtables/2015-07/englishCorpus/compressed/ \
    --output_file data/preprocessed_data/common_crawl.preprocessed.jsonl

python parallel_clean.py