# Features

This directory contains summary files for features that we extracted from SAEs.

* [feature_summaries.json](./feature_summaries.json) includes **all** features and their respective top activating tokens, activation values, and coherence scores.
* [parsed_features.json](./parsed_features.json) includes **just** the features that we deeemed relevant to the "medical" domain as indicated by a keyword match as well as a human evaluation. This is the structured synthesis of both the `parsing_*.txt` files.
  * [parsing_baseline.txt](./parsing_baseline.txt) is an unstructured file that we built during our human evals containing just the features extracted from our fine-tuned `gpt2-small` model that were relevant to the "medical" domain 
  * [parsing_finetuned.txt](./parsing_finetuned.txt) is an unstructured file that we built during our human evals containing just the features extracted from the pretrained `gpt2-small` model that were relevant to the "medical" domain