# How much hate with \#china? Analyze China-related hateful tweets two years since the Covid pandemic

Code repository for the paper 

__How much hate with \#china? Analyze China-related hateful tweets two years since the Covid pandemic__


## Data

Tweets hashtagged China posted during Jan 2020 - Jan 2022

## 3 methods or more?

Focus on the most advanced methods here:

* do I need a baseline here?
* snorkel (also to evaluate snorkel in this large unlabled data user case)
* pre-trained language model
* another pre-trained language model
* continue researching...

## models 

* [HATE BERT](https://huggingface.co/cardiffnlp/twitter-roberta-base-hate)
* [CODID-HATE BERT](https://www.dropbox.com/sh/g9uglvl3cd61k69/AACEk2O2BEKwRTcGthgROOcWa?dl=0)
* [BERTweet](https://huggingface.co/cardiffnlp/bertweet-base-hate)
* [HateXplain](https://huggingface.co/Hate-speech-CNERG/bert-base-uncased-hatexplain)
* [TimeLMs: Diachronic Language Models from Twitter](https://huggingface.co/Hate-speech-CNERG/bert-base-uncased-hatexplain)

* [SNORKEL spam tutorial](https://www.snorkel.org/use-cases/01-spam-tutorial)
* [top models in tweeteval-hate]

-> voting system

## Evaluation

evaluate each system using COVID-HATE corpus

## it might be possible to develope a voting system (or other ways to combine the systems in order to get more reliable results) based on the individual systems?

## analysis

* number/percentage of hateful tweets per day during the two years
* overall percentage: number hateful #china/all #china



### possible further analysis using the "best model"

* report \# china hate speech one year before and after the global pandemic (Jan 2019 - Jan 2021)
* Analyze hateful tweets hashtagged with other countries (largest 10 economies) and visualise the comparison



## References

* [Hate speech detection: Challenges and solutions](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0221152)
* [Deep Learning for Hate Speech Detection in Tweets](https://dl.acm.org/doi/abs/10.1145/3041021.3054223)
* [comparison hate classifiers](https://iopscience.iop.org/article/10.1088/1757-899X/830/3/032006#:~:text=The%20results%20show%20that%20the,the%20classification%20of%20hate%20speech.)
* [resources and benchmark datasets for hate speech detection: a systemetic review](https://link.springer.com/article/10.1007/s10579-020-09502-8)
* [COVID-HATE data](https://dl.acm.org/doi/abs/10.1145/3487351.3488324)
* [COVID-HateBERT](https://ieeexplore.ieee.org/abstract/document/9680128)
* [An Extensive Guide to collecting tweets from Twitter API v2 for academic research using Python 3](https://towardsdatascience.com/an-extensive-guide-to-collecting-tweets-from-twitter-api-v2-for-academic-research-using-python-3-518fcb71df2a)

* [Twitter API dev docs](https://developer.twitter.com/en/docs/twitter-api/tweets/search/api-reference/get-tweets-search-all)


## Noting taking

* number of \#china tweets posted on 2022-02-20: 5345

## MISC

* [dev portal dashboard](https://developer.twitter.com/en/portal/dashboard)

## DATA

collected and annotated data should be eventually uploaded to google drive.

