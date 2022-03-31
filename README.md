## How much hate with \#china? A preliminary ananlysis on hateful tweets hashtagged china two years after the pandemic


* [Paper]()
* _[Author: Jinghua Xu](jinhxu.github.io)_

### Absrtact

> Following the outbreak of a global pandemic,
online contents are filled with hate speech. Donald Trump’s "Chinese Virus" tweet shifted the
blame for the spread of the Covid-19 virus to
China and the Chinese people, which triggered
a new round of anti-China hate both online
and offline. This research intends to examine
China-related hate speech on Twitter during
the two years following the burst of the pan-
demic (2020 and 2021). Through Twitter’s API,
in total 2,172,333 tweets hashtagged #china
posted during the time were collected. By employing multiple state-of-the-art pretrained language models for hate speech detection, a wide
range of hate of various types is ensured to
be detected, yielding an automatically labeled
anti-China hate speech dataset. The analysis
conducted in this research reveals the number
of #china tweets and predicted hateful #china
tweets changing over the two years time span,
and identifies 2.5% of #china tweets hateful
in 2020 and 1.9% in 2021. Both ratios are
found to be above the average rate of online
hate speech on Twitter at 0.6% estimated in
Gao et al. (2017)

### Data 

Tweets (excl. retweets, replies and quotes) hastagged china posted during the year of 2020 and 2021.

* [data](https://drive.google.com/drive/folders/19_IJP2E6HmRHLYsip5cWMnxPIuHgE09r?usp=sharing)

* _[Twitter crawler script](https://github.com/JINHXu/how-much-hate-with-china/tree/main/scripts/notebooks/get_data)_

### Models

Hate speech detection models:

* [COVID-HATE BERT](https://arxiv.org/abs/2005.12423)
* [cardiffnlp/twitter-roberta-base-hate](https://arxiv.org/pdf/2010.12421)
* [hateXplain](https://arxiv.org/abs/2012.10289)


* _[scripts](https://github.com/JINHXu/how-much-hate-with-china/tree/main/scripts/notebooks/get_predictions)_


### Analysis

* [2020](https://colab.research.google.com/drive/1ey7XuGHk8XUdzqfCNHLbpvRt70Yjfq9I?usp=sharing)

* [2021](https://colab.research.google.com/drive/1uMpKYhIZAFVmXXFpuf1kYcraRnN4qkKp?usp=sharing)



<!-- # How much hate with \#china? Analyze China-related hateful tweets two years since the Covid pandemic

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
* [COVID-HATE BERT](https://www.dropbox.com/sh/g9uglvl3cd61k69/AACEk2O2BEKwRTcGthgROOcWa?dl=0)
* [BERTweet](https://huggingface.co/cardiffnlp/bertweet-base-hate)
* [HateXplain/TimeLMs: Diachronic Language Models from Twitter](https://huggingface.co/Hate-speech-CNERG/bert-base-uncased-hatexplain)

* ~~[SNORKEL spam tutorial](https://www.snorkel.org/use-cases/01-spam-tutorial)~~
* ~~[top models in tweeteval-hate]()~~

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

* [Misinformation and Hate Speech: The Case of Anti-Asian Hate Speech During the COVID-19 Pandemic](https://tsjournal.org/index.php/jots/article/view/13)
* [ElSherief 2018](https://github.com/mayelsherif/hate_speech_icwsm18)


## Noting taking

* number of \#china tweets posted on 2022-02-20: 5345

## MISC

* [dev portal dashboard](https://developer.twitter.com/en/portal/dashboard)

## DATA
 
collected and annotated data should be eventually uploaded to google drive.
 -->
