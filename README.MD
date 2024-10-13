
<div align="center"><img src="https://github.com/Cyn7hia/Neurosymbolic_AI-PSA/blob/main/paper/pyramid.png" height="300px"/></div>
<h2 align="center">Neurosymbolic AI for Personalized Sentiment Analysis</h2>

<div align="center">
    <a>
        <img alt="Python Versions" src="https://img.shields.io/badge/python-%3E%3D3.8-blue">
    </a>
</div>

## Overview
🔥 This is a repository for our paper ([Neurosymbolic AI for Personalized Sentiment Analysis](https://www.sentic.net/personalized-sentiment-analysis.pdf)) accepted in HCII'24 ([26th International Conference on
Human-Computer Interaction](https://2024.hci.international/AI-HCI-program.html)). (Author list: Luyao Zhu, Rui Mao, Erik Cambria, and Bernard J. Jansen.)
>In this work, we propose a framework for personalized sentiment analysis that performs automatic user profiling by modeling users based on different levels of personalization, before performing sentiment analysis.
>In particular, such framework leverages **seven** levels of personalization (from bottom to top), namely: 
>***Entity***, to distinguish between humans and other intelligent agents;
>***Culture***, to take into account how different cultures perceive the same concept as positive or negative;
>***Religion***, to consider how specific religious beliefs may affect an individual’s opinion about certain topics;
>***Vocation***, to better gauge people’s opinion based on their job and education level;
>***Ideology***, to take into account political beliefs as well as social, economic, or philosophical viewpoints;
>***Personality***, to better classify certain concepts as positive or negative based on personality traits;
>finally, ***Subjectivity***, to take into account personal preferences and experiences.
<div align="center"><img src="https://github.com/Cyn7hia/Neurosymbolic_AI-PSA/blob/main/paper/personalized_sa.pdf" height="300px"/></div>
## Dataset
The dataset, Harry Potter Dialogue (HPD), can be downloaded from the [link](https://nuochenpku.github.io/HPD.github.io/).

## Usage

### Query different aspects
There are 7 different aspects, i.e., *subjectivity*, *personality*, *ideology*, *vocation*, *religion*,  *culture*, *entity*. Change the flag `aspect` to get the user profile with respect to 7 different levels of personalization.
```python
python run_query.py --aspect "entity" --proposer_model "gpt-4-turbo-2024-04-09"
```
### Sentiment analysis based on different levels of personalization
There are 9 values for flag aspect, including 7 aforementioned aspects, *all*, and *0*.
*all* means utilizing all the 7 aspects for sentiment analysis.
*0* means conduct sentiment analysis without any user profiles.
```python
python run_query_sa.py --aspect "all" --proposer_model "gpt-4-turbo-2024-04-09"
```

### Evaluation
1. `eval_by_category_whole.py` shows the personalization utility analysis by *entity* and *culture* aspects.
The results are adjusted by answer rate.
```python
python eval_by_category_whole.py
```

2. `eval_by_category.py` shows the personalization utility analysis by *entity* and *culture* aspects.
The results are not adjusted by answer rate, which means the metrics are calculated for only the queries that are successfully answered.
```python
python eval_by_category.py
```
## Issues and Usage Q&A
To ask questions, report issues or request features 🤔, please use the [GitHub Issue Tracker](https://github.com/Cyn7hia/Neurosymbolic_AI-PSA/issues). Before creating a new issue, please make sure to search for existing issues that may solve your problem.

## Citation
Please cite as
```bibtex

@inproceedings{zhu2024neurosymbolic,
  title={Neurosymbolic AI for personalized sentiment analysis},
  author={Zhu, Luyao and Mao, Rui and Cambria, Erik and Jansen, Bernard J},
  month = {jul},
  year = {2024},
  address = {Washington DC, USA},
  booktitle={International Conference on Human-Computer Interaction},
  organization={Springer}
}
