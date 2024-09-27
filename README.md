# Adaptability-Index

### Description

The ability for a basketball player to adapt their offensive playstyle given the other players around them is a coveted ability and is central to the decision making about the skill of a player and the design of a team roster. Despite the importance of this ability, there are no clear metrics to capture the adaptability phenomenon. Metrics like versatility capture a players ability to score, rebound or assist while metrics like portability capture a players ability to fit within a playstyle that works on many rosters. Neither of these metrics fully capture a player's ability to *change* or adapt their playstyle given which teammates are on the court with them. This work aims to quantify how much a player changes their playstyle given the lineup around them as a proxy for adaptability. 

If a player predictably changes their playstyle given the lineup around them, then their playstyle and the lineup selection are correlated. However, traditional correlation measures such as Pearson’s correlation can give skewed measures if the data are not linear or even worse, these measures cannot be computed if one of the variables is discrete or multivariate. This work implements “mutual information”, a correlation analog from information theory, which makes no assumptions about the distribution of the data and works on discrete or continuous and high dimensional data. 

The mutual information between lineup and playstyle is then defined as:


$$I(L;P,B)=I(L;B)+I(L;P|B)$$

$$I(L;B)=\sum_{b\in\mathcal{B}}^{}\sum_{l\in\mathcal{L}}^{}p(l,b)\log\left(\frac{p(l,b)}{p(l)p(b)}\right)$$

$$I(L;P|B)=\sum_{b\in\mathcal{B}}^{}p(b)\left[\sum_{l\in\mathcal{L}}^{}\int_{x\in\mathcal{P}}^{}\mu(x,l)\log\left(\frac{\mu(x,l)}{\mu(x)p(l)}\right)dx\right]$$

Where:

* $I(L;P,B)$: is the mutual information between the lineups and the joint of player position and whether they have the ball
* $I(L;B)$: is the mutual information between the lineups and whether or not the player has the ball
* $I(L;P|B)$: is the mutual information between the lineups and player position given whether or not you know the player has the ball
* $\mu(\dot{})$ is a probability density function
* $p(\dot{})$ is a probability mass function

These information theoretic quantities are estimated using a modified version of the Kozachenko-Leonenko kNN estimator described [here](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0087357). The coordinate data used to compute the mutual information estimates comes from the 2015-2016 NBA SportVU dataset, loaded using a modified version of the huggingface script loacted [here](https://huggingface.co/datasets/dcayton/nba_tracking_data_15_16).

The results of the work will soon be documented in a full manuscript and will be uploaded to this repo.
