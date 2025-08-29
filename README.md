# SNAP Sentiment Analysis Project

## Story
<img width="655" height="328" alt="Screenshot 2025-08-26 at 3 42 06 PM" src="https://github.com/user-attachments/assets/268f9495-8c6e-4d40-80ee-66c3dc985b23" />

I made a comment under the "Food Stamps" tag on Meta's Threads app. What resulted was, as the UI reports, 1.5k comments. I defended the purchase of soda on food stamps, as a plus size woman. 
You can think what you want about the issue at hand, but it still remains a juicy bit of data to delve into and get some clarity around. Anecdotally, I believe there to be more negative comments
than positive ones. But we will see! We're turning lemons into lemonade!

## Dashboards
In order of most to least interesting.

### V2
[Individual Algorithm Analysis](https://theresaanna.github.io/SNAP_sentiment_analysis/v2/algorithm_comparison_dashboard.html):
Feed this dashboard one of the `ml_individual_*.csv` files from [this directory](https://github.com/theresaanna/SNAP_sentiment_analysis/tree/main/v2) for basic insights on performance.

[Winner Dashboard](https://theresaanna.github.io/SNAP_sentiment_analysis/v2/results_dashboard.html):
Feed this dashboard a `ml_analysis_*.csv` file, found in [this directory](https://github.com/theresaanna/SNAP_sentiment_analysis/tree/main/v2). 


## The Data
The raw data is 730 comments, non-deduped as of writing. The 1.5k comment total shown on the Threads UI accounts for every comment nested under every other comment
in the thread. I pulled only one level of nested comments from my original seven posts. Any nesting that occurred by third party discussion was not picked up.

## NLP Analysis
My first round of analysis uses TextBlob and VADER combined scores to determine the sentiment of each comment. 
If you load `threads_dashboard.html` with `nlp_analysis/threads_comments_full_analyzed.csv` you will see some interesting charts on the data and analysis. 
You can also peruse the data yourself in a viewer. 

<img width="1064" height="536" alt="Screenshot 2025-08-26 at 3 42 59 PM" src="https://github.com/user-attachments/assets/a7502dea-277d-4291-b0b5-57edbedc7455" />

Please know that many of these comments contain profanity and poke fun at my body. If you dig deep enough, you will find people telling me to kill myself. Don't read the comments, kids.

<img width="662" height="540" alt="Screenshot 2025-08-26 at 3 55 08 PM" src="https://github.com/user-attachments/assets/748ff9b9-1428-433f-a463-e24a9a653654" />

The raw data viewer shows off the bat some egregious misclassifications. Anecdotally, there seem to be a lot of false positives. I don't think either VADER or TextBlob pick up 
on sarcasm very well.

<img width="1059" height="556" alt="Screenshot 2025-08-26 at 3 55 38 PM" src="https://github.com/user-attachments/assets/c25affdb-ef95-47e8-b9f0-25e15640de87" />

## ML Re-analysis
I am not happy with the quality of data I received from basic analysis, so I'm employed machine learning to see if I could get more realistic results. I definitely did, though what I found is depressing!

I gritted my teeth through tagging 101 mostly negative comments in my data set, for a training data set.

I then trained and ran the following ML algorithms:
- Naive Bayes
- SVM
- Random Forest
- Logistic Regression

<img width="670" height="372" alt="Screenshot 2025-08-28 at 2 06 51 PM" src="https://github.com/user-attachments/assets/5afaa746-f6c5-4581-bb53-2c4afea9ba44" />

The winner was Random Forest, predicting 97.6% negative sentiment in the comments, with 88.6% average confidence. Some of the models predicted as much as 100% negative, which is quite imprecise, but gives a feel for the tone of these comments!

## What's Next: Neural Networks
Next I'm going to extend my scripts and dashboard to use Neural Network analysis. 
