from transformers import BertTokenizer, BertForSequenceClassification
from transformers import pipeline

finbert = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone',num_labels=3)
tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone')

nlp = pipeline("sentiment-analysis", model=finbert, tokenizer=tokenizer)

# sentences = ["there is a shortage of capital, and we need extra financing",
#              "growth is strong and we have plenty of liquidity",
#              "there are doubts about our finances",
#              "profits are flat"]
sentences = ["TESLA stock soars as company reports record profits and strong sales growth.",
             "The company's latest product launch has been met with mixed reviews from consumers.",
             "Economic downturn leads to widespread layoffs across multiple industries."]
results = nlp(sentences)
print(results)  #LABEL_0: neutral; LABEL_1: positive; LABEL_2: negative
