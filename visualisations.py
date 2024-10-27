import matplotlib.pyplot as plt
from wordcloud import WordCloud

def plot_prediction_confidence(probability):
    fig, ax = plt.subplots()
    ax.bar(['Fake', 'Real'], probability, color=['red', 'green'])
    ax.set_ylabel('Confidence')
    ax.set_title('Prediction Confidence')
    return fig

def plot_distribution(fake_count, real_count):
    fig, ax = plt.subplots()
    ax.bar(['Fake', 'Real'], [fake_count, real_count], color=['red', 'green'])
    ax.set_ylabel('Number of Articles')
    ax.set_title('Distribution of News Types in Dataset')
    return fig

def plot_wordcloud(text):
    wordcloud = WordCloud(width=800, height=400, background_color ='white').generate(text)
    fig, ax = plt.subplots()
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis("off")
    return fig
