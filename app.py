from flask import Flask, render_template, request, redirect, url_for
import re
import googleapiclient.discovery
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
from requests import HTTPError
from wordcloud import WordCloud
import emoji  # Import the emoji library
from flask import Flask, render_template, request, redirect, url_for
import re
import googleapiclient.discovery
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import emoji  # Import the emoji library
from moviepy.editor import VideoFileClip
from flask import Flask, render_template, request, redirect, url_for
import googleapiclient.discovery
from flask import Flask, render_template, request, redirect, url_for
import googleapiclient.discovery
from youtube_transcript_api import YouTubeTranscriptApi
import spacy
from flask import Flask, render_template, request, redirect, url_for
import googleapiclient.discovery
from youtube_transcript_api import YouTubeTranscriptApi
import spacy
from flask import Flask, render_template, request, redirect, url_for
import googleapiclient.discovery
from youtube_transcript_api import YouTubeTranscriptApi
import spacy
import re
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.summarizers.lex_rank import LexRankSummarizer
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from flask import Flask,render_template,request
from transformers import T5ForConditionalGeneration, T5Tokenizer
from urllib.parse import urlparse, parse_qs
from youtube_transcript_api import YouTubeTranscriptApi as ytt
from flask import send_file
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import os
from flask import Flask, render_template, request, redirect, url_for, make_response
from youtube_transcript_api import YouTubeTranscriptApi
from reportlab.pdfgen import canvas



app = Flask(__name__, static_url_path='/static')

# Initialize df as a global variable
df = None

model = T5ForConditionalGeneration.from_pretrained("t5-small")

tokenizer = T5Tokenizer.from_pretrained("t5-small")

YOUTUBE_API_KEY = "AIzaSyBe2QKHD-j29APhTn-wqa5pd8rWxw8eucQ"


def extract_video_id(link):
    video_id = re.match(r'^.*(?:youtu.be/|v/|u/\w/|embed/|watch\?v=)([^#&?]*).*', link)
    if video_id:
        return video_id.group(1)
    else:
        return None

def google_api(video_id):
    api_service_name = "youtube"
    api_version = "v3"
    DEVELOPER_KEY = "AIzaSyBe2QKHD-j29APhTn-wqa5pd8rWxw8eucQ"

    youtube = googleapiclient.discovery.build(api_service_name, api_version, developerKey=DEVELOPER_KEY)

    request = youtube.commentThreads().list(
        part="id,snippet",
        maxResults=300,
        order="relevance",
        videoId=video_id
    )

    response = request.execute()
    return response

def create_df_author_comments(response):
    authorname = []
    comments = []

    for item in response.get("items", []):
        comment = item.get("snippet", {}).get("topLevelComment", {}).get("snippet", {})
        authorname.append(comment.get("authorDisplayName", ""))
        comments.append(comment.get("textOriginal", ""))

    df = pd.DataFrame({"Author": authorname, "Comment": comments})
    return df

def analyze_sentiment(df):
    sid = SentimentIntensityAnalyzer()
    df["Sentiment"] = df["Comment"].apply(lambda x: sid.polarity_scores(x)["compound"])
    df["Sentiment_Label"] = df["Sentiment"].apply(lambda x: "Positive" if x >= 0.05 else "Neutral" if -0.05 <= x < 0.05 else "Negative")

def calculate_average_sentiment(df):
    total_count = len(df)
    
    positive_count = len(df[df["Sentiment_Label"] == "Positive"])
    neutral_count = len(df[df["Sentiment_Label"] == "Neutral"])
    negative_count = len(df[df["Sentiment_Label"] == "Negative"])
    
    positive_percentage = (positive_count / total_count) * 100
    neutral_percentage = (neutral_count / total_count) * 100
    negative_percentage = (negative_count / total_count) * 100
    
    if positive_percentage >= 50:
        dominant_emoji = "😃"
    elif neutral_percentage == 50:
        dominant_emoji = "😐"
    elif negative_percentage >= 50:
        dominant_emoji = "😞"
    else:
        dominant_emoji = "😐"  # Default to neutral if no clear dominance
    
    return {
        "positive_percentage": positive_percentage,
        "neutral_percentage": neutral_percentage,
        "negative_percentage": negative_percentage,
        "dominant_emoji": dominant_emoji
    }

def get_video_title(video_id):
    api_service_name = "youtube"
    api_version = "v3"
    DEVELOPER_KEY = "AIzaSyBe2QKHD-j29APhTn-wqa5pd8rWxw8eucQ"

    youtube = googleapiclient.discovery.build(api_service_name, api_version, developerKey=DEVELOPER_KEY)

    request = youtube.videos().list(
        part="snippet",
        id=video_id
    )

    response = request.execute()
    if response.get("items"):
        return response["items"][0]["snippet"]["title"]
    else:
        return "Video Title Not Found"

def count_authors_by_sentiment(df):
    positive_authors = df[df["Sentiment_Label"] == "Positive"]["Author"].nunique()
    neutral_authors = df[df["Sentiment_Label"] == "Neutral"]["Author"].nunique()
    negative_authors = df[df["Sentiment_Label"] == "Negative"]["Author"].nunique()
    return positive_authors, neutral_authors, negative_authors

def prepare_chart_data(df):
    sentiment_counts = df["Sentiment_Label"].value_counts().reset_index()
    sentiment_counts.columns = ["Sentiment", "Count"]

    fig = px.pie(sentiment_counts, names="Sentiment", values="Count", title="")
    pie_chart = fig.to_html()

    fig = px.bar(sentiment_counts, x="Sentiment", y="Count", title="")
    bar_chart = fig.to_html()

    return pie_chart, bar_chart


def get_video_caption(video_id, api_key):
    youtube = googleapiclient.discovery.build("youtube", "v3", developerKey=api_key)
    caption_response = youtube.captions().list(part="snippet", videoId=video_id).execute()

    if "items" in caption_response:
        caption_id = caption_response["items"][0]["id"]
        caption = youtube.captions().download(id=caption_id).execute()
        return caption.get("body", "")
    return ""


def get_video_transcript(video_id):
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        text = ' '.join(entry['text'] for entry in transcript)
        return text
    except Exception as e:
        return str(e)
    
def generate_summary(transcript):
    LANGUAGE = "english"
    
    parser = PlaintextParser.from_string(transcript, Tokenizer(LANGUAGE))
    
    # You can choose a different summarizer here (LsaSummarizer, LexRankSummarizer, etc.)
    summarizer = LexRankSummarizer()
    summary = summarizer(parser.document, sentences_count=5)  # Adjust the sentence count as needed
    
    return " ".join([str(sentence) for sentence in summary])


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        analysis_type = request.form.get("analysis_type")
        if analysis_type == "youtube_sentiment":
            return redirect(url_for('load_html_1'))  # Redirect to YouTube sentiment analysis
        elif analysis_type == "youtube_summary":
            return redirect(url_for('load_summary'))  # Redirect to YouTube video summary
    return render_template("index.html")





def get_video_info(video_id):
    try:
        api_service_name = "youtube"
        api_version = "v3"
        DEVELOPER_KEY = "AIzaSyBe2QKHD-j29APhTn-wqa5pd8rWxw8eucQ"

        youtube = googleapiclient.discovery.build(api_service_name, api_version, developerKey=DEVELOPER_KEY)

        request = youtube.videos().list(
            part="snippet,statistics",
            id=video_id
        )

        response = request.execute()
        if response.get("items"):
            video_data = response["items"][0]["statistics"]
            likes = video_data.get("likeCount", "N/A")
            dislikes = video_data.get("dislikeCount", "N/A")
            views = video_data.get("viewCount", "N/A")
            num_comments = video_data.get("commentCount", "N/A")

            return {
                "likes": likes,
                "dislikes": dislikes,
                "views": views,
                "num_comments": num_comments
            }
        else:
            return {
                "likes": "N/A",
                "dislikes": "N/A",
                "views": "N/A",
                "num_comments": "N/A"
            }
    except HTTPError as e:
        return {
            "likes": "N/A",
            "dislikes": "N/A",
            "views": "N/A",
            "num_comments": "N/A"
        }




@app.route("/youtube_video_info", methods=["GET", "POST"])
def youtube_video_info():
    if request.method == "POST":
        video_link = request.form.get("video_link")
        video_id = extract_video_id(video_link)

        if video_id:
            video_title = get_video_title(video_id)  # Get the video title
            video_info = get_video_info(video_id)

            return render_template("video_info.html", video_link=video_link, video_title=video_title, **video_info)

        else:
            return "Invalid YouTube video link."

    return render_template("video_info_form.html")

@app.route('/summary_index')
def summary_index():
    return render_template('summary_index.html')


def extract_video_id(url:str):
    query = urlparse(url)
    if query.hostname == 'youtu.be': return query.path[1:]
    if query.hostname in {'www.youtube.com', 'youtube.com'}:
        if query.path == '/watch': return parse_qs(query.query)['v'][0]
        if query.path[:7] == '/embed/': return query.path.split('/')[2]
        if query.path[:3] == '/v/': return query.path.split('/')[2]
    # fail?
    return None

def summarizer(script):
    input_ids = tokenizer.encode("summarize: " + script, return_tensors="pt", max_length=512, truncation=True)
    
    # Generate a longer summarization output
    outputs = model.generate(
        input_ids,
        max_length=1000,  # Increase this value to make the summary longer
        min_length=100,
        length_penalty=2.0,
        num_beams=4,
        early_stopping=False  # Try setting early_stopping to False
    )

    summary_text = tokenizer.decode(outputs[0])
    return summary_text

@app.route('/summarize', methods=['GET', 'POST'])
def video_transcript():
    if request.method == 'POST':
        url = request.form['youtube_url']
        video_id = extract_video_id(url)
        data = ytt.get_transcript(video_id, languages=['de', 'en'])

        scripts = []
        for text in data:
            for key, value in text.items():
                if key == 'text':
                    scripts.append(value)
        transcript = " ".join(scripts)
        summary = summarizer(transcript)
        
        # Render the summary in the HTML template
        return render_template('summary.html', summary=summary)
    else:
        return "ERROR"



@app.route("/youtube_summary", methods=["GET", "POST"])
def load_summary():
    if request.method == "POST":
        video_link = request.form.get("video_link")
        video_id = extract_video_id(video_link)
        
        if video_id:
            video_title = get_video_title(video_id)
            video_transcript = get_video_transcript(video_id)
            if video_transcript:
                summary = generate_summary(video_transcript)
                return render_template("youtube_summary.html", video_link=video_link, video_title=video_title, summary=summary)
            else:
                return "Transcript not available for this video."

        return "Invalid YouTube video link."

    return render_template("youtube_summary_form.html")


    
@app.route("/index_1", methods=["GET", "POST"])
def load_html_1():
    global df  # Access the global df variable
    if request.method == "POST":
        video_link = request.form.get("video_link")
        video_id = extract_video_id(video_link)

        if video_id:
            video_title = get_video_title(video_id)  # Get the video title

            response = google_api(video_id)
            df = create_df_author_comments(response)
            analyze_sentiment(df)

            # Calculate sentiment statistics
            sentiment_stats = calculate_average_sentiment(df)

            # Filter only positive comments
            positive_comments = df[df["Sentiment_Label"] == "Positive"]

            return render_template("results.html", video_link=video_link, video_title=video_title,
                       positive_comments=positive_comments, **sentiment_stats)

        else:
            return "Invalid YouTube video link."

    return render_template("index_1.html")

@app.route("/charts", methods=["GET"])
def load_charts():
    global df  # Access the global df variable

    if df is not None:
        pie_chart, bar_chart = prepare_chart_data(df)
        return render_template("charts.html", pie_chart=pie_chart, bar_chart=bar_chart)

    return "No data available for generating charts."

@app.route("/word_cloud", methods=["GET"])
def display_word_cloud():
    global df  # Access the global df variable

    if df is not None:
        comments_text = " ".join(df["Comment"].tolist())
        wordcloud = WordCloud(width=800, height=400, background_color="white").generate(comments_text)

        plt.figure(figsize=(8, 4))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig("static/wordcloud.png")

        return render_template("word_cloud.html")

    return "No data available for generating word cloud."

if __name__ == "__main__":
    app.run(debug=True)
