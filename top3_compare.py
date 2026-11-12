import sys
import re
from pathlib import Path
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt

# libraries for NLP
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer

# Word cloud
from wordcloud import WordCloud


# !!! Transform stage !!!

def check_data(df: pd.DataFrame) -> bool:
    if df.empty:
        print("No reviews imported.")
        return False

    required = ["asin", "overall", "reviewText"]
    for c in required:
        if c not in df.columns:
            raise Exception(f"Missing required column: {c}")

    # check for null in key columns
    null_count = df[required].isnull().sum().sum()
    if null_count > 0:
        print(f"Warning: {null_count} null values found in asin/overall/reviewText. These will be handled during cleaning.")

    # check for duplicates
    dup_count = df.duplicated(subset=["asin", "reviewText"]).sum()
    if dup_count > 0:
        print(f"Warning: {dup_count} duplicate reviews found (asin + reviewText). These will be removed during cleaning.")

    return True


# STEP 1 !!! Extract stage !!! 

def step1_find_file(json_filename: str) -> Path:
    print("STEP 1: Looking for the JSON file...")

    p = Path(json_filename)
    if not p.exists():
        print("ERROR: File not found:", p)
        sys.exit()

    print("Found file:", p)
    return p


# STEP 2: To Ensure Natural language toolkit(NLTK) exist 

def step2_setup_nlp():
    print("\nSTEP 2: Setting up NLP resources")

    resources = ["stopwords", "wordnet", "vader_lexicon"] #required NLP datasets

    #need to download resources if they are missing
    for rsrc in resources:
        try:
            nltk.data.find(rsrc)
        except LookupError:
            nltk.download(rsrc)

    #initializing NLP tools
    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()
    sia = SentimentIntensityAnalyzer()

    print("NLP resources ready.")
    return stop_words, lemmatizer, sia


# STEP 3: Finding top 3 products with most reviews

def step3_find_top3_asins(p: Path, chunksize: int = 200000) -> list:
    print("\nSTEP 3: Finding top 3 products by number of reviews ...")

    asin_review_counts = {} #to store review count of each product 
    total_rows_processed = 0 #to keep track of rows that has been read so far

    for chunk in pd.read_json(p, lines=True, chunksize=chunksize):
        
        total_rows_processed += len(chunk) #to update progressing row count

        if "asin" not in chunk.columns:
            continue

        #to count how many times each ASIN appears 
        asin_chunk_counts = chunk["asin"].astype(str).value_counts(dropna=True)
        
        for asin_code, review_count in asin_chunk_counts.items():
            
            asin_review_counts[asin_code] = asin_review_counts.get(asin_code, 0) + int(review_count)

        print("Processed rows so far:", total_rows_processed)

    if not asin_review_counts:
        raise Exception("ERROR: No ASINs found — check the dataset contains an ASIN field for sure")

    #to sort products by highest number of reviews and select top 3 from them
    top3_products = sorted(asin_review_counts.items(), key=lambda item: item[1], reverse=True)[:3]


    #to extract only the ASIN codes
    top_asins = [asin for asin, _ in top3_products]

    print("\nChosen TOP 3 ASINs:")
    for asin, count in top3_products:
        print(f"- {asin} ({count} reviews)")

    return top_asins


# STEP 4: Load only top 3 reviews (chunked)

def step4_load_only_top3(p: Path, top_asins: list, chunksize: int = 200000) -> pd.DataFrame:
    print("\nSTEP 4: Loading the reviews for the selected 3 ASINs only...")

    filtered_parts = [] #to store the filtered chunks
    kept_rows = 0 #to keep track of matching rows so far

    for chunk in pd.read_json(p, lines=True, chunksize=chunksize):
        if "asin" not in chunk.columns:
            continue

        chunk["asin"] = chunk["asin"].astype(str)
        #keep only the reviews for the chosen top 3 products
        chunk = chunk[chunk["asin"].isin(top_asins)]
        
        if chunk.empty:
            continue

        # Keep only relevant columns if they exist
        keep_cols = [c for c in ["asin", "overall", "reviewTime", "unixReviewTime",
                                 "reviewText", "summary", "verified", "vote", "reviewerID", "reviewerName"]
                     if c in chunk.columns]
        chunk = chunk[keep_cols]

        filtered_parts.append(chunk)
        kept_rows += len(chunk)
        print("Kept rows so far:", kept_rows)

    if not filtered_parts:
        raise Exception("ERROR: No reviews found for top 3 ASINs after filtering.")

    df = pd.concat(filtered_parts, ignore_index=True)
    print("Loaded rows for the top 3 products combined:", len(df))
    return df



# STEP 5: check overall data structure

def step5_show_basic_info(df: pd.DataFrame):
    
    print("\nSTEP 5: Showing dataset structure")

    print("Number of rows:", df.shape[0])
    print("Number of columns:", df.shape[1])
    print("Column names:", list(df.columns))
    
    print("\nFirst 5 rows:")
    print(df.head())



# STEP 6: Data Cleaning and preparing the review dataset

def preprocess_text(text: str, stop_words: set, lemmatizer: WordNetLemmatizer) -> str:
    text = str(text).lower()

    # remove URLs
    text = re.sub(r"http\S+|www\.\S+", " ", text)

    # keep letters/spaces only
    text = re.sub(r"[^a-z\s]", " ", text)

    # remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    tokens = text.split()
    tokens = [t for t in tokens if t not in stop_words and len(t) > 2]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]

    return " ".join(tokens)


def step6_clean_transform_nlp(df: pd.DataFrame, stop_words: set, lemmatizer: WordNetLemmatizer, sia: SentimentIntensityAnalyzer) -> pd.DataFrame:
    print("\nSTEP 6: Data Cleaning and analysis + NLP + VADER sentiment...")

    df = df.copy()

    # check key columns
    for col in ["asin", "overall", "reviewText"]:
        if col not in df.columns:
            raise Exception(f"Missing required column: {col}")

    df["asin"] = df["asin"].astype(str)
    df["overall"] = pd.to_numeric(df["overall"], errors="coerce")

    # date handling
    if "unixReviewTime" in df.columns:
        df["review_datetime"] = pd.to_datetime(df["unixReviewTime"], unit="s", errors="coerce")
    elif "reviewTime" in df.columns:
        df["review_datetime"] = pd.to_datetime(df["reviewTime"], errors="coerce")
    else:
        df["review_datetime"] = pd.NaT

    df["month"] = df["review_datetime"].dt.to_period("M").astype(str)

    # clean verified in vote columns
    if "verified" in df.columns:
        df["verified"] = df["verified"].fillna(False).astype(bool)
    else:
        df["verified"] = False

    if "vote" in df.columns:
        df["vote"] = df["vote"].astype(str).str.replace(",", "", regex=False)
        df["vote"] = pd.to_numeric(df["vote"], errors="coerce").fillna(0)
    else:
        df["vote"] = 0

    # text cleanup
    df["reviewText"] = df["reviewText"].astype(str).str.strip()

    # remove bad rows
    before = len(df)
    df = df.dropna(subset=["asin", "overall", "reviewText"])
    df = df[df["reviewText"].str.len() > 0]
    df = df.drop_duplicates(subset=["asin", "reviewText"])
    after = len(df)

    print(f"Rows before cleaning: {before}")
    print(f"Rows after cleaning:  {after}")

    # create numeric features
    df["review_length"] = df["reviewText"].str.len()
    df["review_word_count"] = df["reviewText"].str.split().str.len()

    # clean text for NLP
    df["review_clean"] = df["reviewText"].apply(lambda x: preprocess_text(x, stop_words, lemmatizer))

    # apply VADER sentiment 
    df["sentiment_compound"] = df["review_clean"].apply(lambda x: sia.polarity_scores(x)["compound"])

    # create sentiment label for charts 
    def label_sent(v):
        if v >= 0.05:
            return "positive"
        elif v <= -0.05:
            return "negative"
        else:
            return "neutral"

    df["sentiment_label"] = df["sentiment_compound"].apply(label_sent)

    return df



# STEP 7: Per-product EDA + exports + plots

def safe_mkdir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def top_words(texts, n=20):
    
    tokens = " ".join(texts).split()
    #count each word
    c = Counter(tokens)
    return c.most_common(n)


def top_bigrams(texts, n=15):
    #split all cleaned review text into tokens
    tokens = " ".join(texts).split()
    #create double word phrases
    bigrams = list(zip(tokens, tokens[1:]))
    c = Counter([" ".join(bg) for bg in bigrams])
    return c.most_common(n)


def make_wordcloud(texts, outpath: Path):
    # join all the cleaned review text in 1 string
    text = " ".join(texts).strip()
    
    if len(text) == 0:
        return

    #to produce wordcloud
    wc = WordCloud(width=1200, height=700, background_color="white").generate(text)
    
    plt.figure(figsize=(10,6))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()


def eda_for_one_product(df: pd.DataFrame, asin: str, outdir: Path):
    

    print("\nSTEP 7: Running EDA for product:", asin)

    #filter full dataset to extract review for 1 of 3 selected product
    product_df = df[df["asin"] == asin].copy()
    safe_mkdir(outdir) #for output folder

    if product_df.empty:
        print("No reviews found for this product.")
        return

    print("Number of reviews:", len(product_df))



    # !!! Summary statistics including overall sentiment !!!
    avg_rating = product_df["overall"].mean()
    avg_review_length = product_df["review_length"].mean()
    avg_word_count = product_df["review_word_count"].mean()
    avg_sentiment = product_df["sentiment_compound"].mean()

    print("Average rating:", round(avg_rating,2))
    print("Average review length:", round(avg_review_length,2))
    print("Average word count:", round(avg_word_count,2))
    print("Average sentiment score:", round(avg_sentiment,3))

    # save summary statistics
    summary_df = pd.DataFrame([{
        "asin": asin,
        "review_count": len(product_df),
        "average_rating": avg_rating,
        "average_review_length": avg_review_length,
        "average_word_count": avg_word_count,
        "average_sentiment": avg_sentiment
    }])

    summary_df.to_csv(outdir / "summary.csv", index=False)

    # monthly review trend
    monthly_df = (
        product_df.groupby("month", dropna=False)
        .agg(
            review_count=("asin","size"),
            average_rating=("overall","mean"),
            average_sentiment=("sentiment_compound","mean")
        )
        .reset_index()
        .sort_values("month")
    )

    #export monthly_trend to csv
    monthly_df.to_csv(outdir / "monthly_trends.csv", index=False)

    # analyse cleaned text for top word and phrases
    texts = product_df["review_clean"].dropna().astype(str).tolist()

    #20 topmost used word
    words_df = pd.DataFrame(
        top_words(texts,20),
        columns=["word","count"]
    )

    words_df.to_csv(outdir / "top_words.csv", index=False)
    
    #15 topmost used phrases
    bigrams_df = pd.DataFrame(
        top_bigrams(texts,15),
        columns=["bigram","count"]
    )

    bigrams_df.to_csv(outdir / "top_bigrams.csv", index=False)

    # generate word cloud
    make_wordcloud(texts, outdir / "wordcloud.png")

    # rating distribution plot
    plt.figure(figsize=(8,5))
    product_df["overall"].value_counts().sort_index().plot(kind="bar")
    plt.title(f"Rating Distribution - {asin}")
    plt.xlabel("Star Rating")
    plt.ylabel("Number of Reviews")
    plt.tight_layout()
    plt.savefig(outdir / "rating_distribution.png")
    plt.close()

    # sentiment distribution plot
    plt.figure(figsize=(8,5))
    product_df["sentiment_label"].value_counts().plot(kind="bar")
    plt.title(f"Sentiment Distribution - {asin}")
    plt.xlabel("Sentiment")
    plt.ylabel("Number of Reviews")
    plt.tight_layout()
    plt.savefig(outdir / "sentiment_distribution.png")
    plt.close()

    # VADER compound sentiment histogram
    plt.figure(figsize=(8,5))
    product_df["sentiment_compound"].plot(kind="hist", bins=30)
    plt.title(f"VADER Compound Sentiment Distribution - {asin}")
    plt.xlabel("Compound Sentiment Score (-1 to +1)")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(outdir / "sentiment_compound_hist.png")
    plt.close()

    # bubble chart: rating vs sentiment, sized by review word count
    plot_df = product_df.copy()
    plot_df["bubble_size"] = plot_df["review_word_count"].fillna(1) * 2

    plt.figure(figsize=(8, 5))
    plt.scatter(
    plot_df["overall"],
    plot_df["sentiment_compound"],
    s=plot_df["bubble_size"],
    alpha=0.5
    )

    plt.title(f"Bubble Chart: Rating vs Sentiment - {asin}")
    plt.xlabel("Star Rating")
    plt.ylabel("Sentiment Compound Score")
    plt.tight_layout()
    plt.savefig(outdir / "bubble_rating_vs_sentiment.png")
    plt.close()

    # review length histogram
    plt.figure(figsize=(8,5))
    product_df["review_word_count"].plot(kind="hist", bins=30)
    plt.title(f"Review Word Count Distribution - {asin}")
    plt.xlabel("Word Count")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(outdir / "review_word_count_hist.png")
    plt.close()

    print("Outputs saved in:", outdir)



# MAIN

def main():
    # file path
    JSON_FILE = r"D:\python\assignment1\Industrial_and_Scientific.json\Industrial_and_Scientific.json"

    #step 1 - check file
    p = step1_find_file(JSON_FILE)

    #step 2 - setup NLP tools
    stop_words, lemmatizer, sia = step2_setup_nlp()

    #step 3 - identify top 3 products
    top_asins = step3_find_top3_asins(p)

    #step 4 - load reviews for those 3 products
    raw_df = step4_load_only_top3(p, top_asins)

    print("\nTotal reviews loaded:", len(raw_df))


    #step 5 - validate dataset before cleaning
    print("\nRunning dataset validation checks...")
    if not check_data(raw_df):
        print("Dataset validation failed. Stop program.")
        sys.exit()

    #step 6 - show dataset structure
    step5_show_basic_info(raw_df)


    #step 7 - clean and transform data
    df = step6_clean_transform_nlp(raw_df, stop_words, lemmatizer, sia)

    #step 8 - EDA for each product
    for asin in top_asins:
        eda_for_one_product(df, asin, Path("outputs") / asin)
    
    print("\nProgram finished successfully.")

    
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("Error!", e)
