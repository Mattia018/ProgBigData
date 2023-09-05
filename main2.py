
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns

import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.window import Window
from pyspark.sql.functions import hour, avg, col,trim, to_date, date_format,when, substring, explode, lower,udf, split
from pyspark.sql.types import StringType,StructType,StructField, DoubleType
from pyspark.ml.feature import StringIndexer, VectorAssembler,IndexToString 
from pyspark.ml.stat import Correlation
from pyspark.ml.classification import RandomForestClassifier, LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.ml import Pipeline


import nltk
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import os
import sys

os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable


#spark = SparkSession.builder.config("spark.executor.cores", "4").config("spark.executor.memory", "8g").getOrCreate()

spark = SparkSession.builder.config("spark.executor.cores", "8") \
    .config("spark.executor.memory", "8g") \
    .config("spark.driver.memory", "4g") \
    .config("spark.memory.offHeap.enabled", "true") \
    .config("spark.memory.offHeap.size", "2g") \
    .getOrCreate()

###   Inizializzazione Dataset json  ###

directory="Data Set/001"
file_paths = [os.path.join(directory, file) for file in os.listdir(directory) if file.endswith(".json")]
df1 = spark.read.json(file_paths)
df1= df1.drop("_corrupt_record")


directory2="Data Set/002"
file_paths2 = [os.path.join(directory2, file) for file in os.listdir(directory2) if file.endswith(".json")]
df2 = spark.read.json(file_paths2)



df_start = df1.union(df2)


df_start = df_start.select(
    col("id").alias("tweet_id"),
    col("created_at").alias("tweet_date_full"),
    col("geo.coordinates").alias("tweet_pos"),
    col("place.full_name").alias("tweet_place"),
    col("aidr.crisis_name").alias("crisis_name"),
    col("user.id").alias("user_id"),
    col("user.name").alias("user_name"),
    col("user.location").alias("user_location"),
    col("user.followers_count").alias("user_followers"),
    col("user.verified").alias("user_verified"),       
    "text",
    col("entities.hashtags.text").alias("hashtags")
)


# Divisione campo Date

# Estraggo la parte della data (i primi 10 caratteri)
df_start = df_start.withColumn("tweet_date", substring("tweet_date_full", 1, 10))

# Estraggo la parte dell'ora (dopo il 10° carattere)
df_start = df_start.withColumn("tweet_time", substring("tweet_date_full", 12, 8))

def dataset_start():
    ds1= df_start.select( col("tweet_id").cast("string"), col("user_id").cast("string"), "user_name", "user_followers", \
                         "user_verified", "text", "hashtags", "tweet_date", "tweet_time",\
                         "user_location","tweet_place", "tweet_pos" )
    

    ds1_p=ds1.toPandas()
    ds1_p= ds1_p.sample(15, ignore_index=True,random_state=44)    
    ds1_count= ds1.count() 
    
    return ds1_p, ds1_count


###  Dataset 3   ###

csv_path = "Data Set/harvey_data_2017_aidr_classification.txt"
df3 = spark.read.csv(csv_path, sep="\t", header=True)
df3 = df3.drop("Date")

def dataset_aidr_classification():

    ds2= df3.select("tweet_id", "AIDRLabel", "AIDRConfidence")

    ds2_p= ds2.toPandas()
    ds2_p= ds2_p.sample(15, ignore_index=True,random_state=42)
    ds2_count= ds2.count()
    
    return  ds2_p, ds2_count



###   Dataset join con Aidr_Classification  ###

df_combined = df_start.join(df3, on="tweet_id", how="inner")



# **********   Query **********



### Numero medio dei tweet per ogni ora del giorno ###

def mediaTweetPerOra():

    # Aggiungi una colonna "tweet_hour" per l'ora del giorno
    df_with_hour = df_start.withColumn("tweet_hour", hour("tweet_time"))


    # Calcola il conteggio medio dei tweet per ogni ora utilizzando una finestra
    window_spec = Window.partitionBy("tweet_date").orderBy("tweet_hour")
    hourly_tweet_avg = df_with_hour.groupBy("tweet_date", "tweet_hour").count()\
        .withColumn("avg_tweet_count", avg("count").over(window_spec))    
    
    # Rimuovi le righe con tweet_hour null
    hourly_tweet_avg = hourly_tweet_avg.filter(hourly_tweet_avg.tweet_hour.isNotNull())  
    hourly_tweet_avg = hourly_tweet_avg.groupBy("tweet_hour").agg(avg("count").alias("avg_tweet_count")).orderBy("tweet_hour")    
    hourly_tweet_avg = hourly_tweet_avg.toPandas()
    
    return hourly_tweet_avg


# Conteggio dei tweet verificati e non verificati

def verifTweet():    
    verified_tweet_count = df_start.groupBy("user_verified").count().toPandas()
    verified_tweet_count = verified_tweet_count.dropna(subset=["user_verified"])
    
    return verified_tweet_count

# Conteggio dei tweet per data
def dateTweet():    
    date_tweet_count = df_start.filter(col("tweet_date").isNotNull())
    date_tweet_count = date_tweet_count.withColumn("tweet_date", substring("tweet_date", 5, 10))    
    date_tweet_count = date_tweet_count.withColumn("tweet_date", to_date(date_tweet_count["tweet_date"], "MMM dd"))    
    date_tweet_count = date_tweet_count.withColumn("tweet_date", date_format(date_tweet_count["tweet_date"], "MM-dd"))    
    date_tweet_count = date_tweet_count.groupBy("tweet_date").count().orderBy("tweet_date")        

    return date_tweet_count


# Parole più frequenti nei tweet
def parolePiuFrequenti():    

    # stop words in inglese da NLTK
    nltk.download("stopwords")
    default_stop_words  = set(stopwords.words("english"))

    # elenco personalizzato di parole da eliminare
    custom_stop_words = ["rt", "https", "co", " ", "amp", "like", "1","go", "2", "3","5", "4","edt","fl","mph",
                          "d6vj7","ap","60", "xkcqz4s2ra","2qoqloege2", "kqsptnr4ox"]
    
    all_stop_words = default_stop_words.union(custom_stop_words)

    # testo in minuscolo
    words_df = df_start.select("text").withColumn("words", split(lower("text"), r'\W+'))

    # Espandere l'array di parole in singole righe ed escludere le stop words
    word_df = words_df.select(explode("words").alias("word")).filter(~col("word").isin(all_stop_words))
    word_df = word_df.filter(col("word") != "")

    # Calcolare le parole più frequenti
    word_counts = word_df.groupBy("word").count().orderBy(col("count").desc()).limit(20)
    
    word_counts_pandas = word_counts.toPandas()
    word_counts_pandas = word_counts_pandas.iloc[::-1]

    return word_counts_pandas
   
###  Count dei tweet per ogni tipologia   ###
def labelTweet():
    label_counts = df_combined.groupBy("AIDRLabel").count().orderBy(col("count").desc())
    label_count_pandas= label_counts.toPandas()
    label_count_pandas= label_count_pandas.iloc[::-1]    

    return label_count_pandas


###   parolepiù frequenti per ogni tipologia di tweet   ###
def paroleFrequentiPerTipologia():
    
    nltk.download("stopwords")    

    # estraggo text per ogni tipolofia
    label_not_related_or_irrelevant = df_combined.filter(col("AIDRLabel")=="not_related_or_irrelevant").select("text")    
    label_donation_and_volunteering = df_combined.filter(col("AIDRLabel")=="donation_and_volunteering").select("text")
    label_relevant_information = df_combined.filter(col("AIDRLabel")=="relevant_information").select("text")
    label_sympathy_and_support = df_combined.filter(col("AIDRLabel")=="sympathy_and_support").select("text")
    label_personal = df_combined.filter(col("AIDRLabel")=="personal").select("text")
    label_caution_and_advice = df_combined.filter(col("AIDRLabel")=="caution_and_advice").select("text")
    label_infrastructure_and_utilities_damage= df_combined.filter(col("AIDRLabel")=="infrastructure_and_utilities_damage").select("text")
    label_injured_or_dead_people= df_combined.filter(col("AIDRLabel")=="injured_or_dead_people").select("text")
    label_affected_individual = df_combined.filter(col("AIDRLabel")=="affected_individual").select("text")
    label_missing_and_found_people = df_combined.filter(col("AIDRLabel")=="missing_and_found_people").select("text")
    label_response_efforts = df_combined.filter(col("AIDRLabel")=="response_efforts").select("text")
    label_displaced_and_evacuations = df_combined.filter(col("AIDRLabel")=="displaced_and_evacuations").select("text")

    def parolePiuFrequenti(df):      
        
        default_stop_words  = set(stopwords.words("english"))
        custom_stop_words = ["rt", "https", "co", " ", "amp", "like", "1","go", "2", "3","5", "4","25", "21", "25th", "21st", "26th", "com","26", "54","190", "b",
                              "000", "u","edt","fl","mph", "d6vj7","daca","ap","60", "xkcqz4s2ra","2qoqloege2", "kqsptnr4ox", "rumqxwyfkb"]
        all_stop_words = default_stop_words.union(custom_stop_words)
        words_df = df.select("text").withColumn("words", split(lower("text"), r'\W+'))
        word_df = words_df.select(explode("words").alias("word")).filter(~col("word").isin(all_stop_words))
        word_df = word_df.filter(col("word") != "")        
        word_counts = word_df.groupBy("word").count().orderBy(col("count").desc()).limit(20)
        word_counts_pandas = word_counts.toPandas()
        word_counts_pandas = word_counts_pandas.iloc[::-1]
        return word_counts_pandas
    
    # calcolo parole più frequenti
    label_not_related_or_irrelevant= parolePiuFrequenti(label_not_related_or_irrelevant)   
    label_donation_and_volunteering= parolePiuFrequenti(label_donation_and_volunteering)
    label_relevant_information = parolePiuFrequenti(label_relevant_information)
    label_sympathy_and_support = parolePiuFrequenti(label_sympathy_and_support)
    label_personal = parolePiuFrequenti(label_personal)
    label_caution_and_advice = parolePiuFrequenti(label_caution_and_advice)
    label_infrastructure_and_utilities_damage= parolePiuFrequenti(label_infrastructure_and_utilities_damage)
    label_injured_or_dead_people = parolePiuFrequenti(label_injured_or_dead_people)
    label_affected_individual = parolePiuFrequenti(label_affected_individual)
    label_missing_and_found_people = parolePiuFrequenti(label_missing_and_found_people)
    label_response_efforts = parolePiuFrequenti(label_response_efforts)
    label_displaced_and_evacuations = parolePiuFrequenti(label_displaced_and_evacuations)

    return label_not_related_or_irrelevant,label_donation_and_volunteering,\
        label_relevant_information,label_sympathy_and_support,\
        label_personal,label_caution_and_advice,label_infrastructure_and_utilities_damage,\
        label_injured_or_dead_people,label_affected_individual,label_missing_and_found_people,\
        label_response_efforts,label_displaced_and_evacuations


### hashtag più frequenti ###
def hashtagPiuFrequenti():
    
    #testo in minuscolo
    hash_df = df_start.select(explode("hashtags").alias("hashtag"))

    #split
    hash_df = hash_df.withColumn("words", split(lower(col("hashtag")), ","))

    #singole parole
    hash_df = hash_df.select(explode("words").alias("word"))

    # Calcolo gli hashtag più frequenti
    word_counts_h = hash_df.groupBy("word").count().orderBy(col("count").desc()).limit(15)
    word_counts_pandas_h = word_counts_h.toPandas()
    word_counts_pandas_h = word_counts_pandas_h.iloc[::-1]
    

    return word_counts_pandas_h


### tweet geolocalizzati ###
def geoTweetBello():
    dfLocalizzati = df_start.filter(col("tweet_pos").isNotNull())

    # Converti le coordinate dei tweet in coordinate Basemap
    tweet_lons = dfLocalizzati.select(col("tweet_pos")[1].alias("longitude")).toPandas()["longitude"].tolist()
    tweet_lats = dfLocalizzati.select(col("tweet_pos")[0].alias("latitude")).toPandas()["latitude"].tolist()

    # Simuliamo alcune posizioni di danni causati dall'uragano (dati fittizi)
    damage_locations = [
        {"latitude": 24.4, "longitude": -93.6, "description": "Damage 1"},
        {"latitude": 25.0, "longitude": -94.4, "description": "Damage 2"},
        {"latitude": 25.6, "longitude": -95.1, "description": "Damage 3"},
        {"latitude": 26.3, "longitude": -95.8, "description": "Damage 4"},
        {"latitude": 27.1, "longitude": -96.3, "description": "Damage 5"},
        {"latitude": 27.8, "longitude": -96.8, "description": "Damage 6"},
        {"latitude": 28.0, "longitude": -96.9, "description": "Damage 7"},
        {"latitude": 28.2, "longitude": -97.1, "description": "Damage 8"},
        {"latitude": 28.7, "longitude": -97.3, "description": "Damage 9"},
        {"latitude": 29.0, "longitude": -97.5, "description": "Damage 10"},
        {"latitude": 29.2, "longitude": -97.4, "description": "Damage 11"},
        {"latitude": 29.3, "longitude": -97.6, "description": "Damage 12"},
        {"latitude": 29.1, "longitude": -97.5, "description": "Damage 13"},
        {"latitude": 29.0, "longitude": -97.2, "description": "Damage 14"},
        {"latitude": 28.8, "longitude": -96.8, "description": "Damage 15"},
        {"latitude": 28.6, "longitude": -96.5, "description": "Damage 16"},
        {"latitude": 28.5, "longitude": -96.2, "description": "Damage 17"},
        {"latitude": 28.4, "longitude": -95.9, "description": "Damage 18"},
        {"latitude": 28.2, "longitude": -95.4, "description": "Damage 19"},
        {"latitude": 28.1, "longitude": -95.0, "description": "Damage 20"},
        {"latitude": 28.2, "longitude": -94.6, "description": "Damage 21"},
        {"latitude": 28.5, "longitude": -94.2, "description": "Damage 22"},
        {"latitude": 28.9, "longitude": -93.8, "description": "Damage 23"},
        {"latitude": 29.4, "longitude": -93.6, "description": "Damage 24"},
        {"latitude": 29.8, "longitude": -93.5, "description": "Damage 25"},
        {"latitude": 30.1, "longitude": -93.4, "description": "Damage 26"},
        {"latitude": 30.6, "longitude": -93.1, "description": "Damage 27"},
        {"latitude": 31.3, "longitude": -92.6, "description": "Damage 28"},
        {"latitude": 31.9, "longitude": -92.2, "description": "Damage 29"},
        {"latitude": 32.5, "longitude": -91.7, "description": "Damage 30"},
        {"latitude": 33.4, "longitude": -90.9, "description": "Damage 31"},
        {"latitude": 34.1, "longitude": -89.6, "description": "Damage 32"},
    ]

    # Creare DataFrames separati per i tweet e i punti di danno
    tweet_data = pd.DataFrame({
        "lat": tweet_lats,
        "lon": tweet_lons,
        "type": ["Tweet"] * len(tweet_lats),
        "color": [(0, 0, 255)] * len(tweet_lats)  # Blu per i tweet
    })

    damage_data = pd.DataFrame({
        "lat": [location["latitude"] for location in damage_locations],
        "lon": [location["longitude"] for location in damage_locations],
        "type": ["Damage"] * len(damage_locations),
        "color": [(255, 0, 0)] * len(damage_locations)  # Rosso per i punti di danno
    })

    # Unire i due DataFrames in uno solo
    combined_data = pd.concat([tweet_data, damage_data], ignore_index=True)

    
    return combined_data


### Count dei tweet neei Paesi più colpiti (Texas, Luisiana) ###
def posColpite():
    most=df_combined.filter(trim(col("tweet_place")).isNotNull())
    most = most.withColumn("nation", split(most["tweet_place"], ",")[1])
    filter_state = most.filter((col("nation").contains("TX")) | (col("nation").contains("LA")))

    filter_state = filter_state.groupBy("nation").count().orderBy("count")
    filter_state = filter_state.filter(filter_state["nation"].startswith(" "))
    filter_state = filter_state.toPandas()

    return filter_state


# **********   Query con Machine Learning  **********



### Calcolo Serntiment dei tweet ###

def sentiment_Vader():

    nltk.download("vader_lexicon")

    # Inizializza il SentimentIntensityAnalyzer di NLTK
    sia = SentimentIntensityAnalyzer()

    tweets = df_start.select("text")
    tweets = tweets.filter(col("text").isNotNull())

    # Definizione della funzione UDF per analizzare il sentiment dei tweet
    def analyze_sentiment_vader(text):
        sentiment_scores = sia.polarity_scores(text)
        
        # Assegna il sentiment in base al compound score di VADER
        if sentiment_scores['compound'] >= 0.05:
            sentiment = 'positive'
        elif sentiment_scores['compound'] <= -0.05:
            sentiment = 'negative'
        else:
            sentiment = 'neutral'
        
        return sentiment, sentiment_scores['compound']

    
    analyze_sentiment_vader_udf = udf(analyze_sentiment_vader, StructType([
        StructField("sentiment", StringType()),
        StructField("compound", DoubleType())
    ]))

    # Applicazione della funzione UDF ai tweet per ottenere la colonna del sentiment
    tweets_with_sentiment_vader = tweets.withColumn("sentiment-score", analyze_sentiment_vader_udf(col("text")))
    tweets_with_sent_final = tweets_with_sentiment_vader.withColumn("sentiment", col("sentiment-score.sentiment"))
    tweets_with_sent_final = tweets_with_sent_final.withColumn("score", col("sentiment-score.compound"))    
    tweets_with_sent_final = tweets_with_sent_final.drop("sentiment-score")  

    # Calcola la distribuzione dei sentiment
    sentiment_counts = tweets_with_sent_final.groupBy("sentiment").count().orderBy(col("count").desc()).collect()

    # Sample per ogni tipo di sentiment
    t_pos= tweets_with_sent_final.filter(col("sentiment") == "positive").limit(5)
    t_neg= tweets_with_sent_final.filter(col("sentiment") == "negative").limit(5)
    t_neu= tweets_with_sent_final.filter(col("sentiment") == "neutral").limit(5)
    t_pos = t_pos.toPandas()
    t_neg = t_neg.toPandas()
    t_neu = t_neu.toPandas()   

    return sentiment_counts, t_pos, t_neg, t_neu


###  Calcolo sentiment dei paesi più colpiti   ###
def sentimentPosColpite():
   
    nltk.download("vader_lexicon")
    sia = SentimentIntensityAnalyzer()
    
    tweets = df_combined.filter(col("text").isNotNull())
    filter_state = tweets.filter(trim(col("tweet_place")).isNotNull())
    filter_state = filter_state.withColumn("nation", split(filter_state["tweet_place"], ",")[1])
    filter_state = filter_state.filter((col("nation").contains("TX")) | (col("nation").contains("LA")))   

    def analyze_sentiment_vader(text):
        sentiment_scores = sia.polarity_scores(text)
        
        # Assegna un colore in base al compound score di VADER
        if sentiment_scores['compound'] >= 0.05:
            sentiment = 'positive'
        elif sentiment_scores['compound'] <= -0.05:
            sentiment = 'negative'
        else:
            sentiment = 'neutral'
        
        return sentiment, sentiment_scores['compound']    
    
    analyze_sentiment_vader_udf = udf(analyze_sentiment_vader, StructType([
        StructField("sentiment", StringType()),
        StructField("compound", DoubleType())
    ]))    

    #Applicazione della funzione UDF ai tweet per ottenere la colonna del sentiment
    tweets_with_sentiment_vader = filter_state.withColumn("sentiment-score", analyze_sentiment_vader_udf(col("text")))
    tweets_with_sent_final = tweets_with_sentiment_vader.withColumn("sentiment", col("sentiment-score.sentiment"))
    tweets_with_sent_final = tweets_with_sent_final.withColumn("score", col("sentiment-score.compound"))
    tweets_with_sent_final = tweets_with_sent_final.drop("sentiment-score")
    sentiment_counts = tweets_with_sent_final.groupBy("sentiment").count().orderBy(col("count").desc()).collect()
    
    # Sample per ogni tipo di sentiment
    tweets_with_sent_final= tweets_with_sent_final.select("text","sentiment","score")
    t_pos= tweets_with_sent_final.filter(col("sentiment") == "positive").limit(5)
    t_pos = t_pos.toPandas()    
    t_neg = tweets_with_sent_final.filter(col("sentiment") == "negative").limit(5)
    t_neg = t_neg.toPandas()
    t_neu = tweets_with_sent_final.filter(col("sentiment") == "neutral").limit(5)
    t_neu = t_neu.toPandas()

    return sentiment_counts, t_pos, t_neg, t_neu
    

###   Calcolo Matrice di correlazione   ###
def correlazione():

    pre_enc= df_combined. select("user_followers","user_verified","AIDRLabel", "AIDRConfidence")
    pre_enc = pre_enc.toPandas()
    
    # Encoding delle feature
    indexed_df = df_combined.withColumn("user_verified_index", when(col("user_verified") == "true", 1).otherwise(0))
    indexed_df = indexed_df.withColumn("AIDRConfidence_numeric", col("AIDRConfidence").cast("double"))
    indexer = StringIndexer(inputCol="AIDRLabel", outputCol="AIDRLabel_index")
    indexed_df = indexer.fit(indexed_df).transform(indexed_df)   

    # Seleziona le colonne rilevanti per il calcolo della correlazione
    selected_cols = ["user_followers", "AIDRLabel_index", "AIDRConfidence_numeric","user_verified_index"]

    # VectorAssembler per creare un vettore di features
    assembler = VectorAssembler(inputCols=selected_cols, outputCol="features")
    assembled_df = assembler.transform(indexed_df)

    post_enc= assembled_df. select("user_followers","user_verified_index", "AIDRLabel_index", "AIDRConfidence_numeric")
    post_enc = post_enc.toPandas()    

    # Calcola la matrice di correlazione
    correlation_matrix = Correlation.corr(assembled_df, "features").head()
    corr_matrix = correlation_matrix[0].toArray()
    corr_matrix_df = spark.createDataFrame(corr_matrix.tolist(), selected_cols)
    corr_matrix_df = pd.DataFrame(corr_matrix, columns=selected_cols)
    corr_matrix_pearson = corr_matrix_df.corr()

    return corr_matrix_pearson, pre_enc, post_enc

###   Classificazione con Random Forest  ###
def classification_RandomForest():

    # Encoding delle feature
    indexed_df = df_combined.withColumn("user_verified_index", when(col("user_verified") == "true", 1).otherwise(0))
    indexed_df = indexed_df.withColumn("AIDRConfidence_numeric", col("AIDRConfidence").cast("double"))
    indexer = StringIndexer(inputCol="AIDRLabel", outputCol="AIDRLabel_index")
    indexed_df = indexer.fit(indexed_df).transform(indexed_df)
    
    # Seleziona le colonne rilevanti   
    selected_cols = ["user_followers", "AIDRLabel_index", "AIDRConfidence_numeric","user_verified_index"] 
    df_class= indexed_df.select(selected_cols)   
    
    #VectorAssembler per creare un vettore di features
    assembler = VectorAssembler(inputCols=selected_cols, outputCol="features")
    assembled_df = assembler.transform(df_class)
    
    # Split dataset in training, validation, test sets
    train_data, val_data, test_data = assembled_df.randomSplit([0.6, 0.2, 0.2], seed=123)

    # Calcolo distribuzione etichette
    assembled_df_labels = assembled_df.groupBy("AIDRLabel_index").count().orderBy("AIDRLabel_index")      
    assembled_df_labels = assembled_df_labels.toPandas() 

    # Initializzaione RandomForestClassifier
    rf_classifier = RandomForestClassifier(featuresCol="features", labelCol="AIDRLabel_index", numTrees=100, seed=123)

    # Train su training data
    rf_model = rf_classifier.fit(train_data)

    # Predizione su validation set
    val_predictions = rf_model.transform(val_data)

    # Valutazione
    evaluator = MulticlassClassificationEvaluator(labelCol="AIDRLabel_index", predictionCol="prediction", metricName="accuracy")

    # Accuracy
    val_accuracy = evaluator.evaluate(val_predictions)    

    # Predizione su test set
    test_predictions = rf_model.transform(test_data)   

    # Accuracy su test set
    test_accuracy = evaluator.evaluate(test_predictions)   
   
    # DF con predizioni        
    test_predictions_and_labels_rdd = test_predictions.select("prediction", "AIDRLabel_index").rdd     

    metrics = MulticlassMetrics(test_predictions_and_labels_rdd)

    labels = test_predictions_and_labels_rdd.map(lambda x: x[1]).distinct().collect()   

    metrics_data = []

    label_to_index = {
    'not_related_or_irrelevant': 0,
    'donation_and_volunteering':1,
    'relevant_information':2,
    'sympathy_and_support': 3,
    'personal':4,
    'caution_and_advice':5,
    'infrastructure_and_utilities_damage':6,
    'injured_or_dead_people':7,
    'affected_individual':8,
    'missing_and_found_people':9,
    'response_efforts':10     
    }


    for label in labels:
        label_precision = metrics.precision(label)
        label_recall = metrics.recall(label)
        label_f1_score = metrics.fMeasure(label)
       
        label_name = [key for key, value in label_to_index.items() if value == label][0]

        metrics_data.append({"Label": label_name, "Precision": label_precision, "Recall": label_recall, "F1-Score": label_f1_score})

    metrics_df = pd.DataFrame(metrics_data, columns=["Label", "Precision", "Recall", "F1-Score"])    
 
    # Calcolo delle metriche generali
    overall_precision = metrics.weightedPrecision
    overall_recall = metrics.weightedRecall
    overall_accuracy = metrics.accuracy
    

    # Creazione del DataFrame per le metriche generali
    overall_metrics_df = pd.DataFrame({
        "Metrica": ["Precision", "Recall", "Accuracy"],
        "Valore": [overall_precision, overall_recall, overall_accuracy]
    })


    return val_accuracy, test_accuracy,overall_metrics_df, assembled_df_labels, metrics_df



###   Classificazione con Logistic regretion  ###

def classification_LogReg():

    # Encoding delle feature
    indexed_df = df_combined.withColumn("user_verified_index", when(col("user_verified") == "true", 1).otherwise(0))
    indexed_df = indexed_df.withColumn("AIDRConfidence_numeric", col("AIDRConfidence").cast("double"))
    indexer = StringIndexer(inputCol="AIDRLabel", outputCol="AIDRLabel_index")
    indexed_df = indexer.fit(indexed_df).transform(indexed_df)
    
    # Seleziona le colonne rilevanti   
    selected_cols = ["user_followers", "AIDRLabel_index", "AIDRConfidence_numeric","user_verified_index"] 
    df_class= indexed_df.select(selected_cols)   
    
    # VectorAssembler per creare un vettore di features
    assembler = VectorAssembler(inputCols=selected_cols, outputCol="features")
    assembled_df = assembler.transform(df_class)
    
    # Split dataset in training, validation, test sets
    train_data, val_data, test_data = assembled_df.randomSplit([0.6, 0.2, 0.2], seed=123)

    # Inizializzazione LogisticRegression
    lr_classifier = LogisticRegression(featuresCol="features", labelCol="AIDRLabel_index")

    # Creazione del pipeline
    pipeline = Pipeline(stages=[lr_classifier])

    # Addestramento del modello su training data
    lr_model = pipeline.fit(train_data)

    # Predizione su validation set
    val_predictions = lr_model.transform(val_data)

    # Valutazione
    evaluator = MulticlassClassificationEvaluator(labelCol="AIDRLabel_index", predictionCol="prediction", metricName="accuracy")

    # Accuracy su validation set
    val_accuracy = evaluator.evaluate(val_predictions)    

    # Predizione su test set
    test_predictions = lr_model.transform(test_data)   

    # Accuracy su test set
    test_accuracy = evaluator.evaluate(test_predictions)   

    # # DF con predizioni     
   
    label_to_index = {
    'not_related_or_irrelevant': 0,
    'donation_and_volunteering':1,
    'relevant_information':2,
    'sympathy_and_support': 3,
    'personal':4,
    'caution_and_advice':5,
    'infrastructure_and_utilities_damage':6,
    'injured_or_dead_people':7,
    'affected_individual':8,
    'missing_and_found_people':9,
    'response_efforts':10     
    }

   
    test_predictions_and_labels_rdd = test_predictions.select("prediction", "AIDRLabel_index").rdd     

    metrics = MulticlassMetrics(test_predictions_and_labels_rdd)

    labels = test_predictions_and_labels_rdd.map(lambda x: x[1]).distinct().collect()   

    metrics_data = []

    for label in labels:
        label_precision = metrics.precision(label)
        label_recall = metrics.recall(label)
        label_f1_score = metrics.fMeasure(label)
       
        label_name = [key for key, value in label_to_index.items() if value == label][0]

        metrics_data.append({"Label": label_name, "Precision": label_precision, "Recall": label_recall, "F1-Score": label_f1_score})

    metrics_df = pd.DataFrame(metrics_data, columns=["Label", "Precision", "Recall", "F1-Score"])  

    # Calcolo delle metriche generali
    overall_precision = evaluator.evaluate(test_predictions, {evaluator.metricName: "weightedPrecision"})
    overall_recall = evaluator.evaluate(test_predictions, {evaluator.metricName: "weightedRecall"})
    overall_accuracy = evaluator.evaluate(test_predictions, {evaluator.metricName: "accuracy"})
    
    # Creazione del DataFrame per le metriche generali
    overall_metrics_df = pd.DataFrame({
        "Metrica": ["Precision", "Recall", "Accuracy"],
        "Valore": [overall_precision, overall_recall, overall_accuracy]
    })
    
    # Calcolo della distribuzione delle etichette
    assembled_df_labels = assembled_df.groupBy("AIDRLabel_index").count().orderBy("AIDRLabel_index")      
    assembled_df_labels = assembled_df_labels.toPandas()

    return val_accuracy, test_accuracy, overall_metrics_df, assembled_df_labels,metrics_df

