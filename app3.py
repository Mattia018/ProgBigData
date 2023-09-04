import streamlit as st
import plotly.express as px
from main2 import *
import pydeck as pdk
import plotly.graph_objects as go
import numpy as np

if 'visualizations' not in st.session_state:
    st.session_state.visualizations = {}

st.set_page_config(
    page_title="Visualizzation Big Data",
    layout="wide",
    page_icon="📈"
)

hide_streamlit_style = """
            <style>

            footer {visibility: hidden;}
            
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)


def create_button(title, key):
    button = st.button(title, key=key)
    return button


pre_button = None

with st.container():
    st.markdown("<h1 style='text-align: center; font-family: Arial, sans-serif; font-size: 46px;'>\
                Progetto Big Data </h1>\
                <h2 style='text-align: center; font-family: Arial, sans-serif; '>\
                Autori: Mattia Presta, Silvio Raso </h2>"

                , unsafe_allow_html=True)

    st.divider()

    st.markdown("<h3 style=' font-family: Arial, sans-serif;'>\
                Dataset : Harvey Data 2017, Harvey_Data_2017_aidr_classification </h3>\
                <p style='font-family: Arial, sans-serif; font-size: 20px;'>\
                <b style=' font-size: 20px;' >Harvey Data 2017 </b> contiene dati Twitter Json relativi all' uragano harvey con più di 3 milioni di tuple a partire dal giono 27 Agosto 2017 fino a 19 Settembre 2017,\
                circa 15 giorni dopo il picco massimo del fenomeno </p>\
                <p style='font-family: Arial, sans-serif; font-size: 20px;'>\
                <b style=' font-size: 20px;'>Harvey_Data_2017_aidr_classification</b> contiene dati Twitter relativi all' uragano harvey con l'aggiunta della classificazione \
                in 12 etichette diverse che esprimono la rilevanza e la tipologia del tweet </p>"

                , unsafe_allow_html=True)

    image_1_url = "https://i.imgur.com/znpTHMR.png"
    image_2_url = "https://imgur.com/CeC5QN4.png"

    st.divider()

    st.image(image_1_url, use_column_width=True)

    st.divider()

    st.image(image_2_url, use_column_width=True)

    st.divider()

###  Dataset 1   ###
st.title("_Harvey Data 2017_")

if 'ds1_p_chart' not in st.session_state.visualizations:
    ds1_p, ds1_count = dataset_start()

    st.session_state.visualizations['ds1_p_chart'], st.session_state.visualizations[
        'ds1_count_chart'] = ds1_p, ds1_count

with st.expander("Data Frame"):
    st.dataframe(st.session_state.visualizations['ds1_p_chart'], use_container_width=True, hide_index=True)
    st.divider()
    st.markdown(f"_Size_ : :blue[{st.session_state.visualizations['ds1_count_chart']}]")

st.divider()

###  Dataset 2   ###
st.title("_Harvei_Data_2017_aidr_classification_")

if 'ds2_p_chart' not in st.session_state.visualizations:
    ds2_p, ds2_count = dataset_aidr_classification()
    st.session_state.visualizations['ds2_p_chart'], st.session_state.visualizations[
        'ds2_count_chart'] = ds2_p, ds2_count
    pre_button = True

with st.expander("Data Frame"):
    st.dataframe(st.session_state.visualizations['ds2_p_chart'], use_container_width=True, hide_index=True)
    st.divider()
    st.markdown(f"_Size_ : :blue[{st.session_state.visualizations['ds2_count_chart']}]")

st.divider()

###  date   ###
st.title("Tweet per data")

date_button = create_button("Visualizza grafico", key="date_button")

if date_button or pre_button:
    filtered_date_tweet_count = dateTweet()
    st.session_state.visualizations['date_chart'] = filtered_date_tweet_count

if 'date_chart' in st.session_state.visualizations:
    st.bar_chart(
        st.session_state.visualizations['date_chart'],
        x="tweet_date",
        y="count",
    )

st.divider()

###  Count medio dei tweet per ogni ora del giono   ###

st.title("Numero medio di Tweet per ogni ora del giono")

ora_button = create_button("Visualizza grafico", key="ora_button")

if ora_button or pre_button:
    avg_per_ora = mediaTweetPerOra()
    st.session_state.visualizations['avg_per_ora_chart'] = avg_per_ora

if 'avg_per_ora_chart' in st.session_state.visualizations:
    fig = px.line(st.session_state.visualizations['avg_per_ora_chart'],
                  x="tweet_hour",
                  y="avg_tweet_count",
                  text="tweet_hour",
                  markers=True,
                  )
    fig.update_traces(textposition="top center")
    fig.update_xaxes(title="Ora del Giorno")
    fig.update_yaxes(title="Media dei Tweet")
    st.plotly_chart(fig, use_container_width=True, showlegend=False)

st.divider()

###  verif   ####

st.title("Tweet di utenti verificati")

verif_button = create_button("Visualizza grafico", key="verif_button")

if verif_button or pre_button:
    verif_tweet_data = verifTweet()
    st.session_state.visualizations['verif_chart'] = verif_tweet_data

if 'verif_chart' in st.session_state.visualizations:
    fig = px.pie(
        st.session_state.visualizations['verif_chart'],
        values="count",
        color_discrete_sequence=px.colors.sequential.Darkmint,
        names="user_verified"
    )
    st.plotly_chart(fig, use_container_width=True)

st.divider()

###   parole più frequenti  ###

st.title("Parole più frequenti nei Tweet")

parole_button = create_button("Visualizza grafico", key="parole_button")
parole_data = None

if parole_button or pre_button:
    parole_data = parolePiuFrequenti()
    st.session_state.visualizations['parole_chart'] = parole_data

if 'parole_chart' in st.session_state.visualizations:
    fig = px.bar(
        st.session_state.visualizations['parole_chart'],
        x="count",
        y="word",
        orientation="h",
        height=600,
        title="Top 20 parole più frequenti nei Tweets"
    )
    st.plotly_chart(fig, use_container_width=True)

st.divider()
###   hashtag più frequenti  ###

st.title("HashTag più frequenti nei Tweet")

hash_button = create_button("Visualizza grafico", key="hash_button")
hash_data = None

if hash_button or pre_button:
    hash_data = hashtagPiuFrequenti()
    st.session_state.visualizations['hash_chart'] = hash_data

if 'hash_chart' in st.session_state.visualizations:
    fig = px.bar(
        st.session_state.visualizations['parole_chart'],
        x="count",
        y="word",
        orientation="h",
        height=600,
        title="Top 20 Hashtag più frequenti nei Tweets"
    )
    st.plotly_chart(fig, use_container_width=True)

st.divider()
###   label aidr  ###

st.title("Tipologia Tweet ")

label_button = create_button("Visualizza grafico", key="label_button")

if label_button or pre_button:
    label_data = labelTweet()
    st.session_state.visualizations['label_chart'] = label_data

if 'label_chart' in st.session_state.visualizations:
    # st.bar_chart(
    #     st.session_state.visualizations['label_chart'],
    #     x="AIDRLabel",
    #     y="count",
    #     height=600
    # )

    fig = px.bar(
            st.session_state.visualizations['label_chart'],
            x="AIDRLabel",
            y="count",   
        )
    
    st.plotly_chart(fig, use_container_width=True)

st.divider()
###   parole più frequenti per tipologia tweet  ###

st.title("Parole più Frequenti per Tipologia Tweet ")

label_word_button = create_button("Visualizza grafico", key="label_word_button")

if label_word_button or pre_button:
    label_not_related_or_irrelevant, label_donation_and_volunteering, \
        label_relevant_information, label_sympathy_and_support, \
        label_personal, label_caution_and_advice, label_infrastructure_and_utilities_damage, \
        label_injured_or_dead_people, label_affected_individual, label_missing_and_found_people, \
        label_response_efforts, label_displaced_and_evacuations = paroleFrequentiPerTipologia()

    st.session_state.visualizations['label_not_related_or_irrelevant_chart'] = label_not_related_or_irrelevant
    st.session_state.visualizations['label_donation_and_volunteering_chart'] = label_donation_and_volunteering
    st.session_state.visualizations['label_relevant_information_chart'] = label_relevant_information
    st.session_state.visualizations['label_sympathy_and_support_chart'] = label_sympathy_and_support
    st.session_state.visualizations['label_personal_chart'] = label_personal
    st.session_state.visualizations['label_caution_and_advice_chart'] = label_caution_and_advice
    st.session_state.visualizations[
        'label_infrastructure_and_utilities_damage_chart'] = label_infrastructure_and_utilities_damage
    st.session_state.visualizations['label_injured_or_dead_people_chart'] = label_injured_or_dead_people
    st.session_state.visualizations['label_affected_individual_chart'] = label_affected_individual
    st.session_state.visualizations['label_missing_and_found_people_chart'] = label_missing_and_found_people
    st.session_state.visualizations['label_response_efforts_chart'] = label_response_efforts
    st.session_state.visualizations['label_displaced_and_evacuations_chart'] = label_displaced_and_evacuations

if 'label_not_related_or_irrelevant_chart' in st.session_state.visualizations:
    tab1, tab2, tab3, tab4 = st.tabs(["Not related or irrelevant", "Donation and volunteering",
                                      "Relevant information", "Sympathy and support"])

    with tab1:
        fig = px.bar(
            st.session_state.visualizations['label_not_related_or_irrelevant_chart'],
            x="count",
            y="word",
            orientation="h",
            height=600,
            title="Top 20 parole più frequenti nei Tweets : Not related or irrelevant",

        )

        fig.update_traces(marker_color='blue')
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        fig = px.bar(
            st.session_state.visualizations['label_donation_and_volunteering_chart'],
            x="count",
            y="word",
            orientation="h",
            height=600,
            title="Top 20 parole più frequenti nei Tweets : Donation and volunteering",

        )
        fig.update_traces(marker_color='blue')
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        fig = px.bar(
            st.session_state.visualizations['label_relevant_information_chart'],
            x="count",
            y="word",
            orientation="h",
            height=600,
            title="Top 20 parole più frequenti nei Tweets : Relevant information",

        )
        fig.update_traces(marker_color='blue')
        st.plotly_chart(fig, use_container_width=True)

    with tab4:
        fig = px.bar(
            st.session_state.visualizations['label_sympathy_and_support_chart'],
            x="count",
            y="word",
            orientation="h",
            height=600,
            title="Top 20 parole più frequenti nei Tweets : Sympathy and support",

        )
        fig.update_traces(marker_color='blue')
        st.plotly_chart(fig, use_container_width=True)

    tab5, tab6, tab7, tab8 = st.tabs(["Personal", "Caution and advice", "Infrastructure and utilities damage",
                                      "Injured or dead people"])

    with tab5:
        fig = px.bar(
            st.session_state.visualizations['label_personal_chart'],
            x="count",
            y="word",
            orientation="h",
            height=600,
            title="Top 20 parole più frequenti nei Tweets : Personal",

        )
        fig.update_traces(marker_color='cornflowerblue')
        st.plotly_chart(fig, use_container_width=True)

    with tab6:
        fig = px.bar(
            st.session_state.visualizations['label_caution_and_advice_chart'],
            x="count",
            y="word",
            orientation="h",
            height=600,
            title="Top 20 parole più frequenti nei Tweets : Caution and advice",

        )
        fig.update_traces(marker_color='cornflowerblue')
        st.plotly_chart(fig, use_container_width=True)

    with tab7:
        fig = px.bar(
            st.session_state.visualizations['label_infrastructure_and_utilities_damage_chart'],
            x="count",
            y="word",
            orientation="h",
            height=600,
            title="Top 20 parole più frequenti nei Tweets : Infrastructure and utilities damage",

        )
        fig.update_traces(marker_color='cornflowerblue')
        st.plotly_chart(fig, use_container_width=True)

    with tab8:
        fig = px.bar(
            st.session_state.visualizations['label_injured_or_dead_people_chart'],
            x="count",
            y="word",
            orientation="h",
            height=600,
            title="Top 20 parole più frequenti nei Tweets : Injured or dead people",

        )
        fig.update_traces(marker_color='cornflowerblue')
        st.plotly_chart(fig, use_container_width=True)

    tab9, tab10, tab11, tab12 = st.tabs(["Affected individual", "Missing and found people",
                                         "Response efforts", "Displace and evacuations"])

    with tab9:
        fig = px.bar(
            st.session_state.visualizations['label_affected_individual_chart'],
            x="count",
            y="word",
            orientation="h",
            height=600,
            title="Top 20 parole più frequenti nei Tweets : Affected individual",

        )
        fig.update_traces(marker_color='darkcyan')
        st.plotly_chart(fig, use_container_width=True)

    with tab10:
        fig = px.bar(
            st.session_state.visualizations['label_missing_and_found_people_chart'],
            x="count",
            y="word",
            orientation="h",
            height=600,
            title="Top 20 parole più frequenti nei Tweets : Missing and found people",

        )
        fig.update_traces(marker_color='darkcyan')
        st.plotly_chart(fig, use_container_width=True)

    with tab11:
        fig = px.bar(
            st.session_state.visualizations['label_response_efforts_chart'],
            x="count",
            y="word",
            orientation="h",
            height=600,
            title="Top 20 parole più frequenti nei Tweets : Response efforts",

        )
        fig.update_traces(marker_color='darkcyan')
        st.plotly_chart(fig, use_container_width=True)

    with tab12:
        fig = px.bar(
            st.session_state.visualizations['label_displaced_and_evacuations_chart'],
            x="count",
            y="word",
            orientation="h",
            height=600,
            title="Top 20 parole più frequenti nei Tweets : Displace and evacuations",

        )
        fig.update_traces(marker_color='darkcyan')
        st.plotly_chart(fig, use_container_width=True)

st.divider()
###   Posizioni più colpite  ###

st.title(" Numero Tweet degli Stati più colpiti ")

pos_button = create_button("Visualizza grafico", key="pos_button")

if pos_button or pre_button:
    pos_data = posColpite()
    st.session_state.visualizations['pos_chart'] = pos_data

if 'pos_chart' in st.session_state.visualizations:
    #  st.bar_chart(
    #         st.session_state.visualizations['pos_chart'],
    #         x="nation",
    #         y="count",
    #     )

    fig = px.pie(
        st.session_state.visualizations['pos_chart'],
        values="count",
        color_discrete_sequence=px.colors.sequential.dense,
        names="nation"
    )
    st.plotly_chart(fig, use_container_width=True)

st.divider()
###   tweet geolocalizzati  ###

st.title("Tweet Geolocalizzati")

geo_button = create_button("Visualizza grafico", key="geo_button")

if geo_button or pre_button:
    combined_data = geoTweetBello()
    st.session_state.visualizations['combined_chart'] = combined_data

if 'combined_chart' in st.session_state.visualizations:

    # Aggiungi un selettore per i layer
    selected_layers = st.multiselect('Seleziona i layer:', ['Tweet', 'Punti di Danno', 'Traiettoria', 'Tweet_Grid'])

    # Dividi i dati in tweet, punti di danno
    tweet_data = st.session_state.visualizations['combined_chart'][
        st.session_state.visualizations['combined_chart']['type'] == 'Tweet']
    damage_data = st.session_state.visualizations['combined_chart'][
        st.session_state.visualizations['combined_chart']['type'] == 'Damage']

    layers = []

    # Aggiungi il layer tweet se è selezionato
    if 'Tweet' in selected_layers:
        tweet_layer = pdk.Layer(
            'ScatterplotLayer',
            data=tweet_data,
            get_position='[lon, lat]',
            get_color='[0, 0, 255, 100]',  # Tweet in blu
            get_radius=2000,  # Imposta un raggio adeguato
            pickable=True,
            auto_highlight=True,
            filled=True,
            filled_color="[0, 0, 255, 255]",  # Colore blu per i tweet
            outline=True,
            outline_color="[0, 0, 255, 255]",  # Colore blu per i tweet
        )
        layers.append(tweet_layer)

    # Aggiungi il layer punti di danno se è selezionato
    if 'Punti di Danno' in selected_layers:
        damage_layer = pdk.Layer(
            'ScatterplotLayer',
            data=damage_data,
            get_position='[lon, lat]',
            get_color='[255, 0, 0, 100]',  # Punti di danno in rosso
            get_radius=5000,  # Imposta un raggio adeguato
            pickable=True,
            auto_highlight=True,
            filled=True,
            filled_color="[255, 0, 0, 255]",  # Colore rosso per i punti di danno
            outline=True,
            outline_color="[255, 0, 0, 255]",  # Colore rosso per i punti di danno
        )
        layers.append(damage_layer)

    # Aggiungi il layer degli aeroporti se è selezionato
    if 'Traiettoria' in selected_layers:
        # Crea una lista vuota per memorizzare i percorsi simulati
        paths = []

        # Itera attraverso i dati degli aeroporti per calcolare e creare i percorsi simulati
        for i in range(len(damage_data) - 1):
            start_point = [damage_data.iloc[i]['lon'], damage_data.iloc[i]['lat']]
            end_point = [damage_data.iloc[i + 1]['lon'], damage_data.iloc[i + 1]['lat']]

            path = {
                "start": start_point,
                "end": end_point,
            }

            paths.append(path)

        # Crea il livello di percorsi (linee) utilizzando pydeck.Layer
        line_layer = pdk.Layer(
            "LineLayer",
            paths,
            get_source_position="start",
            get_target_position="end",
            get_color=[255, 140, 0],
            get_width=5,
            highlight_color=[255, 255, 0],
            picking_radius=10,
            auto_highlight=True,
            pickable=True,
        )
        layers.append(line_layer)

    # Definisci lo stato di visualizzazione iniziale
    INITIAL_VIEW_STATE = pdk.ViewState(
        latitude=48.17,
        longitude=-24.27,
        zoom=4.5,
        max_zoom=16,
        pitch=30,
        bearing=0
    )

    # Crea il deck con i layer selezionati
    deck = pdk.Deck(layers=layers, initial_view_state=INITIAL_VIEW_STATE)

    # Visualizza il deck
    st.pydeck_chart(deck)

    # Salva il deck in un file HTML
    deck.to_html("deck.html")

st.divider()

###   Query in Machine Learning ###

st.title("Query con Machine Learning")

st.divider()

###   Sentiment Analysis  ###

st.title("Sentiment Analysis dei Tweet")

sent_button = create_button("Visualizza grafico", key="sent_button")

if sent_button or pre_button:
    sent_data, t_pos, t_neg, t_neu = sentiment_tre()
    sentiment_data = {
        "sentiment": [row["sentiment"] for row in sent_data],
        "count": [row["count"] for row in sent_data]
    }
    st.session_state.visualizations['sent_chart'] = sentiment_data
    st.session_state.visualizations['t_pos'] = t_pos
    st.session_state.visualizations['t_neg'] = t_neg
    st.session_state.visualizations['t_neu'] = t_neu

if 'sent_chart' in st.session_state.visualizations:
    tab1, tab2, tab3 = st.tabs(["Positive", "Negative", "Neutral"])

    with tab1:
        st.table(st.session_state.visualizations['t_pos'].head(5))

    with tab2:
        st.table(st.session_state.visualizations['t_neg'].head(5))

    with tab3:
        st.table(st.session_state.visualizations['t_neu'].head(5))

    st.divider()

    sentiment_colors = {
        "positive": "lightgreen",
        "neutral": "lightgray",
        "negative": "chocolate"
    }

    fig = px.bar(
        st.session_state.visualizations['sent_chart'],
        x="sentiment",
        y="count",
        color="sentiment",
        color_discrete_map=sentiment_colors,        
    )

    left, middle, right = st.columns((1, 5, 1))
    with middle:
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig)

st.divider()

###   Sentiment Analysis Posizioni più colpite ###
st.title("Sentiment Analysis dei Tweet dei Paesi più colpiti")

sent_pos_button = create_button("Visualizza grafico", key="sent_pos_button")

if sent_pos_button or pre_button:
    sent_pos_data, tu_pos, tu_neg, tu_neu = sentimentPosColpite()
    sentiment_pos_data = {
        "sentiment": [row["sentiment"] for row in sent_pos_data],
        "count": [row["count"] for row in sent_pos_data]
    }
    st.session_state.visualizations['sent_pos_chart'] = sentiment_pos_data
    st.session_state.visualizations['tu_pos'] = tu_pos
    st.session_state.visualizations['tu_neg'] = tu_neg
    st.session_state.visualizations['tu_neu'] = tu_neu

if 'sent_pos_chart' in st.session_state.visualizations:
    tab1, tab2, tab3 = st.tabs(["Positive", "Negative", "Neutral"])

    with tab1:
        st.table(st.session_state.visualizations['tu_pos'].head(5))

    with tab2:
        st.table(st.session_state.visualizations['tu_neg'].head(5))

    with tab3:
        st.table(st.session_state.visualizations['tu_neu'].head(5))

    st.divider()

    sentiment_colors = {
        "positive": "lightgreen",
        "neutral": "lightgray",
        "negative": "chocolate"
    }

    fig = px.bar(
        st.session_state.visualizations['sent_pos_chart'],
        x="sentiment",
        y="count",
        color="sentiment",
        color_discrete_map=sentiment_colors,        
    )

    left, middle, right = st.columns((1, 5, 1))
    with middle:
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width = True)

st.divider()

###   Predizione su AIDRLabel ###
st.title("Predizione su AIDRLabel")
st.divider()
st.title("Matrice di Correlazione ")

matrix_button = create_button("Visualizza matrice", key="matrix_button")

if matrix_button or pre_button:
    corr_matrix, pre_enc, post_enc = correlazione()

    st.session_state.visualizations['corr_matrix_chart'] = corr_matrix
    st.session_state.visualizations['pre_enc_chart'] = pre_enc
    st.session_state.visualizations['post_enc_chart'] = post_enc

if 'corr_matrix_chart' in st.session_state.visualizations:
    plt.figure(figsize=(10, 8))
    sns.heatmap(st.session_state.visualizations['corr_matrix_chart'], annot=True, cmap='RdBu', center=0)
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot(use_container_width=True)

    st.divider()
    st.markdown("Feature Pre Encoding")
    st.table(st.session_state.visualizations['pre_enc_chart'].head(5))

    st.divider()
    st.markdown("Feature Post Encoding")

    st.table(st.session_state.visualizations['post_enc_chart'].head(5))

st.divider()
st.title("Classificazione con Random Forest")

class_button = create_button("Visualizza Classificazione", key="class_button")

if class_button or pre_button:
    val_accuracy, test_accuracy, metrics_df, overall_metrics_df, assembled_df_labels, metrics_df_label = classification()

    st.session_state.visualizations['val_accuracy_chart'] = val_accuracy
    st.session_state.visualizations['test_accuracy_chart'] = test_accuracy
    st.session_state.visualizations['metrics_df_chart'] = metrics_df
    st.session_state.visualizations['overall_metrics_df_chart'] = overall_metrics_df
    st.session_state.visualizations['assembled_df_labels_chart'] = assembled_df_labels
    st.session_state.visualizations['metrics_df_label_chart'] = metrics_df_label

if 'val_accuracy_chart' in st.session_state.visualizations:

    st.markdown("Distribuzione etichette")
    fig = px.bar(
        st.session_state.visualizations['assembled_df_labels_chart'],
        x="AIDRLabel_index",
        y="count",
        color="AIDRLabel_index",
    )
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.markdown("Risultati Classificazione")

    st.markdown(f"_Validation Accuracy_: :blue[{st.session_state.visualizations['val_accuracy_chart']:.4f}]")
    st.divider()
    st.markdown(f"_Test Accuracy_: :blue[{st.session_state.visualizations['test_accuracy_chart']:.4f}]")

    st.divider()

    # Assuming you have defined st.session_state.visualizations['metrics_df_chart'] appropriately

    st.markdown("Metriche")

    metriche = ['Precision', 'Recall', 'F1-Score']
    fig = go.Figure()

    for label in st.session_state.visualizations['metrics_df_chart']['Label']:
        data_frame = st.session_state.visualizations['metrics_df_chart'][metriche].loc[
            st.session_state.visualizations['metrics_df_chart']['Label'] == label]
        text = np.trunc(data_frame.values * 1000) / 1000
        fig.add_trace(go.Bar(x=metriche, y=data_frame.values[0], text=text, name=label))

    fig.update_layout(barmode='group', title="Metriche per ciascuna etichetta", xaxis_title="Metrica",
                      yaxis_title="Valore", width=1500)
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    metriche = ['Precision', 'Recall', 'F1-Score']

    fig = go.Figure()

    for label in st.session_state.visualizations['metrics_df_label_chart']['Label']:
        data_frame_label = st.session_state.visualizations['metrics_df_label_chart'][metriche].loc[
            st.session_state.visualizations['metrics_df_label_chart']['Label'] == label]
        text = np.trunc(data_frame_label.values * 1000) / 1000
        fig.add_trace(go.Bar(x=metriche, y=data_frame_label.values[0], text=text, name=label))

    fig.update_layout(barmode='group', title="Metriche per ciascuna etichetta", xaxis_title="Metrica",
                      yaxis_title="Valore", width=1500)
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    fig_overall = px.bar(st.session_state.visualizations['overall_metrics_df_chart'], x="Metrica", y="Valore",
                         title="Metriche Generali",
                         labels={"Metrica": "Metrica", "Valore": "Valore"},
                         height=400,                                                  
                         )

    left, middle, right = st.columns((1, 5, 1))
    with middle:
        st.plotly_chart(fig_overall)
        st.divider()

pre_button = False


