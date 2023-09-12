# imports
import pandas as pd
import streamlit as st
import plotly
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import math
import requests

# import data
# use dataframe without outliers
outliers_path = "Data/df_wo_outliers.xlsx"
df = pd.read_excel(outliers_path)
pricing_path = "Data/get_around_pricing_project.csv"
pricing = pd.read_csv(pricing_path)

# set page configuration
st.set_page_config(
    page_title="Getaround Dashboard",
    # page_icon="getaround_favicon.png",
    layout="wide"
)

tab1, tab2 = st.tabs(["Dashboard", "Price Optimizer"])

with tab1:
    # quote
    st.header('"Informed decision-making comes from a long tradition of guessing and blaming others for inadequate results" - Scott Adams')
    # image
    st.image("https://download.logo.wine/logo/Getaround/Getaround-Logo.wine.png")
    st.divider()

    # objectives
    st.subheader("Objectives")
    st.write("- Create a web dashboard to support decisionmakers analyze data.")
    st.write("- Through data analysis help product manager choose a time threshold between rentals.")
    st.write("- Create an API that serves a machine learning algorithm to suggest optimal rental prices to car owners.")
    st.write("- Host the API and dashboard on an online server.")
    st.divider()

    # questions to be answered
    st.header("Data Analysis")
    st.divider()

    # QUESTION 1
    st.subheader("How often are drivers late for the next check-in? How does it impact the next driver?")

    # metric
    perc_delays = round((len(df[df["delay_minutes"] > 0]) / len(df)) * 100 ,2)
    st.metric("% of Delays", perc_delays)

    # graph
    labels = {"late":""}
    fig1 = px.histogram(df, "late",
                        labels=labels)
    st.plotly_chart(fig1, use_container_width=True)

    st.divider()

    # QUESTION 2
    st.subheader("When there is a delay, how many minutes do late drivers take after programmed end of rental time?")

    # metrics
    col1, col2 = st.columns(2)
    with col1:
        median_delay = df[df["delay_minutes"]>0]["delay_minutes"].median()
        st.metric("Median Delay", median_delay)

    with col2:
        median_diff_rentals = df["difference_rentals_minutes"].median()
        st.metric("Median Minutes between Rentals", median_diff_rentals)
    # graph
    labels = {"delay_minutes":"Delay in Minutes"}
    fig2 = px.histogram(df, x="delay_minutes",
                        color="late",
                        labels=labels)
    st.plotly_chart(fig2, use_container_width=True)

    st.divider()

    # QUESTIONS 3 and 4
    st.subheader("How many rentals would be affected by the feature depending on the threshold and scope we choose?")
    st.subheader("How many problematic cases will it solve depending on the chosen threshold and scope?")

    # data display

    # use percentiles as thresholds, from 10 to 100
    # values will be % of EARLY check ins that would be early, for each threshold
    percentiles_delay = {}
    for percentile in range(10,110,10):
        percentiles_delay[percentile] = \
            math.trunc(np.percentile(df[df["delay_minutes"]>0]["delay_minutes"], percentile))
        
    # convert to dataframe    
    thresholds = pd.DataFrame.from_dict(percentiles_delay, orient="index").reset_index()
    thresholds.rename(columns={"index": "Percentage of Early Check Ins",
                            0: "Minutes Threshold"},
                            inplace=True)
    thresholds = thresholds
    # show table
    st.table(thresholds.T)

    # chart
    selected_threshold = st.select_slider("Select Minutes Threshold:", thresholds["Minutes Threshold"])

    # metric and graph

    for threshold in thresholds["Minutes Threshold"]:
        if selected_threshold==threshold:
            perc_early_checkins = thresholds[thresholds["Minutes Threshold"]==selected_threshold]["Percentage of Early Check Ins"]
            st.metric("Percentage of Early Checkins", perc_early_checkins)

    # graph
    delay = df[df["delay_minutes"]>0]
    labels = {"delay_minutes":"Delay in Minutes"}
    fig3 = px.histogram(delay, x="delay_minutes",
                        labels=labels,
                        title="Threshold for Early Checkins")
    fig3.add_vline(x=selected_threshold,
                line_width=5,
                line_color="red")
    st.plotly_chart(fig3, use_container_width=True)

    st.divider()

    # chosen threshold
    st.subheader("Recommended Threshold")

    st.markdown("##### The recommended threshold is be of 120 minutes, or 2 hours, between the requested end time and the next rental planned start time.\nThese are the reasons this threshold is considered as optimal:")

    st.caption("- The current median difference of time between end of rental and start of next one is of 180 minutes. This time will be reduced by 33.33%,\
                and eliminate a full hour of dead time!")
    st.caption("- Only 20% of delays were of a greater waiting time than 98 minutes, so the expectation is that delays will be reduced between 80% and 90%, and the total delay rate from 57% to less than 10%. \
                Delays make processes inefficient and may leave users with bad experiences, avoiding them is key!")    
    st.caption("- An additional point: Users who use Connect arrive 8 minutes in advance, while users who check in with mobile arrive 9 minutes late.\
            It is important, but not an immediate priority, to start using more the Connect technology.")

# Machine Learning
with tab2:
    st.header("Get your optimal rental price with Machine Learning!")
    st.divider()

    st.markdown("#### Please fill in the following information about your car:")
    st.markdown("##### For the right columns, select 0 for NO, 1 for YES.")


    # input values
    col1, col2 = st.columns(2)
    with col1:
        model_key = st.selectbox(label="Model Key", options=["Audi", "BMW", "Citroën", "Peugeot", "Renault", "other"])
        mileage = st.number_input(label="Mileage", min_value=0)
        engine_power = st.number_input(label="Engine Power", min_value=0)
        fuel = st.selectbox(label="Fuel", options=["diesel", "other"])
        paint_color = st.selectbox(label="Paint Color", options=["black", "blue", "grey", "white", "other"])
        car_type = st.selectbox(label="Car Type", options=["estate", "hatchback", "sedan", "suv", "other"])
        st.write("")
        st.write("")
        st.markdown('##### Once you are done with your selections, click "Submit".')

    with col2:
        private_parking_available = st.number_input(label="Private Parking Available", min_value=0, max_value=1)
        has_gps = st.number_input(label="Has GPS", min_value=0, max_value=1)
        has_air_conditioning = st.number_input(label="Has Air Conditioning", min_value=0, max_value=1)
        automatic_car = st.number_input(label="Automatic Car", min_value=0, max_value=1)
        has_getaround_connect = st.number_input(label="Has Getaround Connect", min_value=0, max_value=1)
        has_speed_regulator = st.number_input(label="Has Speed Regulator", min_value=0, max_value=1)
        winter_tires = st.number_input(label="Winter Tires", min_value=0, max_value=1)


    # create dictionary with inputs
    data = {
    "model_key": model_key,
    "mileage": mileage,
    "engine_power": engine_power,
    "fuel": fuel,
    "paint_color": paint_color,
    "car_type": car_type,
    "private_parking_available": private_parking_available,
    "has_gps": has_gps,
    "has_air_conditioning": has_air_conditioning,
    "automatic_car": automatic_car,
    "has_getaround_connect": has_getaround_connect,
    "has_speed_regulator": has_speed_regulator,
    "winter_tires": winter_tires
    }

    # request prediction from API
    def predict_price(data):
        api_input = data
        r = requests.post("http://host.docker.internal:4001/predict", json=api_input)
        st.write(r.json())

    # submit button
    if st.button('Submit'):
            # Display output
            st.subheader("Optimal Price:")
            output = predict_price(data)
            st.write(output)