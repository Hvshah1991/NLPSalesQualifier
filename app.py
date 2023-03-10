#Core Packages
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import river
matplotlib.use('Agg')
import seaborn as sns
import altair as alt
import ast
from datetime import datetime
from utils.get_text import get_text
from PIL import Image

#Welcome Banner
display = image = Image.open("img/valutico_teal.png")
display = np.array(display)
st.image(display, width=250)
st.title(":teal[NLP Sales Qualifier...]")

#Create Footer
st.caption('''App created by Raj Shah''')

#Online ML Pkgs
from river.naive_bayes import MultinomialNB
from river.feature_extraction import BagOfWords,TFIDF
from river.compose import Pipeline

#Training Data

data = [("value of their company","business valuation"),("business valuations are frequently requested not only by business owners","business valuation"),("A business valuation conducted by a certified appraiser","business valuation"),("the real worth of a business","business valuation"),("an accurate assessment of the value of any business requires in-depth, specialized knowledge","business valuation"),("an accredited business valuation professional","business valuation"),("refinancing or gaining capital loans and, like any investment, it's important to know whether your assets are growing or declining in value","business valuation"),("real estate asset class","real estate valuation"),("future real estate values and risk","real estate valuation"),("commercial real estate appraisal, evaluation, and feasibility study reports","real estate valuation"),("business valuation","business valuation"),("real estate valuation","real estate valuation"),("property valuation","real estate valuation"),("hotel valuation experts","real estate valuation"),("Commercial Appraisal","real estate valuation"),("business appraisal","business valuation"),("company valuation","business valuation")]

#Model Building
model = Pipeline(('vectorizer',BagOfWords(lowercase=True)),('nv',MultinomialNB()))
for x,y in data:
    model = model.learn_one(x,y)
    
#Storage in a database
import sqlite3
conn = sqlite3.connect('data.db')
c = conn.cursor()

#Create Fxn from SQL
def create_table():
    c.execute('CREATE TABLE IF NOT EXISTS predictionTable(message TEXT,prediction TEXT,probability NUMBER,businessvaluation_proba NUMBER, realestatevaluation_proba NUMBER,postdate DATE)')
    
def add_data(message,prediction,probability,businessvaluation_proba,realestatevaluation_proba,postdate):
    c.execute('INSERT INTO predictionTable(message,prediction,probability,businessvaluation_proba,realestatevaluation_proba,postdate) VALUES (?,?,?,?,?,?)',(message,prediction,probability,businessvaluation_proba,realestatevaluation_proba,postdate))
    conn.commit()
    
def view_all_data():
    c.execute("SELECT * FROM predictionTable")
    data = c.fetchall()
    return data


def main():
    menu = ["Home","Manage","Web Scraper","About"]
    create_table()
    
    choice = st.sidebar.selectbox("Menu",menu)
    if choice == "Home":
        st.subheader("Home")
        with st.form(key='mlform'):
            col1,col2 = st.columns([2,1])
            with col1:
                message = st.text_area("Message")
                submit_message = st.form_submit_button(label='Predict')
                
            with col2:
                st.write("Online Machine Learning Qualifier")
                st.write("Predict Text as Business Valuation or Real Estate Valuation")
                
                
        if submit_message:
            prediction = model.predict_one(message)
            prediction_proba = model.predict_proba_one(message)
            probability = max(prediction_proba.values())
            postdate = datetime.now()
            #Add data to database
            add_data(message,prediction,probability,prediction_proba['business valuation'],prediction_proba['real estate valuation'],postdate)
            st.success("Data Submitted")
            
            res_col1, res_col2 = st.columns(2)
            with res_col1:
                st.info("Original Text")
                st.write(message)
                
                st.success("Prediction")
                st.write(prediction)
                
            with res_col2:
                st.info("Probability")
                st.write(prediction_proba)
                
                #Plot of Probability
                df_proba = pd.DataFrame({'label':prediction_proba.keys(),'probability':prediction_proba.values()})
                st.dataframe(df_proba)
                #visualization
                fig = alt.Chart(df_proba).mark_bar().encode(x='label',y='probability')
                st.altair_chart(fig,use_container_width=True)
    
    elif choice == "Manage":
        st.subheader("Manage & Monitor Results")
        stored_data = view_all_data()
        new_df = pd.DataFrame(stored_data,columns=['message','prediction','probability','businessvaluation_proba','realestatevaluation_proba','postdate'])
        st.dataframe(new_df)
        new_df['postdate'] = pd.to_datetime(new_df['postdate'])
        
        #c = alt.Chart(new_df).mark_line().encode(x='minutes(postdate)',y='probability') #For Minutes
        c = alt.Chart(new_df).mark_line().encode(x='postdate',y='probability')
        st.altair_chart(c)
        
        c_businessvaluation_proba = alt.Chart(new_df['businessvaluation_proba'].reset_index()).mark_line().encode(x='businessvaluation_proba',y='index')
        c_realestatevaluation_proba = alt.Chart(new_df['realestatevaluation_proba'].reset_index()).mark_line().encode(x='realestatevaluation_proba',y='index')
        
        
        c1,c2 = st.columns(2)
        with c1:
            with st.expander("Business Valuation Probability"):
                st.altair_chart(c_businessvaluation_proba,use_container_width=True)
                
        with c2:
            with st.expander("Real Estate Valuation Probability"):
                st.altair_chart(c_realestatevaluation_proba,use_container_width=True)
                
        with st.expander("Prediction Distribution"):
            fig2 = plt.figure()
            sns.countplot(x='probability',data=new_df)
            st.pyplot(fig2)
    #Web Scraper
    elif choice == "Web Scraper":
        st.subheader("Web Scraper")
        st.markdown('##### This option helps you to scrape, and extract the text and show only first 10 lines')
        URL = st.text_input("Enter the URL of the webpage you want to scrape")
        if URL is not None:
            if st.button("Scrape"):
                text = get_text(URL)
                df = pd.DataFrame(text.splitlines(),columns=["Webpage_text"],index=None)
                st.markdown('## Showing the first ten lines of the text')
                st.dataframe(df.head(10))
                st.info('''Download the text as a csv file if you like.''')
                st.download_button(label="Download the text as a csv file", data=df.to_csv(index=False, encoding='utf-8'),file_name='webpage_text.csv',mime='text/csv')
        else:
            st.warning("Please enter a valid URL")
        
    else:
        st.subheader("About")
        st.caption("NLP Sales Qualifier can be used for Web Scraping for Lead Sourcing. When you have identified potential leads, you can access their URL and web scrape their website using this program. The program instantly downloads a csv file, which you can copy-paste into the message section in Home screen - so it can start an analysis if this client Qualifies for a sles outreach or sequencing. This program uses Naive Bayes to predict and classify the lead. This program also stores the data which was produced in the form of output and utilizes it as training data for new queries. For further information on this program, contact: r.shah@valutico.com")
    
if __name__ == '__main__':
    main()
    
