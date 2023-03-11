# import libraries
from selenium import webdriver
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.common.by import By
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from webdriver_manager.firefox import GeckoDriverManager
import streamlit as st


# get text from the webpage and dump it into a json variable
@st.cache_data
def get_text(URL):
    firefoxOptions = Options()
    firefoxOptions.add_argument("--headless")
    service = Service(GeckoDriverManager().install())
    driver = webdriver.Firefox(options=firefoxOptions,service=service,)
    driver.get(URL)
    elem = driver.find_element(By.TAG_NAME, "body")
    text = elem.text
    driver.quit()
    return text
 
