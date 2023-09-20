from selenium import webdriver
import pyautogui
import time
from bs4 import BeautifulSoup
import re

# Initialize the web driver (e.g., Chrome)
driver = webdriver.Chrome()

# Open a website
driver.get("https://garden.org/plants/search/text/?q=Abelia+chinensis")

time.sleep(5)

pyautogui.click(86, 474)

time.sleep(5)

page_content = driver.page_source
soup = BeautifulSoup(page_content, "html.parser")
print(page_content)
res = soup.find_all("a", href=re.compile(r"^/plants/view/.*\/$"))
results = []
for r in res:
    if r.text:
        results.append(r.text)
