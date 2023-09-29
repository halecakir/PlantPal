import json
import os
import random
import re

import requests
from bs4 import BeautifulSoup

PLANT_DB_PATH = "../plant-database/json/"
import os

os.environ["GOOGLE_CSE_ID"] = "24d20bd66d15d467f"
os.environ["GOOGLE_API_KEY"] = "AIzaSyDy1PR0mPMNg6j3qQk5pPwqLDR9tua437M"
from langchain.tools import Tool
from langchain.utilities import GoogleSearchAPIWrapper


def _latin2common_name(latin_name):
    """Convert latin name to common name"""

    search = GoogleSearchAPIWrapper()
    tool = Tool(
        name="Google Search",
        description="Search Google for recent results.",
        func=search.run,
    )
    out = tool.run(latin_name)
    return out


def latin2common_name(latin_name):
    from selenium import webdriver
    import pyautogui
    import time
    from bs4 import BeautifulSoup
    import re

    driver = webdriver.Chrome()
    query = "+".join(latin_name.split(" "))
    url = f"https://garden.org/plants/search/text/?q={query}"
    driver.get(url)
    print("URL " + url)
    time.sleep(5)

    pyautogui.click(86, 474)

    time.sleep(5)

    page_content = driver.page_source

    driver.close()

    soup = BeautifulSoup(page_content, "html.parser")
    res = soup.find_all("a", href=re.compile(r"^/plants/view/.*\/$"))
    results = []
    for r in res:
        if r.text:
            results.append(r.text)

    return results


def extract_common_name(search_results, latin_name):
    """Extract common name from search results"""
    regex = r"Plant database entry for ([^.]*) \("
    print(regex)
    matches = re.match(regex, search_results, flags=re.IGNORECASE | re.MULTILINE)

    if matches:
        return matches.group(1)
    else:
        return "NONE"


if __name__ == "__main__":
    import time
    # exist = 0
    # not_exist = 0
    # for filename in os.listdir(PLANT_DB_PATH):
    #     try:
    #         full_path = os.path.join(PLANT_DB_PATH, filename)
    #         with open(full_path, "r") as f:
    #             plant = json.load(f)
    #             if "common_name" in plant:
    #                 exist += 1
    #             else:
    #                 not_exist += 1
    #     except Exception as e:
    #         print("Exception: ", e, filename)
    # print(exist, not_exist)
        # try:
        #     with open(full_path, "r+") as f:
        #         plant = json.load(f)
        #         latin_name = plant["display_pid"]
        #         search_res = latin2common_name(latin_name)
        #         if not search_res and "'" in latin_name:
        #             ln = latin_name.split("'")[0]
        #             search_res = latin2common_name(ln)
        #         plant["search_results"] = search_res
        #         if plant["search_results"]:
        #             common_name = plant["search_results"][0].split("(")[0].strip()
        #         else:
        #             common_name = "NONE"
        #         plant["common_name"] = common_name
        #     with open(full_path, "w") as f:
        #         json.dump(plant, f, indent=4)
        # except Exception as e:
        #     print("Exception: ", e, filename)
        # # sleep for 100 ms
        # time.sleep(0.1)
