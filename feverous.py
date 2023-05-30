from feverous.database.feverous_db import FeverousDB
from feverous.utils.wiki_page import WikiPage

db =  FeverousDB("path_to_the_wiki")

page_json = db.get_doc_json("Anarchism")
wiki_page = WikiPage("Anarchism", page_json)

context_sentence_14 = wiki_page.get_context('sentence_14') # Returns list of context Wiki elements

prev_elements = wiki_page.get_previous_k_elements('sentence_5', k=4) # Gets Wiki element before sentence_5
next_elements = wiki_page.get_next_k_elements('sentence_5', k=4)