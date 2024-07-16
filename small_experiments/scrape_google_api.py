from scholarly import scholarly

query = "First in-vivo human imaging at 10.5T: Imaging the body at 447 MHz"
search_query = scholarly.search_pubs(query)
search_query.bibtex
list(search_query)
pub = next(search_query)