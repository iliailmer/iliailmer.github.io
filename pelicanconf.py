#!/usr/bin/env python
# -*- coding: utf-8 -*- #
# TODO build theme
AUTHOR = "Ilia Ilmer"
SITENAME = "Ilia Ilmer"
SITEURL = ""

PATH = "content"

TIMEZONE = "America/New_York"

DEFAULT_LANG = "en"

# Feed generation is usually not desired when developing
FEED_ALL_ATOM = None
CATEGORY_FEED_ATOM = None
TRANSLATION_FEED_ATOM = None
AUTHOR_FEED_ATOM = None
AUTHOR_FEED_RSS = None

# Blogroll
# LINKS = (
#     ("Pelican", "https://getpelican.com/"),
#     ("Python.org", "https://www.python.org/"),
#     ("Jinja2", "https://palletsprojects.com/p/jinja/"),
#     ("You can modify those links in your config file", "#"),
# )

# Social widget
# SOCIAL = (
#     ("You can add links in your config file", "#"),
#     ("Another social link", "#"),
# )

DEFAULT_PAGINATION = 10
PLUGIN_PATHS = ["./pelican-plugins", "./pelican-themes"]
PLUGINS = ["render_math"]


THEME = "Flex"  # bootstrap2-dark
ARTICLE_PATHS = ["Posts"]
ARTICLE_SAVE_AS = "{date:%Y}/{date:%m}/{slug}.html"
ARTICLE_URL = "{date:%Y}/{date:%m}/{slug}.html"


# Uncomment following line if you want document-relative URLs when developing
# RELATIVE_URLS = True
