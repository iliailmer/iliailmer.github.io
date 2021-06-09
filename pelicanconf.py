#!/usr/bin/env python
# -*- coding: utf-8 -*- #


DEFAULT_PAGINATION = 10

THEME = "./pelican-themes/Flex"

ARTICLE_PATHS = ["Posts"]
ARTICLE_SAVE_AS = "{date:%Y}/{date:%m}/{slug}.html"
ARTICLE_URL = "{date:%Y}/{date:%m}/{slug}.html"
AUTHOR = "Ilia Ilmer"
SITEURL = "https://iliailmer.github.io"
SITENAME = "Ilia's Blog"
SITETITLE = "Ilia Ilmer"
SITESUBTITLE = "Algorithms and Coffee"

# SITEDESCRIPTION = "Foo Bar's Thoughts and Writings"
SITELOGO = SITEURL + "/images/profile.png"
# FAVICON = SITEURL + "/images/favicon.ico"

BROWSER_COLOR = "#333"
ROBOTS = "index, follow"

CC_LICENSE = {
    "name": "Creative Commons Attribution-ShareAlike",
    "version": "4.0",
    "slug": "by-sa",
}

COPYRIGHT_YEAR = 2021

EXTRA_PATH_METADATA = {
    "extra/custom.css": {"path": "static/custom.css"},
}

CUSTOM_CSS = "static/custom.css"

# ADD_THIS_ID = "ra-77hh6723hhjd"

# Enable i18n plugin.
PLUGIN_PATHS = ["./pelican-plugins", "./pelican-themes"]
PLUGINS = ["render_math", "i18n_subsites"]
JINJA_EXTENSIONS = ['jinja2.ext.i18n']
# Default theme language.
I18N_TEMPLATES_LANG = "en"

PATH = "content"

TIMEZONE = "America/New_York"

DEFAULT_LANG = "en"

# Feed generation is usually not desired when developing
FEED_ALL_ATOM = None
CATEGORY_FEED_ATOM = None
TRANSLATION_FEED_ATOM = None
AUTHOR_FEED_ATOM = None
AUTHOR_FEED_RSS = None

NEWEST_FIRST_ARCHIVES = True
MAIN_MENU = True
MENUITEMS = (
    # ("Home", "/"),
    ("About", "/pages/about.html"),
)
