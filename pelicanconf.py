#!/usr/bin/env python
# -*- coding: utf-8 -*- #

import datetime
from pelican.plugins import render_math, simple_footnotes

DEFAULT_PAGINATION = 10

AUTHOR = "Ilia Ilmer"
SITEURL = "http://localhost:8000"
SITENAME = "Ilia Ilmer"
SITETITLE = "Ilia Ilmer"
SITESUBTITLE = "Algorithms and Coffee"

THEME = "./pelican-themes/Flex"
# THEME_COLOR = "dark"
THEME_COLOR_AUTO_DETECT_BROWSER_PREFERENCE = True
THEME_COLOR_ENABLE_USER_OVERRIDE = True
PYGMENTS_STYLE = "monokai"

ARTICLE_PATHS = ["Posts"]
PATH = "content"
ARTICLE_SAVE_AS = "{date:%Y}/{date:%m}/{slug}.html"
ARTICLE_URL = "{date:%Y}/{date:%m}/{slug}.html"

SITEDESCRIPTION = ""
SITELOGO = SITEURL + "/images/compressed.jpeg"
FAVICON = SITEURL + "/images/favicon.ico"

BROWSER_COLOR = "#333"
ROBOTS = "index, follow"

CC_LICENSE = {
    "name": "Creative Commons Attribution-ShareAlike",
    "version": "4.0",
    "slug": "by-sa",
}

COPYRIGHT_YEAR = datetime.date.today().year

EXTRA_PATH_METADATA = {
    "extra/custom.css": {"path": "static/custom.css"},
}

# Enable i18n plugin.
PLUGIN_PATHS = ["./pelican-themes"]
PLUGINS = [render_math, simple_footnotes]
# MARKDOWN = {"extensions": ["toc", "fenced_code", "codehilite"]}

DEFAULT_METADATA = {"author": "Ilia Ilmer"}
# Default theme language.
I18N_TEMPLATES_LANG = "en"

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
SOCIAL = (
    ("github", "https://github.com/iliailmer"),
    ("gitlab", "https://gitlab.com/iliailmer"),
    ("linkedin", "https://linkedin.com/in/iilmer"),
)
MENUITEMS = (
    # ("About", "/pages/about.html"),
    # ("Talks", "/pages/talks.html"),
    # ("CV", "/files/resume.pdf"),
)
