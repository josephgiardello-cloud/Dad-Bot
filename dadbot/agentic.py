

from __future__ import annotations

import email.utils
import json
import re
import time
import uuid
from datetime import datetime
from email.message import EmailMessage
from pathlib import Path
from urllib.parse import urlencode, urlparse
from urllib.request import Request, urlopen

from dateutil import parser as dateutil_parser
from dateutil.relativedelta import relativedelta


