#!/usr/bin/env python
import os
import datetime
from auth_app import create_app

app = create_app()
app.jinja_env.globals['now'] = datetime.datetime.utcnow()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8001, debug=True) 