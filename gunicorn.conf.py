import multiprocessing
import os

bind = '127.0.0.1:5000'
workers = int(os.environ.get('GUNICORN_WORKERS', '2'))
worker_class = 'sync'
timeout = 120
graceful_timeout = 30
keepalive = 5
limit_request_line = 4000
limit_request_fields = 100
limit_request_body = 1048576
accesslog = '-'
errorlog = '-'
loglevel = 'info'
proc_name = 'lsp-contrib'