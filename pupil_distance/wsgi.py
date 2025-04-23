"""
WSGI config for pupil_distance project.

It exposes the WSGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/5.0/howto/deployment/wsgi/
"""

import os

from django.core.wsgi import get_wsgi_application

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'pupil_distance.settings')

application = get_wsgi_application()

# Força o uso da porta 80
if os.environ.get('RUN_MAIN') or os.environ.get('WERKZEUG_RUN_MAIN'):
    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(('0.0.0.0', 80))
    sock.close()
