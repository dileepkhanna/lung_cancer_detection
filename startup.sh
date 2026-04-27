#!/bin/bash
cd web_app
gunicorn --config ../gunicorn.conf.py app:app
