#!/usr/bin/make -f

venv:
	virtualenv venv

requirements:
	pip install -r requirements.txt

run_jupyter:
	jupyter notebook
