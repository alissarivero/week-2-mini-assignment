.PHONY: install test run clean

install:
	python -m pip install -r requirements.txt

test:
	python -m pytest -q

run:
	python question1.py

clean:
	rm -rf __pycache__/ .pytest_cache/ tests/__pycache__/
