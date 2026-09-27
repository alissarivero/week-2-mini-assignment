.PHONY: install test run clean format lint

install:
	python -m pip install -r requirements.txt

test:
	python -m pytest -q

run:
	python question1.py

format:
	python -m black analysis.py visuals.py question1.py tests

lint:
	python -m black --check analysis.py visuals.py question1.py tests
	python -m flake8 analysis.py visuals.py question1.py tests

clean:
	rm -rf __pycache__/ .pytest_cache/ tests/__pycache__/
