.PHONY: install test train docker

install:
	pip install -r requirements.txt

test:
	pytest -q

train:
	python -m app.train --model both --epochs 2

docker:
	docker compose up --build
