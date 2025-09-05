PY=python

.PHONY: data train evaluate predict clean

data:
	$(PY) -m src.data

train: data
	$(PY) -m src.train

evaluate:
	$(PY) -m src.evaluate

predict:
	$(PY) -m src.predict --text "Oil prices soar to all-time record, posing menace to US economy"

clean:
	rm -rf artifacts
