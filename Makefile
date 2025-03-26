.PHONY: all make_dataset run_main

all: make_dataset run_main

make_dataset:
	@echo "Creating dataset..."
	python preprocess/process_csv.py
	python preprocess/make_dataset.py

run_main:
	for loss in PNL PUL NUL; do \
		echo "Running experiment with loss_type=$$loss"; \
		python src/main.py train.loss.type=$${loss}; \
	done
