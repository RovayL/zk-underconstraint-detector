.PHONY: quickstart


quickstart:
	python -m zkuc.cli features --r1cs src/circuits/examples/tiny_add.r1cs.json --out data/features.jsonl
