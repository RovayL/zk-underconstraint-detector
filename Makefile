.PHONY: quickstart
quickstart:
	python -m zkuc.cli features --r1cs src/circuits/examples/tiny_add.r1cs.json --out data/features.jsonl


.PHONY: seedbank
seedbank:
    ./scripts/build_seedbank.sh


.PHONY: seedbank-dataset
seedbank-dataset: seedbank
    python -m zkuc.cli seed-dataset \
      --src-glob 'src/circuits/seedbank/build/*.r1cs.json' \
      --out-dir data/seeds \
      --per-src 8 \
      --trials 10 --subset 32 \
      --freeze-const --freeze-pub-inputs \
      --jsonl data/dataset.jsonl
