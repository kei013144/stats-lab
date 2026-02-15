.RECIPEPREFIX := >

PYTHON ?= python
OUT_DIR ?= data/synthetic

.PHONY: help data data-full clean-data

help:
> @echo "Available targets:"
> @echo "  make data       Generate lightweight sample synthetic datasets"
> @echo "  make data-full  Generate larger synthetic datasets"
> @echo "  make clean-data Remove generated CSV files under $(OUT_DIR)"

data:
> $(PYTHON) scripts/make_people_stats.py --n 800 --seed 42 --out_dir $(OUT_DIR)
> $(PYTHON) scripts/make_retail_synth.py --seed 42 --out_dir $(OUT_DIR) --n_customers 3000 --n_products 800 --n_orders 30000

data-full:
> $(PYTHON) scripts/make_people_stats.py --n 50000 --seed 42 --out_dir $(OUT_DIR)
> $(PYTHON) scripts/make_retail_synth.py --seed 42 --out_dir $(OUT_DIR) --n_customers 30000 --n_products 5000 --n_orders 250000

clean-data:
> $(PYTHON) -c "from pathlib import Path; d=Path('$(OUT_DIR)'); d.mkdir(parents=True, exist_ok=True); [p.unlink() for p in d.glob('*.csv')]; print('Removed CSV files from', d)"
