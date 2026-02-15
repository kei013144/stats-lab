.RECIPEPREFIX := >

PYTHON ?= python
OUT_DIR ?= data/synthetic

.PHONY: help data data-full clean-data viz-smoke clean-viz

help:
> @echo "Available targets:"
> @echo "  make data       Generate lightweight sample synthetic datasets"
> @echo "  make data-full  Generate larger synthetic datasets"
> @echo "  make clean-data Remove generated CSV files under $(OUT_DIR)"
> @echo "  make viz-smoke  Run matplotlib smoke test and save PNGs"
> @echo "  make clean-viz  Remove smoke-test PNG files under reports dirs"

data:
> $(PYTHON) scripts/make_people_stats.py --n 800 --seed 42 --out_dir $(OUT_DIR)
> $(PYTHON) scripts/make_retail_synth.py --seed 42 --out_dir $(OUT_DIR) --n_customers 3000 --n_products 800 --n_orders 30000

data-full:
> $(PYTHON) scripts/make_people_stats.py --n 50000 --seed 42 --out_dir $(OUT_DIR)
> $(PYTHON) scripts/make_retail_synth.py --seed 42 --out_dir $(OUT_DIR) --n_customers 30000 --n_products 5000 --n_orders 250000

clean-data:
> $(PYTHON) -c "from pathlib import Path; d=Path('$(OUT_DIR)'); d.mkdir(parents=True, exist_ok=True); [p.unlink() for p in d.glob('*.csv')]; print('Removed CSV files from', d)"

viz-smoke:
> $(PYTHON) scripts/viz_smoke_test.py

clean-viz:
> $(PYTHON) -c "from pathlib import Path; targets=[Path('04_reports/smoke_test'), Path('reports/smoke_test')]; files=[p for t in targets if t.exists() for p in t.glob('*.png')]; [p.unlink() for p in files]; print('Removed PNG files:', len(files))"
