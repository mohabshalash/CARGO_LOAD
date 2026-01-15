# Load Optimizer

A Python package for optimizing load distribution and packing using various heuristic algorithms.

## Features

- 3D box packing algorithms (First-fit, Best-fit)
- Constraint checking (volume, weight, dimensions)
- Pydantic-based data validation
- Extensible architecture for custom heuristics

## Installation

```bash
pip install -e .
```

For development dependencies:

```bash
pip install -e ".[dev]"
```

## Usage

```python
from load_optimizer.geometry import Box, Item
from load_optimizer.packing.heuristics import first_fit

# Create a container
container = Box(width=10, height=10, depth=10)

# Create items to pack
items = [
    Item(width=3, height=3, depth=3, weight=1.0),
    Item(width=2, height=2, depth=2, weight=0.5),
    Item(width=4, height=4, depth=4, weight=2.0),
]

# Pack items using first-fit heuristic
bins = first_fit(items, container)
print(f"Number of bins used: {len(bins)}")
```

## Project Structure

```
load-optimizer/
├── src/
│   └── load_optimizer/
│       ├── __init__.py
│       ├── main.py
│       ├── geometry.py
│       ├── packing/
│       │   ├── __init__.py
│       │   ├── heuristics.py
│       │   └── constraints.py
│       └── io/
│           ├── __init__.py
│           └── schemas.py
├── tests/
├── data/
├── pyproject.toml
└── README.md
```

## Development

Run tests:

```bash
pytest
```

Format code:

```bash
black src/ tests/
```

Lint code:

```bash
ruff check src/ tests/
```

## License

MIT







