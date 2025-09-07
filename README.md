# MuJoCo CPU Simulation (src layout)

This repository is organized with a src/ layout and split into logical packages.

## Structure

- `src/`
  - `main.py`: Entry point for running simulations.
  - `core/`: Core simulation logic
    - `model_loader.py`: Load MuJoCo models and resolve XML paths.
    - `utils.py`: Utilities (GL setup, model analysis, XML helpers).
    - `geometry.py`: Geometry, bounds, packing and placement algorithms.
    - `initialization.py`: State initialization for scenes.
    - `simulation.py`: Single/multiprocess simulation and pose export.
    - `visualization.py`: Visualization and rendering helpers.
  - `config/`
    - `config_manager.py`: CLI + YAML configuration loader.
  - `tools/`
    - `xml_processor.py`: Process part/body/scene XMLs for the project.

Top-level files:

- `config.yaml`: Default configuration values.
- `.gitignore`: Standard Python/VS Code ignores + project artifacts.
- `README.md`: This file.

## Usage

Do NOT run Python on the host if it doesn't have an environment. When running elsewhere:

- Ensure MuJoCo, numpy, matplotlib, mediapy, tqdm, and PyYAML are installed.
- From the repo root, run the entrypoint:

  python -m src.main

- Override config via CLI flags or edit `config.yaml`.

## Config

`config.yaml` drives paths and parameters. `src/config/config_manager.py` merges CLI args with YAML.

Key sections include:

- model.path, model.part_name, model.replicate_count
- simulation.batch, simulation.steps, simulation.seed
- multiprocessing.workers
- initialization.xy_extent, z_low, min_clearance, layer_gap, yaw_only, allow_3d_rot, jitter_frac, fit_mode
- rendering.viewer, rendering.save_images
- paths.npy_out_dir, paths.viz_out_dir
- xml_processing.models_dir, scenes_dir, mesh_scale, texture_width, texture_height, ground_size

## Notes

- Entry scripts have been moved under `src/`. You should import from `core` and `config` packages.
- If you previously used modules at repo root, update imports to `from core...` or `from config...`.
- The old root modules are kept temporarily; migrate your usage to the new paths then delete them.
