<p align="center">
    <br>
    <h3 align="center">Quantum NeRF</h3>
    <p align="center">
        Master's thesis on quantum implicit networks for novel view synthesis
        <br>
        <a href="https://github.com/yeray142/QDraw/issues/new?template=bug.md">Report bug</a>
        ·
        <a href="https://github.com/yeray142/QDraw/issues/new?template=feature.md&labels=feature">Request feature</a>
    </p>
</p>

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#file-structure">File Structure</a></li>
    <li><a href="#registering-with-nerfstudio">Registering with Nerfstudio</a></li>
    <li><a href="#running-q-nerf">Running Q-NeRF</a></li>
    <li><a href="#license">License</a></li>
  </ol>
</details>

## File Structure
The project's file structure is organized as follows:

```
├── qnerf/
│   ├── modules/
│   │   ├── __init__.py
│   │   ├── modules.py   
│   ├── __init__.py
│   ├── qnerf.py
│   ├── qnerf_config.py
│   ├── qnerf_field.py
│   ├── ...
├── pyproject.toml
```

## Registering with Nerfstudio
Ensure that `nerfstudio` has been installed according to the [official installation instructions](https://docs.nerf.studio/en/latest/quickstart/installation.html). Clone or fork this repository and run the commands:

```
conda activate nerfstudio
cd quantum-nerf/
pip install -e .
ns-install-cli
```

## Running Q-NeRF
This repository creates a new Nerfstudio method named `qnerf`. To train with it, run the command:
```
ns-train qnerf --data [PATH]
```

## License

Distributed under the Apache License. See `LICENSE` for more information.
