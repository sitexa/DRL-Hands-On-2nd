# Deep Reinforcement Learning Hands-On (Updated for Gymnasium)

This repository contains an updated version of the code from the book "Deep Reinforcement Learning Hands-On, Second Edition," originally published by Packt and available [here](https://github.com/PacktPublishing/Deep-Reinforcement-Learning-Hands-On-Second-Edition). The original code was designed to work with older versions of the OpenAI Gym API. This fork has been modified to be compatible with the new Gymnasium interface.

## Modifications

The following is a summary of the key modifications made to the original codebase:

- **Gymnasium Compatibility**: Updated all environment interaction calls to match the new Gymnasium API. For example:
  - `env.reset()` now returns a tuple `(obs, info)`.
  - `env.step()` now returns a tuple `(obs, reward, terminated, truncated, info)`.
- **Wrapper Update**: Replaced the deprecated `gym.wrappers.Monitor` with `gymnasium.wrappers.RecordVideo` for recording videos of the agent's performance.
- **PTAN Integration**: Integrated the third-party library PTAN directly into the repository's code to streamline dependencies and improve ease of use.

## Requirements

To run the updated code, you will need to install the following:

- Python 3.x
- Gymnasium (latest version)
- PyTorch (version compatible with your system)
- Other dependencies listed in `requirements.txt`

## Installation

To set up your environment to run the code, follow these steps:

1. Clone this repository:

   ```bash
   git clone https://github.com/Theigrams/DRL-Hands-On-2nd.git
   ```

2. Navigate to the cloned repository's directory:

   ```bash
   cd DRL-Hands-On-2nd
   ```

3. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

After installation, you can run the individual chapter scripts to see the algorithms in action.

Before running the scripts, you should add the parent directory of the project to your Python path to ensure modules are found correctly. You can do this dynamically by using the following command in your terminal:

```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)
```

This command temporarily sets the `PYTHONPATH` environment variable for the current terminal session to include the directory you are currently in.

To run a script from Chapter 7, for example, you would do the following:

```bash
cd DRL-Hands-On-2nd
export PYTHONPATH=$PYTHONPATH:$(pwd)
cd Chapter07
python 01_actions.py
```

This ensures that when you run `01_actions.py`, Python will be able to find any modules that are located in the parent directory.

Please note that the `$(pwd)` command will only work in Unix-like terminals, such as those found on Linux or macOS. If you are using Windows, you will need to use an equivalent command for the Command Prompt or PowerShell.

## Contributing

We welcome contributions to this project! To contribute, please follow these steps:

1. Fork the repository and create your branch from `master`.
2. Make your changes and ensure your code adheres to the project's coding standards.
3. Install `pre-commit` to your development environment. `pre-commit` is a framework for managing and maintaining multi-language pre-commit hooks. It will use the configuration set in `pyproject.toml` to format your code automatically.

    To install `pre-commit`, run:

    ```bash
    pip install pre-commit
    ```

    Once installed, set up the hooks with:

    ```bash
    pre-commit install
    ```

    Before committing your changes, `pre-commit` will run automatically and format your code according to the style defined in `pyproject.toml`. If there are any formatting issues, `pre-commit` will fix them automatically or provide feedback on what needs to be changed.

4. After ensuring your changes pass all checks and the code is properly formatted, commit your changes.
5. Push to your fork and submit a pull request to the original repository.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Original authors of "Deep Reinforcement Learning Hands-On, Second Edition"
- Contributors to the original Packt Publishing repository
- The Gymnasium team for their continuous efforts in advancing reinforcement learning environments

## Contact

If you have any questions or comments about this update, please feel free to open an issue in this repository, or contact me directly at [theigrams@buaa.edu.cn](mailto:theigrams@buaa.edu.cn).

Happy learning and coding!
