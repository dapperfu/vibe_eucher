# Installation Guide

This guide will help you install and set up the Euchre CLI Game on your system.

## Prerequisites

- **Python**: Version 3.8 or higher
- **pip**: Python package installer (usually comes with Python)
- **Git**: For cloning the repository
- **Make**: For using the provided Makefile (optional but recommended)

## Installation Methods

### Method 1: Traditional Installation (Recommended for Development)

This method is best for developers who want to modify the code or contribute to the project.

#### Step 1: Clone the Repository
```bash
git clone https://github.com/dapperfu/vibe_eucher.git
cd vibe_eucher
```

#### Step 2: Create Virtual Environment
```bash
python3 -m venv venv
```

#### Step 3: Activate Virtual Environment

**On Linux/macOS:**
```bash
source venv/bin/activate
```

**On Windows:**
```bash
venv\Scripts\activate
```

#### Step 4: Install Dependencies
```bash
pip install -r requirements.txt
```

#### Step 5: Verify Installation
```bash
python -m euchre.cli_main --help
```

### Method 2: Using Makefile (Recommended for Development)

The project includes a Makefile that automates the installation process:

```bash
# Clone and setup everything
git clone https://github.com/dapperfu/vibe_eucher.git
cd vibe_eucher
make install
```

### Method 3: Pip Installation (Recommended for End Users)

This method installs the game as a system command, making it available from anywhere.

#### Step 1: Clone and Setup
```bash
git clone https://github.com/dapperfu/vibe_eucher.git
cd vibe_eucher
```

#### Step 2: Install in Development Mode
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -e .
```

#### Step 3: Verify Installation
```bash
euchre --help
```

## Quick Installation Commands

### One-liner for Linux/macOS:
```bash
git clone https://github.com/dapperfu/vibe_eucher.git && cd vibe_eucher && python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt
```

### One-liner with Make:
```bash
git clone https://github.com/dapperfu/vibe_eucher.git && cd vibe_eucher && make install
```

## Running the Game

### After Traditional Installation:
```bash
# Activate virtual environment (if not already active)
source venv/bin/activate

# Run the game
python -m euchre.cli_main play
```

### After Pip Installation:
```bash
# Run directly (no virtual environment activation needed)
euchre play
```

### Using Makefile:
```bash
make run
```

## Troubleshooting

### Common Issues

#### 1. "Python command not found"
**Solution**: Install Python 3.8+ from [python.org](https://python.org)

#### 2. "pip command not found"
**Solution**: Python 3.4+ includes pip by default. If missing:
```bash
python3 -m ensurepip --upgrade
```

#### 3. "Permission denied" errors
**Solution**: Ensure you have write permissions in the project directory:
```bash
sudo chown -R $USER:$USER /path/to/vibe_eucher
```

#### 4. "Euchre requires exactly 4 players"
**Solution**: This is expected behavior - euchre is a 4-player game

#### 5. Import errors
**Solution**: Ensure you're using the virtual environment:
```bash
source venv/bin/activate
```

#### 6. "euchre command not found" (after pip installation)
**Solution**: 
- Verify the virtual environment is activated
- Check installation location: `which euchre`
- Reinstall: `pip uninstall euchre && pip install -e .`

### Platform-Specific Issues

#### Windows
- Use `venv\Scripts\activate` instead of `source venv/bin/activate`
- Ensure Python is added to PATH during installation
- Use `python` instead of `python3` if that's your system's Python command

#### macOS
- If you get SSL errors, you may need to install certificates:
  ```bash
  /Applications/Python\ 3.x/Install\ Certificates.command
  ```

#### Linux
- Install system dependencies if needed:
  ```bash
  sudo apt-get install python3-venv python3-pip  # Ubuntu/Debian
  sudo yum install python3-venv python3-pip      # CentOS/RHEL
  ```

## Verification

After installation, verify everything works:

```bash
# Test basic functionality
python -m euchre.cli_main --help

# Test game startup
python -m euchre.cli_main play --help

# Run a quick AI vs AI game
python -m euchre.cli_main ai-vs-ai
```

## Next Steps

Once installation is complete:

1. **Read the [Quick Start Guide](quick-start.md)** to learn how to play
2. **Check [Game Rules](game-rules.md)** to understand euchre
3. **Try [Human vs AI](human-vs-ai.md)** to play against the computer
4. **Explore [CLI Commands](cli-commands.md)** for advanced usage

## Getting Help

If you encounter issues:

1. Check this troubleshooting section
2. Review the [GitHub Issues](https://github.com/dapperfu/vibe_eucher/issues) page
3. Check the [Development Setup](development/setup.md) for advanced configuration
4. Ensure you're using the latest version of the code

---

*For development setup, see [Development Setup](development/setup.md)* 