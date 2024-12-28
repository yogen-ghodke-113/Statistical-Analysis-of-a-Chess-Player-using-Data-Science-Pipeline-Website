# Statistical Analysis of a Chess Player using Data Science Pipeline Website

## Prerequisites

Before running the program, ensure that you have the following installed:

- Python 3.x (3.12 or higher recommended)
- pip (Python package installer)
- virtualenv (optional but recommended)
- Streamlit (for running the app)

## Step-by-Step Guide

### 1. Clone the Repository

First, clone the project repository to your local machine:

```bash
git clone https://github.com/yogen-ghodke-113/Statistical-Analysis-of-a-Chess-Player-using-Data-Science-Pipeline-Website.git
```

Navigate to the project directory:

```bash
cd Statistical-Analysis-of-a-Chess-Player-using-Data-Science-Pipeline-Website
```

### 2. Install Python 3 and Required Dependencies

If you don't have Python 3 and pip installed, you can install them by following the steps below.

#### For Ubuntu/Debian:

```bash
sudo apt update
sudo apt install python3 python3-pip python3-venv
```

#### Install Virtual Environment (optional but recommended)

To create an isolated Python environment, use venv:

#### For Ubuntu/Linux/macOS:

```bash
python3 -m venv venv
```

#### For Windows:

```bash
python -m venv venv
```

### 3. Activate the Virtual Environment

Activate the virtual environment using the appropriate command for your operating system:

#### For Ubuntu/Linux/macOS:

```bash
source venv/bin/activate
```

#### For Windows:

```bash
.\venv\Scripts\activate
```

> Note: If you see an error like `bash: venv/bin/activate: No such file or directory`, it means that the virtual environment was not created successfully. Ensure the python3-venv package is installed and try again.

### 4. Install Required Python Packages

With the virtual environment activated, install all the required dependencies using pip:

```bash
pip install -r requirements.txt
```

This will install all necessary libraries specified in requirements.txt, including:

- Streamlit (for the web interface)
- Pandas (for data manipulation)
- Matplotlib (for visualization)
- Seaborn (for statistical visualizations)
- CairoSVG (for chess board rendering)
- Python-chess (for chess game analysis)

### 5. Run the Streamlit Application

Once everything is set up, you can run the Streamlit app with the following command:

```bash
streamlit run main.py
```

This will launch the application and automatically open it in your default web browser.

### 6. Deactivate the Virtual Environment

When you're done working on the project, you can deactivate the virtual environment:

```bash
deactivate
```

## Troubleshooting

1. Error: `command not found: pip`

   - Solution: Ensure that python3-pip is installed:
     ```bash
     sudo apt install python3-pip
     ```

2. Error: `command not found: python3`

   - Solution: Ensure that Python 3 is installed:
     ```bash
     sudo apt install python3
     ```

3. Error: `No such file or directory: venv/bin/activate`

   - Solution: Make sure that the virtual environment was created successfully:
     ```bash
     python3 -m venv venv
     ```

4. Error: `streamlit: command not found`

   - Solution: If Streamlit isn't installed, run:
     ```bash
     pip install streamlit
     ```

5. Python version issues
   - For Ubuntu 20.04 or later, Python 3.x should be available by default. If needed, install Python 3.12:
     ```bash
     sudo apt install python3.12
     ```

## Built With

The following libraries and frameworks were used in the making of this Project:

- [Python Imaging Library (PIL)](https://pypi.org/project/Pillow/)
- [CairoSVG](https://pypi.org/project/CairoSVG/)
- [Tkinter](https://docs.python.org/3/library/tkinter.html)
- [ZipFile](https://docs.python.org/3/library/zipfile.html)
- [Python Requests Module](https://docs.python-requests.org/)
- [Pandas](https://pandas.pydata.org/)
- [Streamlit](https://streamlit.io/)

## License

[MIT License](LICENSE)

---

For more information or if you encounter any issues, please open an issue in the GitHub repository.
