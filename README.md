```markdown
# Project Spotify Dashboard by Group 3

## Description
This repo contains dashboard.py which is an interactive Spotify data dashboard built using Python and Streamlit. It allows users to explore, analyze, and visualize Spotify music data, including track features, artist trends, and listening patterns.

The goal of this project is to provide meaningful insights into Spotify datasets through data analysis and interactive visualizations.

---

## Features
- Interactive dashboard built with Streamlit  
- Data exploration and filtering  
- Visualization of music trends  
- Analysis of Spotify track features  

---

## Tech Stack
- matplotlib
- numpy
- pandas
- PuLP
- scipy
- seaborn
- streamlit  

---

## Project Structure
```

Project-Spotify-Group-3/
│── dashboard.py          # Main Streamlit application
│── requirements.txt      # Project dependencies
│── data/                 # Dataset files 
│── analysis.py           # Data Analysis Python file from Part 1 of the assignment           
│── database_analysis.py  # Database Analysis Python file from Parts 2 and 3 of the assignment
│── data_wrangling.py     # Data Wrangling Python file from Part 4 of the assignment

---

## Setup Instructions

1. Install **Python 3.13**

2. Clone the repository:
```bash
git clone <your-repository-url>
cd Project-Spotify-Group-3
````

3. Create a virtual environment:

```bash
python -m venv venv
```

4. Activate the virtual environment:

* **Windows:**

```bash
venv\Scripts\activate
```

* **Mac/Linux:**

```bash
source venv/bin/activate
```

5. Install dependencies:

```bash
pip install -r requirements.txt
```

6. Verify installation:

```bash
pip list
```

7. Run the dashboard:

```bash
streamlit run dashboard.py
```

---

## Requirements

Make sure the following libraries are installed (included in `requirements.txt`):

* matplotlib
* numpy
* pandas
* PuLP
* scipy
* seaborn
* streamlit

---

## Output

The application launches a local Streamlit dashboard in your browser where you can interact with Spotify data visualizations.

---

## Authors

Project developed as part of a university assignment by:
- Sander van den Berg
- Serena Cederboom
- Gisele Henry
- Anton Ruban

---

## License

This project is for educational purposes only.

```
```
