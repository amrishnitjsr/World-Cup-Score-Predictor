@echo off
echo Starting Cricket World Cup Score Predictor Dashboard...
echo.
echo Make sure you have run the Jupyter notebooks first to generate the required model files:
echo - data extraction.ipynb
echo - feature extraction and prediction.ipynb
echo.
echo Required files:
echo - pipe.pkl (trained model)
echo - dataset_level2.pkl (processed data)
echo.
pause
streamlit run streamlit_app.py