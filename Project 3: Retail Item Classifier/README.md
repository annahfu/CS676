---
title: Retail Item Classifier
emoji: 📈
colorFrom: indigo
colorTo: yellow
sdk: gradio
sdk_version: 6.0.1
app_file: app.py
pinned: false
license: mit
---

## Retail Item Hierarchy Classifier

Automatically categorize retail items into their correct store sections using Machine Learning! Link to project: https://huggingface.co/spaces/annahfu1/retail-item-classifier

## What Does It Do?

This ML-powered application helps retail workers and inventory managers quickly classify items into the correct store sections. Simply enter an item name (like "bananas" or "chicken breast"), and the model will predict which section it belongs to with a confidence score.

## Why I chose this project:

I created this retail item classifier because in the retail industry there are thousands of SKUs and store sections to keep track of, and it can be challenging to quickly understand where every item belongs. In my day-to-day work, I constantly see how important it is to connect items to the right department, aisle, and category in order to analyze performance, build assortments, and understand shopper behavior.
This tool is my way of turning that problem into a practical solution: by using machine learning to map item descriptions to store sections, it helps accelerate the process of learning the category structure, reduces manual lookup, and creates a starting point for cleaner, more structured retail data. It is not just a model for prediction—it is also a learning and exploration tool for anyone working with retail data who wants to better understand where items fit within the store. 

## Use Cases

- **Retail Workers**: Quickly determine where to stock new items
- **Inventory Management**: Automate item categorization
- **E-commerce**: Auto-categorize products for online stores
- **Training**: Help new employees learn store layouts

## Technical Stack

- **Python**: Core programming language
- **scikit-learn**: Machine learning model
- **Gradio**: User interface framework
- **pandas**: Data processing
- **numpy**: Numerical computations
- - **Hugging Face Spaces**: Hosting the app


## Model Training

The model was trained on a comprehensive dataset of retail items covering all major product categories found in US supermarkets. It uses:
- TF-IDF (Term Frequency-Inverse Document Frequency) for text feature extraction
- Random Forest ensemble method for robust predictions
- N-gram analysis (1-3 words) to capture multi-word product names

  
## Features

- **Real-time Classification**: Instant predictions for any retail item
- **Confidence Scores**: Visual indicators showing prediction reliability
- **Batch Processing**: Upload CSV files to classify hundreds of items at once
- **16 Store Sections**: Based on standard US retail layouts (Whole Foods, Kroger, H-E-B, Walmart, Target, etc.)

## Store Sections

The classifier recognizes these 16 sections:
- Produce
- Meat & Seafood
- Dairy & Eggs
- Frozen Foods
- Bakery
- Deli & Prepared Foods
- Pantry & Dry Goods
- Beverages
- Snacks & Candy
- International Foods
- Health & Wellness
- Personal Care
- Household & Cleaning
- Baby & Infant
- Pet Supplies
- Floral

## Model Details

- **Algorithm**: Random Forest Classifier
- **Features**: TF-IDF vectorization with n-grams (1-3)
- **Training Data**: 280+ labeled retail items
- **Test Accuracy**: ~85-95%

## How to Use

### Single Item Classification
1. Go to the "Single Item Classification" tab
2. Enter an item name (e.g., "organic bananas")
3. Click "Classify Item"
4. View the predicted section and confidence score

### Batch Classification
1. Go to the "Batch Classification" tab
2. Upload a CSV file with an 'item' column
3. Click "Process Batch"
4. Download the results with all classifications


## Results and conclusions

This project demonstrates how a machine learning model combined with rule-based logic can accurately classify retail items into detailed store sections, even from messy or complex product names. By automating item-to-category mapping, the tool reduces manual work, improves data quality, and helps analysts quickly learn the structure of retail departments. Deployed on Hugging Face, it provides a fast, accessible way to standardize retail data and support real-world analytics workflows.