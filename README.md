# Delivery Time Prediction for Logistics

## Project Overview

This project focuses on analyzing historical logistics delivery data to understand the factors that affect delivery time and eventually develop a machine learning model for delivery-time prediction.

The project uses the **Amazon Delivery Dataset** from Kaggle. The dataset contains information about delivery agents, locations, order and pickup times, weather, traffic, vehicles, areas, product categories, and delivery time.

The overall goal is to use data analysis and machine learning to support better logistics planning and delivery operations.

## Problem Statement

Delivery time can be affected by several factors such as traffic conditions, weather, delivery area, travel distance, and operational conditions.

This project aims to analyze these factors and identify patterns that can help predict delivery time for future deliveries.

The main question is:

> **Can historical delivery, traffic, weather, location, and operational data be used to understand and predict delivery times?**

## Project Objectives

- Understand delivery-time patterns.
- Identify factors that may influence delivery time.
- Analyze important logistics KPIs.
- Clean and preprocess the dataset.
- Explore relationships between delivery time and different features.
- Develop a machine learning model for delivery-time prediction.
- Provide data-driven recommendations for logistics planning.

## Dataset

**Dataset:** Amazon Delivery Dataset

**Source:** Kaggle

The dataset contains **43,739 delivery records and 16 columns**.

Important features include:

- Agent Age
- Agent Rating
- Store Latitude and Longitude
- Drop Latitude and Longitude
- Order Date
- Order Time
- Pickup Time
- Weather
- Traffic
- Vehicle
- Area
- Category
- Delivery Time

`Delivery_Time` is the main target variable for the future prediction model.

The dataset is used for educational and analytical purposes.

## Week 1: Strategic Planning & Data Exploration

The first stage of the project focuses on understanding the logistics problem, defining KPIs, researching existing approaches, and performing initial exploratory data analysis.

### Initial Analysis

The following relationships were explored:

- Traffic vs Delivery Time
- Weather vs Delivery Time
- Area vs Delivery Time
- Delivery distance using geographical coordinates

### Initial Findings

**Traffic:**  
Average delivery time generally increases as traffic becomes heavier. Low traffic has an average delivery time of approximately 101 minutes, while traffic jams have an average of approximately 148 minutes.

**Weather:**  
Delivery time varies across weather conditions. Sunny conditions show a lower average delivery time, while cloudy and foggy conditions show higher averages.

**Area:**  
Semi-Urban deliveries have a high average delivery time. However, this category contains far fewer records than Metropolitan and Urban areas, so the result should be interpreted carefully.

**Distance:**  
The distance between store and delivery locations was estimated using the Haversine formula. Some records contain suspicious geographical coordinates, which can produce unrealistic distances. These records will be investigated during the data cleaning stage.

## Project Roadmap

Dataset Collection
       ↓
Data Cleaning
       ↓
Data Preprocessing
       ↓
Exploratory Data Analysis
       ↓
KPI Analysis
       ↓
Feature Selection
       ↓
Predictive Modeling
       ↓
Model Evaluation
       ↓
Delivery Time Prediction
       ↓
Logistics Recommendations