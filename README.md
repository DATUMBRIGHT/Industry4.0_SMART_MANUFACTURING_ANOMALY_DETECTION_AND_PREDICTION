# Industry 4.0 Smart Manufacturing Anomaly Detection and Prediction

## 🚀 Project Overview

This project demonstrates a modern, end-to-end Industry 4.0 anomaly detection and predictive maintenance pipeline using real manufacturing sensor data, Kafka streaming, AWS, and robust data validation.

- **Data Source:** Real IoT sensor data from 50 machines, sampled every minute
- **Streaming:** Apache Kafka (Dockerized)
- **Data Validation:** Great Expectations
- **Cloud Integration:** AWS S3, Glue, Athena, QuickSight, SageMaker
- **ML:** PatchTST (TensorFlow implementation for time series forecasting and anomaly detection)
- **Real-Time Inference:** SageMaker endpoint + Lambda for live anomaly alerts

---

## 🏗️ Architecture

```mermaid
graph TD
    A[Kafka Producer] --> B[Kafka Topic]
    B --> C[Kafka Consumer]
    C --> D[S3 Bucket]
    D --> E[Glue Crawler]
    E --> F[Athena/QuickSight]
    D --> G[PatchTST Model Training (TFpatch)]
    G --> H[SageMaker Endpoint]
    B --> I[Lambda/Consumer]
    I --> H
    H --> J[S3 (Predictions/Alerts)]
    H --> K[QuickSight (Live Analytics)]
```

---

## 📦 Features

- **Kafka Streaming:** Real-time ingestion and distribution of sensor data
- **Data Preprocessing:** Outlier handling, normalization, one-hot encoding
- **Data Validation:** Automated checks with Great Expectations (nulls, ranges, one-hot, etc.)
- **Cloud-Ready:** Data lands in S3, cataloged with Glue, queryable with Athena, visualized in QuickSight
- **ML-Ready:** PatchTST (TFpatch) for time series prediction and anomaly detection
- **Model Training:** Train PatchTST on historical data, save model to S3, deploy to SageMaker
- **Real-Time Inference:** Deployed model on SageMaker, triggered by Lambda for live alerts

---

## 📝 How to Run

1. **Clone the repo**
   ```bash
   git clone https://github.com/DATUMBRIGHT/Industry4.0_SMART_MANUFACTURING_ANOMALY_DETECTION_AND_PREDICTION.git
   cd Industry4.0_SMART_MANUFACTURING_ANOMALY_DETECTION_AND_PREDICTION
   ```

2. **Start Kafka and Zookeeper**
   ```bash
   docker compose up -d
   ```

3. **Process and Validate Data**
   ```bash
   python src/preprocessor.py
   python src/main.py  # or your validation script
   ```

4. **Train PatchTST Model (TFpatch)**
   - See `notebooks/model_training_patchtst.ipynb` for code and workflow.
   - Example (Python):
     ```python
     from tfpatch import PatchTST
     # Load processed data
     # Prepare time series windows
     # Train PatchTST model
     # Save model to S3
     ```

5. **Deploy Model to SageMaker**
   - Use SageMaker Python SDK to deploy the trained PatchTST model as a real-time endpoint.

6. **Set Up Real-Time Inference**
   - Use Lambda or a microservice to call the SageMaker endpoint as new data arrives from Kafka or S3.

7. **Query and Visualize**
   - Use Glue Crawler to catalog S3 data.
   - Query with Athena.
   - Visualize with QuickSight.

---

## 📊 Example Data Validation Rules

- All sensor columns are numeric and within expected ranges
- One-hot encoded columns sum to 1 per row
- `machine_id` between 1 and 50
- `anomaly_flag` is 0 or 1
- `timestamp` is a valid datetime

---

## 💡 Why This Project Stands Out

- **End-to-end pipeline:** From raw data to real-time ML inference
- **Modern stack:** Kafka, AWS, PatchTST (TFpatch), Great Expectations
- **Production-ready:** Dockerized, cloud-integrated, robust validation
- **Recruiter-friendly:** Clean code, clear documentation, real-world use case

---

## 📚 Skills Demonstrated

- Data engineering (streaming, ETL, validation)
- Machine learning (time series, anomaly detection, PatchTST)
- Cloud architecture (AWS S3, Glue, Athena, SageMaker)
- DevOps (Docker, CI/CD ready)
- Communication (clear code and documentation)

---

## 📂 Project Structure

```
.
├── docker-compose.yml
├── src/
│   ├── preprocessor.py
│   ├── main.py
│   └── my_expectations/
│       └── validation_rules.py
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   └── model_training_patchtst.ipynb
└── README.md
```

---

**This README structure will clearly show recruiters your full-stack data/ML engineering skills, including the use of PatchTST (TFpatch) for model training and deployment in a modern, cloud-based pipeline.**  
If you want a sample PatchTST training code snippet for the README, let me know!

---

## 📬 Contact

- [Your Name](mailto:your.email@example.com)
- [LinkedIn](https://www.linkedin.com/in/yourprofile)

---

## **Step-by-Step: Push Your Project to GitHub**

### **1. Make Sure Your Project Structure Looks Like This:**

```
Industry4.0_SMART_MANUFACTURING_ANOMALY_DETECTION_AND_PREDICTION/
├── docker-compose.yml
├── src/
│   ├── preprocessor.py
│   ├── main.py
│   └── my_expectations/
│       └── validation_rules.py
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   └── model_training_patchtst.ipynb
├── README.md
└── requirements.txt
```

- Make sure your README.md is detailed as described above.
- Ensure your code is clean and all files are included (except large raw data, if any).

---

### **2. Initialize Git (if not already done):**

```bash
git init
git remote add origin https://github.com/DATUMBRIGHT/Industry4.0_SMART_MANUFACTURING_ANOMALY_DETECTION_AND_PREDICTION.git
```

---

### **3. Add All Files:**

```bash
<code_block_to_apply_changes_from>
```

---

### **4. Commit with a Clear Message:**

```bash
git commit -m "kafka initialized, data processed and saved, great_expectations created, PatchTST training and deployment pipeline added"
```

---

### **5. Push to GitHub:**

```bash
git branch -M main
git push -u origin main
```

---

### **6. Double-Check on GitHub**
- Go to your repo: https://github.com/DATUMBRIGHT/Industry4.0_SMART_MANUFACTURING_ANOMALY_DETECTION_AND_PREDICTION
- Make sure all files and the README are visible and formatted correctly.

---

## **What Recruiters Will See**

- **README.md**: Clear project overview, architecture diagram, and step-by-step instructions.
- **Code**: All pipeline components (Kafka, preprocessing, validation, PatchTST training, deployment scripts).
- **Notebooks**: For model training and experimentation.
- **Docker Compose**: For easy local setup.
- **Validation scripts**: Showing your data quality focus.

---

**You are now ready to impress recruiters with a professional, end-to-end, cloud-ready ML pipeline!**

If you need a sample PatchTST training notebook or want to check your README before pushing, let me know!
