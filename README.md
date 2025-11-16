
# **🌸 CerviScan – Website**

CerviScan is a web-based application designed to support **early cervical pre-cancer detection** using **Visual Inspection with Acetic Acid (VIA)** images. This system integrates a machine learning model developed by our team to classify VIA images into **normal** or **abnormal** categories.

The model achieves the following performance:

* **Accuracy:** 90.91%
* **Precision:** 93.75%
* **Specificity:** 93.75%
* **Recall:** 88.24%
* **F1-score:** 90.91%

Full machine learning development:
👉 **[https://github.com/FillipusAditya/cerviscan-cervical-cancer-detection](https://github.com/FillipusAditya/cerviscan-cervical-cancer-detection)**

---

## **🧠 Machine Learning Model Overview**

* **Model:** XGBoost
* **Segmentation Technique:** Multi-Otsu Thresholding
* **Feature Extraction:**

  * Color Moment (YUV)
  * Texture: GLRLM, Tamura, LBP
* **Dataset:** IARC Colposcopy Image Bank

  * Final: **162 training images** (75 abnormal, 87 normal)

---

## **🛠️ Technology Stack**

* HTML, CSS
* Python Flask

---

## **✨ Website Features**

* User Registration
* Login & Logout
* Patient Profile Input (first name, last name, date of birth)
* Upload VIA images for detection
* Visualization of detection output:

  * Uploaded image
  * Grayscale image
  * Segmentation mask
  * Segmented result
  * Final detection result
* Detection history page (view & delete past results)
* Usage guide page

---

## **👩‍🏫 Supervisors**

1. Prof. Dr. Eng. Ir. Retno Supriyanti, S.T., M.T.
2. Mohammad Irham Akbar, S.Kom., M.Cs.
3. Yogi Ramadhani, S.T., M.Eng.
4. Katon Muhammad, S.T., M.T.

---

## **🏥 Medical Partners**

1. dr. Futiat Diana Kartika
2. Kartika Dwi Hapsari, S.Tr.Keb

---

## **👨‍💻 Development Team**

*Electrical Engineering, Universitas Jenderal Soedirman*

1. Fillipus Aditya Nugroho
2. M. Saujana Shafi Kehaulani
3. Tegar Dwi Agung Saputra
4. M. Rizqy Maulana Sarwono

---

## **🎓 Supported by**

Kementerian Pendidikan Tinggi, Sains dan Teknologi
Scheme: **"Penelitian Terapan"** 2025–2026

---

## **📌 Important Note**

This repository is part of an *undergraduate thesis project* and is intended **for academic and research purposes only**.
The dataset from **IARC Colposcopy Image Bank** is **not redistributed** within this repository and **must be obtained directly from the original source** according to its usage license.

---

## **🚀 How to Run the Website Locally**

### **1. Clone the Repository**

```bash
git clone https://github.com/FillipusAditya/cerviscan-website.git
```

### **2. Enter the Project Directory**

```bash
cd cerviscan-website
```

### **3. Install Dependencies**

```bash
pip install -r requirements.txt
```

### **4. Run the Flask App**

```bash
python app.py
```

### **5. Open in Browser**

The application will run at:

```
http://127.0.0.1:5000/
```

---

## **🙏 Thank You!**

Thank you for your interest in **CerviScan – Website!** 🧬✨

For any questions or collaboration opportunities, feel free to reach out.
