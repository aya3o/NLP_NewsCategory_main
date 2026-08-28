# NLP_NewsCategory_main
Benchmarking_Traditional_and_Transformer_Based_Models_for_News_Topics 

# NLP News Category Classification & Topic Modeling
# تصنيف فئات الأخبار واستخراج المواضيع

**Date / التاريخ:** 20/01/2026  
**Course / المقرر:** Natural Language Processing - Final Project

---

## 📌 Overview / نظرة عامة

This project performs **News Classification** and **Topic Modeling** on the HuffPost News Category Dataset (210K articles). It compares traditional ML models (SVM, Random Forest, MLP) with deep learning (DistilBERT) across multiple text representations (TFIDF, Word2Vec, LDA Topics).

يقوم هذا المشروع بـ **تصنيف الأخبار** و **استخراج المواضيع** من مجموعة بيانات HuffPost (210 ألف خبر). يقارن بين نماذج التعلم الآلي التقليدية (SVM, Random Forest, MLP) والتعلم العميق (DistilBERT) عبر عدة تمثيلات نصية (TFIDF, Word2Vec, مواضيع LDA).

---

## 🗂️ Project Structure / هيكل المشروع

```
NLP_NewsCategory-main/
├── Dataset/
│   ├── News_Category_Dataset_v3.json       ← Original dataset (فك الضغط هنا)
│   └── Final_NewsCorpus_Clean.json         ← Generated after preprocessing (يُنشأ تلقائياً)
├── distilbert-base-uncased-finetuned/      ← DistilBERT checkpoints (ignored by Git)
├── W2V/
│   └── fasttext_100.kv                     ← Word2Vec vectors (يُنشأ تلقائياً)
├── lda_final_model/                        ← Final LDA model (يُنشأ تلقائياً)
├── lda_topics.html                         ← LDA visualization (يُنشأ تلقائياً)
├── coherence_scores.pkl                    ← LDA coherence scores (يُنشأ تلقائياً)
├── *.pkl                                   ← Saved ML models (يُنشأ تلقائياً)
├── *.model / *.state / *.npy              ← LDA model files (ignored by Git)
├── app.py                                  ← Flask app (if applicable)
├── static/ / templates/ / flask/           ← Web app files
├── Figures/                                ← Generated figures
└── results/                                ← Training outputs
```



## ⚙️ Requirements / المتطلبات

- Python 3.8+
- Java 8 (JDK) — *only if using Mallet (currently disabled)*
- 8GB+ RAM (16GB recommended for DistilBERT)
- GPU recommended for DistilBERT training

### Python Libraries / المكتبات
```bash
pip install numpy pandas matplotlib seaborn scikit-learn
pip install gensim==3.8.3
pip install pyLDAvis==2.1.2
pip install spacy
pip install transformers datasets evaluate
pip install torch  # or tensorflow
pip install joblib
pip install termcolor
python -m spacy download en_core_web_md
```

---

## 🚀 Setup & Running / الإعداد والتشغيل

### 1. Extract Dataset / فك ضغط البيانات
Extract `Dataset.zip` into the `Dataset/` folder so the path becomes:
فك ضغط `Dataset.zip` داخل مجلد `Dataset/` بحيث يصبح المسار:

```
Dataset/News_Category_Dataset_v3.json
```

---

### 2. Update Project Path / تحديث مسار المشروع
In the first code cell, update this line to match your local machine:
في أول خلية كود، عدّل هذا السطر حسب جهازك:

```python
path_to_folder = r"C:\Users\YOUR_NAME\...\NLP_NewsCategory-main"
os.chdir(path_to_folder)
```

---

### 3. Run Order / تسلسل التشغيل

| Step | Description | Cells to Run |
|------|-------------|--------------|
| 1 | Imports & Setup | Run all import cells |
| 2 | Data Acquisition | Run preprocessing (run **once only**) |
| 3 | Data Loading | Run Section 1 cells |
| 4 | Vectorization | Run Section 2 cells (TFIDF, Word2Vec, LDA) |
| 5 | Classification | Run Section 3 cells (ML models) |
| 6 | Transformers | Run DistilBERT cells (run **once only**) |
| 7 | Evaluation | Run evaluation & confusion matrix cells |

---

## ⚠️ One-Time Execution Cells / خلايا تُشغل مرة واحدة فقط

These cells are **commented out** in the notebook. Uncomment them, run **once**, then comment them back. They create model files that are auto-loaded on subsequent runs.

هذه الخلايا **مُعلّقة** في المفكرة. أزل التعليق عنها، شغّلها **مرة واحدة**، ثم أعد تعليقها. تقوم بإنشاء ملفات الموديلات التي تُحمّل تلقائياً في المرات القادمة.

| Cell / الخلية | Generates / تُنشئ | File / الملف |
|--------------|-------------------|--------------|
| SpaCy Preprocessing | Cleaned corpus | `Dataset/Final_NewsCorpus_Clean.json` |
| FastText Training | Word vectors | `W2V/fasttext_100.kv` |
| LDA Coherence Search | LDA models + scores | `coherence_scores.pkl`, `lda_model_*.model` |
| Final LDA Training | Best LDA model | `lda_final_model/` |
| SVM (TFIDF) | SVM classifier | `svm_tfidf_rbf_model.pkl` |
| MLP (TFIDF) | Neural network | `mlp_tfidf_model.pkl` |
| SVM (Word2Vec) | SVM on embeddings | `svm_w2v_rbf_model_fast.pkl` |
| Random Forest (W2V) | RF classifier | `rf_w2v_model.pkl` |
| MLP (Word2Vec) | MLP on embeddings | `mlp_w2v_model.pkl` |
| Linear SVM (LDA) | SVM on topics | `linear_svm_lda_model.pkl` |
| DistilBERT Training | Transformer model | `C:/Models/my_model/` |

> **Logic:** The code checks `if os.path.exists(path)` — if the file exists, it loads it; otherwise, it trains and saves it.  
> **المنطق:** الكود يتحقق `if os.path.exists(path)` — إذا الملف موجود يُحمّله، وإلا يُدربه ويحفظه.

---

## 📊 Results Summary / ملخص النتائج

| Model / النموذج | Representation / التمثيل | Accuracy / الدقة |
|----------------|------------------------|-----------------|
| SVM (RBF) | TFIDF (10% data) | 75.85% |
| Random Forest | TFIDF | 66.16% |
| **MLP** | **TFIDF** | **81.19%** |
| SVM (RBF) | Word2Vec (10% data) | 79.51% |
| Random Forest | Word2Vec | 69.49% |
| MLP | Word2Vec | 81.05% |
| Linear SVM | LDA Topics | 68.34% |
| Random Forest | LDA Topics | 66.16% |
| MLP | LDA Topics | 68.49% |
| **DistilBERT** | **Transformer** | **86.34%** |

**Best Model / أفضل نموذج:** DistilBERT (86.34% Accuracy, Precision, Recall, F1 ≈ 0.86)

---

## 🔧 Important Notes / ملاحظات مهمة

1. **Dataset Size / حجم البيانات:**  
   The original dataset contains **209,527** articles with **42 categories**. We simplified it to **4 classes**: OTHER (0), POLITICS (1), WELLNESS (2), ENTERTAINMENT (3).

2. **SVM Training Limitation / محدودية تدريب SVM:**  
   SVM with RBF kernel has quadratic complexity. Training on the full dataset (>200K samples) is computationally impractical. Therefore, a **10% subset** is used for training while the full test set is used for evaluation.  
   SVM بنواة RBF له تعقيد تربيعي. التدريب على البيانات الكاملة (>200 ألف عينة) غير عملي. لذا يُستخدم **10% فقط** للتدريب مع الاختبار على كامل البيانات.

3. **DistilBERT / المحول:**  
   - Training takes ~16 hours on CPU. Use GPU for faster training.  
   - The model is saved to `C:/Models/my_model/` by default. Change the path in the code if needed.  
   - التدريب يستغرق ~16 ساعة على CPU. استخدم GPU للتسريع.  
   - النموذج يُحفظ في `C:/Models/my_model/` افتراضياً. غيّر المسار في الكود إذا لزم الأمر.

4. **LDA Topic Modeling / نماذج LDA:**  
   - Number of topics searched: [15, 20, 30, 35, 40, 45, 50, 60, 70]  
   - Best coherence achieved at **70 topics**  
   - Topics identified: Covid-19, Russia-Ukraine war, US Elections, Donald Trump, Hollywood, Gun shootings, Climate change, etc.

5. **Preprocessing / المعالجة المسبقة:**  
   - Lemmatization, stopword removal, POS filtering (NOUN, VERB, ADJ, PROPN)  
   - Bigram detection: `donald_trump`, `white_house`, `climate_change`  
   - Custom stopwords for news domain (huffpost, reporter, click, etc.)

---

## 📝 Citation / الاقتباس

Dataset: [News Category Dataset](https://www.kaggle.com/datasets/rmisra/news-category-dataset) — HuffPost, 2012-2022

---

Natural Language Processing Course — Final Project  
Date: 20/01/2026
```
