# 🎉 PhotoMind Web Demo - Quick Start

## 🚀 Launch the Web Interface

**Option 1: One-Click Launch (Recommended)**
```bash
start_web_demo.bat
```

**Option 2: Manual Launch**
```bash
cd d:\Downloads\Project1_Photomind
set PYTHONPATH=src
conda activate clipsex
python demos\web_app.py
```

Then open your browser to: **http://localhost:5000**

---

## ✨ What You Can Do

### 🔍 Search Tab
- Type queries like **"a person"**, **"outdoor scene"**, **"text document"**
- Choose search method: **NumPy**, **FAISS**, or **Compare Both**
- Select top **3, 5, or 8** results
- See **similarity scores** and **search time**
- View **actual images** from your `my_photos` folder

### ⚡ Performance Tab
- Benchmark **NumPy vs FAISS** performance
- Run **5, 10, 20, or 50 trials** for accuracy
- See **mean, std dev, min, max** query times
- Calculate **speedup** (how much faster FAISS is)
- Compare different **FAISS index types** (Flat/IVF/HNSW)

### 🖼️ Gallery Tab
- Browse **all 8 images** in your collection
- View **thumbnails** with image names
- See total vs displayed count

---

## 📊 Live Dashboard

The header shows:
- **Total Images**: 8
- **CLIP Model**: ViT-B-32::laion2b_s34b_b79k
- **NumPy Index**: ✅ Loaded
- **FAISS Index**: ❌ Not loaded (build with `--faiss` flag)

---

## 🎨 Features

✅ **Beautiful UI** - Modern gradient design, card-based layout  
✅ **Visual Results** - See your photos, not just filenames  
✅ **Real-time Metrics** - Search time in milliseconds  
✅ **Side-by-side Comparison** - NumPy vs FAISS results  
✅ **Interactive** - Try different queries instantly  
✅ **Professional** - Smooth animations, hover effects  

---

## 📸 Sample Queries

Try these with your photos:
- `"a person"` → Find images with people
- `"outdoor scene"` → Natural landscapes
- `"text document"` → Documents or papers
- `"colorful"` → Vibrant images
- `"dark"` → Low-light photos

---

## 🛠️ Troubleshooting

**Server won't start?**
```bash
# Install Flask
pip install flask
```

**Port 5000 busy?**
Edit `demos/web_app.py` line 301:
```python
app.run(debug=False, host='0.0.0.0', port=5001)  # Change to 5001
```

**Images not showing?**
Rebuild your index:
```bash
python scripts\build_index.py my_photos index_out
```

**Want FAISS comparison?**
Build FAISS index:
```bash
python scripts\build_index.py my_photos index_out --faiss
```

---

## 📖 Full Documentation

- **Web Demo Guide**: `demos/WEB_DEMO_README.md`
- **CLI Demos**: `demos/README.md`
- **Main README**: `readme.md`

---

## 🎯 What This Proves

✅ **Phase 1 Works**: Natural language search with CLIP  
✅ **Phase 2 Works**: FAISS integration and optimization  
✅ **Real Data**: Uses your actual `my_photos` images  
✅ **Production Ready**: Clean UI, error handling, performance metrics  

---

## 🔮 Next Steps

After you confirm the web demo works:
- **Phase 3**: Add explainability (Grad-CAM heatmaps)
- **Phase 4**: Add feedback and personalization
- **Configuration**: YAML configs for easy customization
- **Quality Gates**: Comprehensive testing and validation

---

**Server Status**: 
- ✅ Running on http://localhost:5000
- ✅ CLIP model loaded
- ✅ 8 images indexed
- 🎉 Ready to search!

**Press Ctrl+C in terminal to stop the server**
