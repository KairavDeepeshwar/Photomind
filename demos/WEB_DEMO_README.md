# PhotoMind Web Demo 🌐

A beautiful, interactive web interface to explore PhotoMind's Phase 1 & 2 features with your actual photos!

## 🚀 Quick Start

### Option 1: Using Batch File (Recommended)
```bash
start_web_demo.bat
```

### Option 2: Manual Start
```bash
cd d:\Downloads\Project1_Photomind
set PYTHONPATH=src
conda activate clipsex
pip install flask  # if not already installed
python demos/web_app.py
```

Then open your browser to: **http://localhost:5000**

## ✨ Features

### 🔍 Search Tab
- **Natural language search** - Type queries like "a person", "outdoor scene", "text document"
- **Multiple search modes**:
  - NumPy Search (exact brute-force)
  - FAISS Search (optimized ANN)
  - Compare Both (side-by-side comparison)
- **Visual results** - See your photos with similarity scores
- **Adjustable K** - Choose top 3, 5, or 8 results
- **Real-time performance metrics** - See search time in milliseconds

### ⚡ Performance Tab
- **Benchmark NumPy vs FAISS** with configurable trials (5, 10, 20, 50)
- **Detailed statistics**:
  - Mean, standard deviation, min, max query times
  - Speedup calculation
- **Compare index types** - See FAISS index type in use
- **Reproducible benchmarks** - Multiple trials for accurate measurements

### 🖼️ Gallery Tab
- **View all indexed photos** - Visual gallery of your collection
- **Image thumbnails** - Automatic resizing for web display
- **Image count** - See total vs displayed images
- **Load on demand** - Click to load gallery

### 📊 Live Statistics
Header displays:
- Total indexed images
- CLIP model info
- NumPy index status
- FAISS index status and type

## 🎨 Interface Features

- **Modern, responsive design** - Gradient background, card-based layout
- **Color-coded results** - Easy-to-read similarity scores
- **Tabbed navigation** - Switch between Search, Performance, Gallery
- **Hover effects** - Interactive cards with smooth animations
- **Loading indicators** - Visual feedback during operations
- **Error handling** - Clear error messages if something goes wrong

## 📸 Example Queries to Try

With your `my_photos` directory:
- `"a person"` - Find images with people
- `"outdoor scene"` - Natural landscapes
- `"text document"` - Documents or papers
- `"colorful"` - Vibrant images
- `"dark"` - Low-light photos
- `"close up"` - Macro/detail shots

## 🔧 Technical Details

### Backend (Flask)
- **web_app.py** - Flask application with 5 API endpoints:
  - `/` - Serve main HTML page
  - `/api/gallery` - Get all indexed images
  - `/api/search` - Perform search (NumPy/FAISS/both)
  - `/api/stats` - Get system statistics
  - `/api/compare` - Benchmark performance

### Frontend (HTML/CSS/JavaScript)
- **templates/index.html** - Single-page application
- Pure JavaScript (no frameworks needed)
- Responsive grid layout
- Base64 image embedding for instant display

### Features Demonstrated
✅ **Phase 1:**
- Natural language search with CLIP
- NumPy-based exact search
- Image loading and display
- Result ranking

✅ **Phase 2:**
- FAISS index integration
- Multiple index types (Flat/IVF/HNSW)
- Performance comparison
- Speedup metrics

## 📦 Dependencies

The web demo requires Flask:
```bash
pip install flask
```

All other dependencies should already be installed from previous phases:
- torch
- open-clip-torch
- pillow
- numpy
- faiss-cpu

## 🎯 Usage Examples

### Basic Search
1. Go to **Search** tab
2. Select **NumPy Search**
3. Type `"a person"` in search box
4. Click **Search**
5. View results with scores and images

### Performance Comparison
1. Go to **Performance** tab
2. Enter a query (e.g., `"outdoor"`)
3. Select **10 trials**
4. Click **Run Benchmark**
5. Compare NumPy vs FAISS statistics

### Side-by-Side Search
1. Go to **Search** tab
2. Select **Compare Both**
3. Type a query
4. Click **Search**
5. See NumPy and FAISS results side-by-side with timing

## 🖼️ Screenshots

### Search Interface
- Clean search bar with method selector
- Visual result cards with images
- Similarity scores displayed prominently
- Search time metrics

### Performance Benchmark
- Side-by-side comparison cards
- Mean, std dev, min, max times
- Speedup calculation
- Index type information

### Gallery View
- Grid of all indexed photos
- Thumbnail previews
- Image names and paths
- Total count display

## 🛠️ Troubleshooting

### Port Already in Use
If port 5000 is busy, edit `web_app.py`:
```python
app.run(debug=True, host='0.0.0.0', port=5001)  # Change to 5001
```

### Flask Not Found
Install Flask:
```bash
pip install flask
```

### Images Not Loading
Ensure your index is built:
```bash
python scripts/build_index.py my_photos index_out
```

### FAISS Index Not Available
Build FAISS index:
```bash
python scripts/build_index.py my_photos index_out --faiss
```

## 🚀 Performance Tips

- **Small datasets (<100 images)**: NumPy is fast enough
- **Medium datasets (100-10K)**: FAISS Flat shows slight improvement
- **Large datasets (>10K)**: FAISS IVF/HNSW show significant speedup
- **First search**: May be slower due to model loading (subsequent searches are faster)

## 📝 API Reference

### POST /api/search
```json
Request:
{
  "query": "a person",
  "k": 5,
  "method": "both"  // "numpy", "faiss", or "both"
}

Response:
{
  "numpy": {
    "results": [...],
    "elapsed_ms": 1.23,
    "count": 5
  },
  "faiss": {
    "results": [...],
    "elapsed_ms": 0.85,
    "count": 5,
    "index_type": "Flat"
  }
}
```

### POST /api/compare
```json
Request:
{
  "query": "outdoor",
  "k": 5,
  "trials": 10
}

Response:
{
  "numpy": {
    "mean_ms": 1.234,
    "std_ms": 0.123,
    "min_ms": 1.100,
    "max_ms": 1.500
  },
  "faiss": {
    "mean_ms": 0.856,
    "std_ms": 0.089,
    "min_ms": 0.750,
    "max_ms": 1.000,
    "index_type": "Flat"
  },
  "speedup": 1.44
}
```

### GET /api/stats
```json
Response:
{
  "numpy_loaded": true,
  "faiss_loaded": true,
  "clip_loaded": true,
  "num_images": 8,
  "numpy_dim": 512,
  "faiss_index_type": "Flat",
  "faiss_ntotal": 8,
  "clip_model": "ViT-B-32::laion2b_s34b_b79k"
}
```

### GET /api/gallery
```json
Response:
{
  "total": 8,
  "displayed": 8,
  "images": [
    {
      "name": "image1.png",
      "path": "d:\\...\\my_photos\\image1.png",
      "image_b64": "data:image/png;base64,..."
    },
    ...
  ]
}
```

## 🎉 What Makes This Demo Special

- ✅ **Visual proof** - See your actual photos and search results
- ✅ **Interactive** - Try different queries and methods
- ✅ **Educational** - Understand NumPy vs FAISS performance
- ✅ **Professional UI** - Clean, modern design
- ✅ **Real data** - Uses your actual `my_photos` directory
- ✅ **No dependencies** - Pure HTML/CSS/JS frontend
- ✅ **Fast setup** - Single batch file to start

## 📊 Use Cases

1. **Demo for stakeholders** - Show working prototype visually
2. **Testing queries** - Experiment with different searches
3. **Performance analysis** - Compare search methods
4. **Educational** - Learn how CLIP search works
5. **Debugging** - Verify index contains correct images

## 🔮 Future Enhancements (Phase 3 & 4)

This web interface will be extended with:
- **Explainability**: Visual attention heatmaps showing what CLIP "sees"
- **Feedback**: Thumbs up/down for results
- **Personalization**: Learn from user preferences
- **Advanced filters**: Date, location, format filters
- **Batch operations**: Multi-image search, collections

---

**Created:** October 30, 2025  
**Technology:** Flask + Pure JavaScript  
**Status:** ✅ Ready to use  
**Next:** Phase 3 - Explainability Module
