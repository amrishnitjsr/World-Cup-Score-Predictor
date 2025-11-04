# 📸 How to Add Screenshots to README

## Quick Steps:
1. Take screenshots of your running Streamlit dashboard
2. Save them as actual `.png` image files (not text files!)
3. Place them in the `assets/` folder with these exact names:
   - `dashboard-preview-1.png` - Input form with team selection
   - `dashboard-preview-2.png` - Prediction results with metric cards  
   - `dashboard-preview-3.png` - Win probability gauge and analysis

## Then Update README:
Replace the text descriptions with actual image links like:

```markdown
<!-- Example when you have real image files: -->
![Dashboard Screenshot](assets/dashboard-preview-1.png)
*Beautiful cricket prediction interface*
```

## Alternative: Use GitHub Issues for Image Hosting
1. Create a new issue in your GitHub repository
2. Drag & drop your screenshot images into the issue description
3. GitHub will generate permanent URLs like: `https://github.com/user-attachments/assets/xyz.png`
4. Use those URLs in your README instead of local file paths

The README currently describes your dashboard features in text - once you add real images, they'll display perfectly! 🎯