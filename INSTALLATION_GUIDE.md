# 🚀 Complete Installation Guide: GSPO & Dr. GRPO Training Methods

## 📋 **What You're Installing**

You're adding **4 advanced training methods** to your MLX fine-tuning system:
- **SFT**: Standard fine-tuning (enhanced)
- **GSPO**: 2x faster reasoning training 🆕
- **Dr. GRPO**: Domain-expert reasoning 🆕  
- **GRPO**: Multi-step reasoning

## ✅ **Step 1: Prerequisites Check**

Make sure you have these installed on your computer:

### Required Software
```bash
# Check Python (need 3.8+)
python --version
# Should show: Python 3.8.x or higher

# Check Node.js (need 16+)  
node --version
# Should show: v16.x.x or higher

# Check Git
git --version
# Should show: git version 2.x.x
```

### If Missing Any Prerequisites:
- **Python**: Download from [python.org](https://python.org)
- **Node.js**: Download from [nodejs.org](https://nodejs.org) 
- **Git**: Download from [git-scm.com](https://git-scm.com)

## 📁 **Step 2: Navigate to Your Project**

Open your terminal/command prompt and go to your project folder:

```bash
# Replace with your actual path
cd /path/to/your/Droid-FineTuning

# Or if you cloned from GitHub:
cd Droid-FineTuning

# Confirm you're in the right place
ls
# Should see: backend/, frontend/, README.md, etc.
```

## 🔄 **Step 3: Switch to the Enhanced Version**

Currently you're on the "main" version. We need to switch to the "enhanced" version:

```bash
# See what versions are available
git branch -a

# Switch to the enhanced version
git checkout feature/gspo-dr-grpo-integration

# Confirm you're on the enhanced version
git branch
# Should show: * feature/gspo-dr-grpo-integration
```

## 🔧 **Step 4: Install Backend Dependencies**

Navigate to the backend folder and install required packages:

```bash
# Go to backend folder
cd backend

# Install Python packages
pip install -r requirements.txt

# If you get permission errors, try:
pip install --user -r requirements.txt

# Go back to main folder
cd ..
```

### If You Get Errors:
```bash
# Create a virtual environment (recommended)
python -m venv venv

# Activate it (Windows)
venv\\Scripts\\activate

# Activate it (Mac/Linux)  
source venv/bin/activate

# Then install packages
pip install -r backend/requirements.txt
```

## 🎨 **Step 5: Install Frontend Dependencies**

Navigate to the frontend folder and install packages:

```bash
# Go to frontend folder
cd frontend

# Install Node.js packages (this may take a few minutes)
npm install

# If you get errors, try:
npm install --legacy-peer-deps

# Go back to main folder
cd ..
```

## 🧪 **Step 6: Test the Installation**

Let's make sure everything is working:

```bash
# Run the integration test
python test_integration.py

# You should see:
# 🚀 Starting GSPO and Dr. GRPO integration tests...
# ✅ All training methods configured correctly
# ✅ Resource estimation working correctly  
# ✅ Data validation working correctly
# 🎉 All integration tests passed!
```

## 🚀 **Step 7: Start the Application**

Now let's start both the backend and frontend:

### Terminal/Window 1 - Backend:
```bash
# Make sure you're in the main project folder
cd backend

# Start the backend server
python main.py

# You should see:
# INFO:     Started server process
# INFO:     Uvicorn running on http://0.0.0.0:8000
```

### Terminal/Window 2 - Frontend:
```bash
# Open a NEW terminal window/tab
# Navigate to your project again
cd /path/to/your/Droid-FineTuning

# Go to frontend folder
cd frontend

# Start the frontend
npm start

# You should see:
# webpack compiled successfully
# Local: http://localhost:3000
```

## 🎯 **Step 8: Access the Enhanced Features**

1. **Open your web browser**
2. **Go to**: `http://localhost:3000`
3. **Look for**: "Enhanced Setup" or "Enhanced Training" page
4. **You should see**: Method selection with GSPO, Dr. GRPO, etc.

## 🧪 **Step 9: Test the New Features**

### Test Method Selection:
1. Navigate to the Enhanced Setup page
2. Click on different training methods (GSPO, Dr. GRPO)
3. See the method descriptions and parameters
4. Try the "Generate Sample" button

### Test Resource Estimation:
1. Enter a model path (any text for testing)
2. Select a training method  
3. Watch the resource estimation update
4. See recommendations appear

## 🔍 **Step 10: Verify Everything Works**

### Backend Check:
Visit `http://localhost:8000/health` in your browser
- Should show: `{"status":"healthy","timestamp":"..."}`

### Enhanced API Check:
Visit `http://localhost:8000/api/training/methods` in your browser
- Should show JSON with training methods including "gspo" and "dr_grpo"

### Frontend Check:
- The enhanced setup page should load without errors
- Method selection should work
- Forms should validate properly

## 🚨 **Common Issues & Solutions**

### Issue: "Module not found" errors
**Solution**: Make sure you installed all dependencies
```bash
# Backend
cd backend && pip install -r requirements.txt

# Frontend  
cd frontend && npm install
```

### Issue: Port already in use
**Solution**: Kill the existing process
```bash
# Find what's using port 8000
lsof -i :8000

# Kill it (replace PID with actual number)
kill -9 PID

# Or use different port
python main.py --port 8001
```

### Issue: Permission denied
**Solution**: Use virtual environment
```bash
python -m venv venv
source venv/bin/activate  # Mac/Linux
# or
venv\\Scripts\\activate   # Windows
```

### Issue: Frontend won't start
**Solution**: Clear cache and reinstall
```bash
cd frontend
rm -rf node_modules package-lock.json
npm install
npm start
```

## 📱 **What You Should See**

### Enhanced Setup Page:
- Beautiful cards for each training method
- GSPO with "🆕 Most Efficient" badge
- Dr. GRPO with "🆕 Domain Expert" badge  
- Method-specific configuration panels
- Real-time validation and resource estimation

### New Training Methods:
- **GSPO**: Sparse optimization parameters
- **Dr. GRPO**: Domain and expertise selection
- **GRPO**: Reasoning step configuration
- **SFT**: Enhanced with new features

## 🎉 **Success! What Now?**

Once everything is running:

1. **Explore the Methods**: Try selecting different training methods
2. **Generate Sample Data**: Use the sample data generation feature
3. **Test with Real Data**: Try with your actual training datasets
4. **Read the Documentation**: Check `ENHANCED_TRAINING_METHODS.md`

## 🆘 **Need Help?**

### Check These Files:
- `ENHANCED_TRAINING_METHODS.md` - Complete user guide
- `INTEGRATION_SUMMARY.md` - Technical details
- `test_integration.py` - Run this if something breaks

### Logs to Check:
- Backend terminal: Shows API errors and requests
- Frontend terminal: Shows build and runtime errors
- Browser console: Shows JavaScript errors (F12 → Console)

## 🔄 **Next Steps (Optional)**

### Make It Permanent:
Once you've tested and everything works, you can make this the permanent version:

```bash
# Switch to main branch
git checkout main

# Merge the enhanced features
git merge feature/gspo-dr-grpo-integration

# Now the enhancements are permanent
```

### Future Updates:
The architecture supports adding more training methods like:
- DPO (Direct Preference Optimization)
- CPO (Contrastive Preference Optimization)  
- RLHF (Reinforcement Learning from Human Feedback)

## 🎯 **Quick Start Summary**

For experienced users, here's the TL;DR:

```bash
# 1. Switch to enhanced branch
git checkout feature/gspo-dr-grpo-integration

# 2. Install dependencies  
pip install -r backend/requirements.txt
cd frontend && npm install && cd ..

# 3. Test integration
python test_integration.py

# 4. Start servers
# Terminal 1:
cd backend && python main.py

# Terminal 2:  
cd frontend && npm start

# 5. Visit http://localhost:3000
```

**You now have GSPO and Dr. GRPO integrated! 🚀**