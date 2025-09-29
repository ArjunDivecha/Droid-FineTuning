# 🚀 Quick Install: GSPO & Dr. GRPO Training Methods

## 📥 **Get the Enhanced Features**

You need to download these files to your Droid-FineTuning directory:

### **Required Files:**
1. `gspo-dr-grpo-integration.patch` - Contains all the enhanced training methods
2. `apply-enhancements.sh` - Auto-install script (Mac/Linux)
3. `apply-enhancements.bat` - Auto-install script (Windows)

## 🔧 **Installation Methods**

### **Method 1: Automatic Installation (Recommended)**

#### **For Mac/Linux:**
```bash
# 1. Navigate to your Droid-FineTuning directory
cd /path/to/your/Droid-FineTuning

# 2. Make the script executable
chmod +x apply-enhancements.sh

# 3. Run the installation script
./apply-enhancements.sh
```

#### **For Windows:**
```cmd
# 1. Navigate to your Droid-FineTuning directory
cd C:\path\to\your\Droid-FineTuning

# 2. Run the installation script
apply-enhancements.bat
```

### **Method 2: Manual Installation**

If the automatic script doesn't work:

```bash
# 1. Navigate to your project
cd /path/to/your/Droid-FineTuning

# 2. Create backup
git checkout -b backup-before-enhancements
git checkout main

# 3. Apply the patch
git apply gspo-dr-grpo-integration.patch

# 4. Create feature branch
git checkout -b feature/gspo-dr-grpo-integration
git add .
git commit -m "Add GSPO and Dr. GRPO training methods"

# 5. Test installation
python test_integration.py
```

## ✅ **What You Get**

After installation, you'll have:

### **4 Training Methods:**
- ✅ **SFT** - Enhanced supervised fine-tuning
- 🆕 **GSPO** - 2x faster reasoning training
- 🆕 **Dr. GRPO** - Domain expert reasoning
- 🆕 **GRPO** - Multi-step reasoning

### **Enhanced Features:**
- 🎨 Beautiful method selection interface
- 🔍 Automatic data validation
- 📊 Resource estimation
- 🧪 Sample data generation
- 📚 Comprehensive documentation

## 🚀 **Start Using**

After installation:

```bash
# 1. Install dependencies
pip install -r backend/requirements.txt
cd frontend && npm install && cd ..

# 2. Start backend (Terminal 1)
cd backend && python main.py

# 3. Start frontend (Terminal 2)
cd frontend && npm start

# 4. Open browser
# Go to: http://localhost:3000
```

## 📚 **Documentation**

- `INSTALLATION_GUIDE.md` - Complete setup instructions
- `ENHANCED_TRAINING_METHODS.md` - How to use new methods
- `INTEGRATION_SUMMARY.md` - Technical details

## 🔄 **Make It Permanent**

Once you've tested and everything works:

```bash
# Merge the enhancements
git checkout main
git merge feature/gspo-dr-grpo-integration

# Now GSPO and Dr. GRPO are permanent!
```

## 🔙 **Rollback (If Needed)**

If something goes wrong:

```bash
# Go back to your original version
git checkout backup-before-enhancements
```

## 🆘 **Need Help?**

1. **Check the logs** in your terminal for error messages
2. **Run the test** with `python test_integration.py`
3. **Read the documentation** files for detailed instructions
4. **Make sure** you have Python 3.8+ and Node.js 16+ installed

## 🎉 **Success Indicators**

You'll know it worked when:
- ✅ `git branch` shows `feature/gspo-dr-grpo-integration`
- ✅ You can see 4 training methods in the UI
- ✅ The enhanced setup page loads properly
- ✅ Method selection works and shows different configurations

**Ready to unlock advanced MLX training capabilities! 🚀**