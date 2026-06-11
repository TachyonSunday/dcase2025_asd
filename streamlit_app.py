"""Streamlit Cloud 部署入口。"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from ui.app import main
main()
