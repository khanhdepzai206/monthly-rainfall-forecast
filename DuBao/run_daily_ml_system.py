#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Wrapper: chạy từ thư mục DuBao — python run_daily_ml_system.py train"""
import os
import sys

if __name__ == "__main__":
    os.chdir(os.path.join(os.path.dirname(__file__), "src"))
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
    from daily_ml_system.cli import main
    sys.exit(main())
