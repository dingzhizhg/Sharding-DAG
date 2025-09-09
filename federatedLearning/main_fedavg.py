#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

"""
FedAvg Client Entry Point
这个文件作为FedAvg客户端的入口点，可以直接运行或调用fedAvg/clientRun.py
"""

import sys
import os

# 添加fedAvg路径（从federatedLearning目录到fedAvg目录）
fedavg_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'fedAvg')
sys.path.insert(0, fedavg_path)

# 如果直接运行此文件，则调用clientRun
if __name__ == '__main__':
    from clientRun import main
    
    print(f"Starting FedAvg client...")
    print("Note: Each client will randomly select devices for each iteration.")
    main()

