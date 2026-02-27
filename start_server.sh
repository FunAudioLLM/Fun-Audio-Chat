#!/bin/bash
cd ~/Fun-Audio-Chat
source venv/bin/activate
export PATH=$HOME/bin:$PATH
export PYTHONPATH=$PWD:$PYTHONPATH
python simple_server.py
