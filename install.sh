#!/bin/bash

# Navigate to the project directory
cd /var/webapps/daisy_backend

# Set proxy environment variables
export http_proxy=http://150.6.13.63:3128
export https_proxy=http://150.6.13.63:3128

# Install the required libraries
pip3.12 install -r requirements.txt
