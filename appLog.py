{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "373c8a92-b38d-4eda-8b0b-eb16f8a52537",
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "import os\n",
    "import sys\n",
    "import logging\n",
    "\n",
    "logging_str = \"[%(asctime)s: %(levelname)s: %(module)s: %(message)s]\"\n",
    "\n",
    "log_dir = \"logs\"\n",
    "log_filepath = os.path.join(log_dir,\"running_logs.log\")\n",
    "os.makedirs(log_dir, exist_ok=True)\n",
    "\n",
    "\n",
    "logging.basicConfig(\n",
    "    level= logging.INFO,\n",
    "    format= logging_str,\n",
    "\n",
    "    handlers=[\n",
    "        logging.FileHandler(log_filepath),\n",
    "        logging.StreamHandler(sys.stdout)\n",
    "    ]\n",
    ")\n",
    "\n",
    "logger = logging.getLogger(\"cnnClassifierLogger\")"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.4"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
