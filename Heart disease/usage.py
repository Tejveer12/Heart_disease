{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "207f549d",
   "metadata": {},
   "outputs": [],
   "source": [
    "from joblib import load\n",
    "model=load(\"heart_disease.joblib\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "ce22d1eb",
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "input=np.array([[ 54. ,   1. ,   2. , 125. , 273. ,   0. ,   0. , 152. ,   0. ,\n",
    "         0.5,   0. ,   1. ,   2. ]])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "b73466f4",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([1], dtype=int64)"
      ]
     },
     "execution_count": 3,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "model.predict(input)"
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
   "version": "3.9.12"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
