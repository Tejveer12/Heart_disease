{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "445553bf",
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
   "id": "93568cd6",
   "metadata": {},
   "outputs": [],
   "source": [
    "%store -r test_set\n",
    "%store -r my_pipeline"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "b6fefc12",
   "metadata": {},
   "outputs": [],
   "source": [
    "test_features = test_set.drop(\"target\",axis=1)\n",
    "test_target = test_set['target'].copy()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "6362f239",
   "metadata": {},
   "outputs": [],
   "source": [
    "train_features=my_pipeline.fit_transform(test_features)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "20739ecd",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\Tejveer\\anaconda3\\lib\\site-packages\\sklearn\\base.py:443: UserWarning: X has feature names, but DecisionTreeClassifier was fitted without feature names\n",
      "  warnings.warn(\n"
     ]
    }
   ],
   "source": [
    "pred=model.predict(test_features)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "ae07d5eb",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "1.0"
      ]
     },
     "execution_count": 6,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from sklearn.metrics import accuracy_score,confusion_matrix\n",
    "\n",
    "accuracy_score(test_target,pred)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "c7ad2a33",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[103,   0],\n",
       "       [  0, 102]], dtype=int64)"
      ]
     },
     "execution_count": 7,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "confusion_matrix(test_target,pred)"
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
