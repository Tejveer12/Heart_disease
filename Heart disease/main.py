{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "688d692d",
   "metadata": {},
   "source": [
    "# Heart disease classification algoritm"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "c6eae3d5",
   "metadata": {},
   "source": [
    "### Read data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "a30f315c",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "a35a940e",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>age</th>\n",
       "      <th>sex</th>\n",
       "      <th>cp</th>\n",
       "      <th>trestbps</th>\n",
       "      <th>chol</th>\n",
       "      <th>fbs</th>\n",
       "      <th>restecg</th>\n",
       "      <th>thalach</th>\n",
       "      <th>exang</th>\n",
       "      <th>oldpeak</th>\n",
       "      <th>slope</th>\n",
       "      <th>ca</th>\n",
       "      <th>thal</th>\n",
       "      <th>target</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>52</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>125</td>\n",
       "      <td>212</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>168</td>\n",
       "      <td>0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "      <td>3</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>53</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>140</td>\n",
       "      <td>203</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>155</td>\n",
       "      <td>1</td>\n",
       "      <td>3.1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>70</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>145</td>\n",
       "      <td>174</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>125</td>\n",
       "      <td>1</td>\n",
       "      <td>2.6</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>61</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>148</td>\n",
       "      <td>203</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>161</td>\n",
       "      <td>0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>3</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>62</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>138</td>\n",
       "      <td>294</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>106</td>\n",
       "      <td>0</td>\n",
       "      <td>1.9</td>\n",
       "      <td>1</td>\n",
       "      <td>3</td>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   age  sex  cp  trestbps  chol  fbs  restecg  thalach  exang  oldpeak  slope  \\\n",
       "0   52    1   0       125   212    0        1      168      0      1.0      2   \n",
       "1   53    1   0       140   203    1        0      155      1      3.1      0   \n",
       "2   70    1   0       145   174    0        1      125      1      2.6      0   \n",
       "3   61    1   0       148   203    0        1      161      0      0.0      2   \n",
       "4   62    0   0       138   294    1        1      106      0      1.9      1   \n",
       "\n",
       "   ca  thal  target  \n",
       "0   2     3       0  \n",
       "1   0     3       0  \n",
       "2   0     3       0  \n",
       "3   1     3       0  \n",
       "4   3     2       0  "
      ]
     },
     "execution_count": 2,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "heart=pd.read_csv(\"heart.csv\")\n",
    "heart.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "8dfc9662",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 1025 entries, 0 to 1024\n",
      "Data columns (total 14 columns):\n",
      " #   Column    Non-Null Count  Dtype  \n",
      "---  ------    --------------  -----  \n",
      " 0   age       1025 non-null   int64  \n",
      " 1   sex       1025 non-null   int64  \n",
      " 2   cp        1025 non-null   int64  \n",
      " 3   trestbps  1025 non-null   int64  \n",
      " 4   chol      1025 non-null   int64  \n",
      " 5   fbs       1025 non-null   int64  \n",
      " 6   restecg   1025 non-null   int64  \n",
      " 7   thalach   1025 non-null   int64  \n",
      " 8   exang     1025 non-null   int64  \n",
      " 9   oldpeak   1025 non-null   float64\n",
      " 10  slope     1025 non-null   int64  \n",
      " 11  ca        1025 non-null   int64  \n",
      " 12  thal      1025 non-null   int64  \n",
      " 13  target    1025 non-null   int64  \n",
      "dtypes: float64(1), int64(13)\n",
      "memory usage: 112.2 KB\n"
     ]
    }
   ],
   "source": [
    "heart.info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "bf0ce32a",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>age</th>\n",
       "      <th>sex</th>\n",
       "      <th>cp</th>\n",
       "      <th>trestbps</th>\n",
       "      <th>chol</th>\n",
       "      <th>fbs</th>\n",
       "      <th>restecg</th>\n",
       "      <th>thalach</th>\n",
       "      <th>exang</th>\n",
       "      <th>oldpeak</th>\n",
       "      <th>slope</th>\n",
       "      <th>ca</th>\n",
       "      <th>thal</th>\n",
       "      <th>target</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>count</th>\n",
       "      <td>1025.000000</td>\n",
       "      <td>1025.000000</td>\n",
       "      <td>1025.000000</td>\n",
       "      <td>1025.000000</td>\n",
       "      <td>1025.00000</td>\n",
       "      <td>1025.000000</td>\n",
       "      <td>1025.000000</td>\n",
       "      <td>1025.000000</td>\n",
       "      <td>1025.000000</td>\n",
       "      <td>1025.000000</td>\n",
       "      <td>1025.000000</td>\n",
       "      <td>1025.000000</td>\n",
       "      <td>1025.000000</td>\n",
       "      <td>1025.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>mean</th>\n",
       "      <td>54.434146</td>\n",
       "      <td>0.695610</td>\n",
       "      <td>0.942439</td>\n",
       "      <td>131.611707</td>\n",
       "      <td>246.00000</td>\n",
       "      <td>0.149268</td>\n",
       "      <td>0.529756</td>\n",
       "      <td>149.114146</td>\n",
       "      <td>0.336585</td>\n",
       "      <td>1.071512</td>\n",
       "      <td>1.385366</td>\n",
       "      <td>0.754146</td>\n",
       "      <td>2.323902</td>\n",
       "      <td>0.513171</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>std</th>\n",
       "      <td>9.072290</td>\n",
       "      <td>0.460373</td>\n",
       "      <td>1.029641</td>\n",
       "      <td>17.516718</td>\n",
       "      <td>51.59251</td>\n",
       "      <td>0.356527</td>\n",
       "      <td>0.527878</td>\n",
       "      <td>23.005724</td>\n",
       "      <td>0.472772</td>\n",
       "      <td>1.175053</td>\n",
       "      <td>0.617755</td>\n",
       "      <td>1.030798</td>\n",
       "      <td>0.620660</td>\n",
       "      <td>0.500070</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>min</th>\n",
       "      <td>29.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>94.000000</td>\n",
       "      <td>126.00000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>71.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>25%</th>\n",
       "      <td>48.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>120.000000</td>\n",
       "      <td>211.00000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>132.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>2.000000</td>\n",
       "      <td>0.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>50%</th>\n",
       "      <td>56.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>130.000000</td>\n",
       "      <td>240.00000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>152.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.800000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>2.000000</td>\n",
       "      <td>1.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>75%</th>\n",
       "      <td>61.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>2.000000</td>\n",
       "      <td>140.000000</td>\n",
       "      <td>275.00000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>166.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>1.800000</td>\n",
       "      <td>2.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>3.000000</td>\n",
       "      <td>1.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>max</th>\n",
       "      <td>77.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>3.000000</td>\n",
       "      <td>200.000000</td>\n",
       "      <td>564.00000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>2.000000</td>\n",
       "      <td>202.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>6.200000</td>\n",
       "      <td>2.000000</td>\n",
       "      <td>4.000000</td>\n",
       "      <td>3.000000</td>\n",
       "      <td>1.000000</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "               age          sex           cp     trestbps        chol  \\\n",
       "count  1025.000000  1025.000000  1025.000000  1025.000000  1025.00000   \n",
       "mean     54.434146     0.695610     0.942439   131.611707   246.00000   \n",
       "std       9.072290     0.460373     1.029641    17.516718    51.59251   \n",
       "min      29.000000     0.000000     0.000000    94.000000   126.00000   \n",
       "25%      48.000000     0.000000     0.000000   120.000000   211.00000   \n",
       "50%      56.000000     1.000000     1.000000   130.000000   240.00000   \n",
       "75%      61.000000     1.000000     2.000000   140.000000   275.00000   \n",
       "max      77.000000     1.000000     3.000000   200.000000   564.00000   \n",
       "\n",
       "               fbs      restecg      thalach        exang      oldpeak  \\\n",
       "count  1025.000000  1025.000000  1025.000000  1025.000000  1025.000000   \n",
       "mean      0.149268     0.529756   149.114146     0.336585     1.071512   \n",
       "std       0.356527     0.527878    23.005724     0.472772     1.175053   \n",
       "min       0.000000     0.000000    71.000000     0.000000     0.000000   \n",
       "25%       0.000000     0.000000   132.000000     0.000000     0.000000   \n",
       "50%       0.000000     1.000000   152.000000     0.000000     0.800000   \n",
       "75%       0.000000     1.000000   166.000000     1.000000     1.800000   \n",
       "max       1.000000     2.000000   202.000000     1.000000     6.200000   \n",
       "\n",
       "             slope           ca         thal       target  \n",
       "count  1025.000000  1025.000000  1025.000000  1025.000000  \n",
       "mean      1.385366     0.754146     2.323902     0.513171  \n",
       "std       0.617755     1.030798     0.620660     0.500070  \n",
       "min       0.000000     0.000000     0.000000     0.000000  \n",
       "25%       1.000000     0.000000     2.000000     0.000000  \n",
       "50%       1.000000     0.000000     2.000000     1.000000  \n",
       "75%       2.000000     1.000000     3.000000     1.000000  \n",
       "max       2.000000     4.000000     3.000000     1.000000  "
      ]
     },
     "execution_count": 4,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "heart.describe()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "acf64b27",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "1    526\n",
       "0    499\n",
       "Name: target, dtype: int64"
      ]
     },
     "execution_count": 5,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "heart['target'].value_counts()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "0be874ac",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "age         0\n",
       "sex         0\n",
       "cp          0\n",
       "trestbps    0\n",
       "chol        0\n",
       "fbs         0\n",
       "restecg     0\n",
       "thalach     0\n",
       "exang       0\n",
       "oldpeak     0\n",
       "slope       0\n",
       "ca          0\n",
       "thal        0\n",
       "target      0\n",
       "dtype: int64"
      ]
     },
     "execution_count": 6,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "heart.isnull().sum()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "3b9486d5",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0    872\n",
       "1    153\n",
       "Name: fbs, dtype: int64"
      ]
     },
     "execution_count": 7,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "heart['fbs'].value_counts()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "c3b94444",
   "metadata": {},
   "source": [
    "### Train-Test split"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "ae054782",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.model_selection import StratifiedShuffleSplit\n",
    "split=StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)\n",
    "for train_index,test_index in split.split(heart,heart['fbs']):\n",
    "    train_set=heart.iloc[train_index]\n",
    "    test_set=heart.iloc[test_index]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "7f11d6ac",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0    698\n",
       "1    122\n",
       "Name: fbs, dtype: int64"
      ]
     },
     "execution_count": 9,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "train_set['fbs'].value_counts()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "eab1e559",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0    174\n",
       "1     31\n",
       "Name: fbs, dtype: int64"
      ]
     },
     "execution_count": 10,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "test_set['fbs'].value_counts()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "68e0d2b7",
   "metadata": {},
   "source": [
    "### Creating a pipeline"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "eb0a830b",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.pipeline import Pipeline\n",
    "from sklearn.impute import SimpleImputer \n",
    "my_pipeline=Pipeline(\n",
    "[\n",
    "    (\"imputer\",SimpleImputer(strategy=\"median\"))\n",
    "])"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "5a571c28",
   "metadata": {},
   "source": [
    "### Features and label"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "214df60a",
   "metadata": {},
   "outputs": [],
   "source": [
    "train_features = train_set.drop(\"target\",axis=1)\n",
    "train_target = train_set['target'].copy()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "5e24c384",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(820, 13)"
      ]
     },
     "execution_count": 13,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "train_features.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "4f871617",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(820,)"
      ]
     },
     "execution_count": 14,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "train_target.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "f5bae3d3",
   "metadata": {},
   "outputs": [],
   "source": [
    "train_features=my_pipeline.fit_transform(train_features)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "ea315211",
   "metadata": {},
   "source": [
    "### Selecting a desired model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "id": "b1d76473",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.linear_model import LogisticRegression\n",
    "from sklearn.neighbors import KNeighborsClassifier\n",
    "from sklearn.tree import DecisionTreeClassifier\n",
    "from sklearn.ensemble import RandomForestClassifier\n",
    "\n",
    "#model=LogisticRegression()\n",
    "#model=KNeighborsClassifier()\n",
    "model=DecisionTreeClassifier()\n",
    "#model=RandomForestClassifier()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "id": "9ce58934",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "DecisionTreeClassifier()"
      ]
     },
     "execution_count": 17,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "model.fit(train_features,train_target)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "id": "9a43aeaa",
   "metadata": {},
   "outputs": [],
   "source": [
    "pred=model.predict(train_features)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "id": "04c8131a",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "1.0"
      ]
     },
     "execution_count": 19,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from sklearn.metrics import accuracy_score,confusion_matrix\n",
    "\n",
    "accuracy_score(train_target,pred)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "id": "a7b3342e",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[396,   0],\n",
       "       [  0, 424]], dtype=int64)"
      ]
     },
     "execution_count": 20,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "confusion_matrix(train_target,pred)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "c3676d90",
   "metadata": {},
   "source": [
    "### Dumping model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "id": "193f7789",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "['heart_disease.joblib']"
      ]
     },
     "execution_count": 21,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from joblib import dump\n",
    "dump(model,\"heart_disease.joblib\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "id": "94a744bf",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Stored 'test_set' (DataFrame)\n"
     ]
    }
   ],
   "source": [
    "%store test_set"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "id": "7b7153a7",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Stored 'my_pipeline' (Pipeline)\n"
     ]
    }
   ],
   "source": [
    "%store my_pipeline"
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
