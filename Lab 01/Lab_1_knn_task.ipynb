{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Лабораторная работа №1. \"Разработка классификатора методом k ближайших соседей\""
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "1. Загрузить данные из файла heart.csv, используя функцию read_csv() библиотеки pandas"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    "data = pd.read_csv('diabetes.csv')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "2. Вывести несколько строк датафрейма. Более подробная информация о датасете https://www.kaggle.com/mathchi/diabetes-data-set"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
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
       "      <th>Pregnancies</th>\n",
       "      <th>Glucose</th>\n",
       "      <th>BloodPressure</th>\n",
       "      <th>SkinThickness</th>\n",
       "      <th>Insulin</th>\n",
       "      <th>BMI</th>\n",
       "      <th>DiabetesPedigreeFunction</th>\n",
       "      <th>Age</th>\n",
       "      <th>Outcome</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>6</td>\n",
       "      <td>148</td>\n",
       "      <td>72</td>\n",
       "      <td>35</td>\n",
       "      <td>0</td>\n",
       "      <td>33.6</td>\n",
       "      <td>0.627</td>\n",
       "      <td>50</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>1</td>\n",
       "      <td>85</td>\n",
       "      <td>66</td>\n",
       "      <td>29</td>\n",
       "      <td>0</td>\n",
       "      <td>26.6</td>\n",
       "      <td>0.351</td>\n",
       "      <td>31</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>8</td>\n",
       "      <td>183</td>\n",
       "      <td>64</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>23.3</td>\n",
       "      <td>0.672</td>\n",
       "      <td>32</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>1</td>\n",
       "      <td>89</td>\n",
       "      <td>66</td>\n",
       "      <td>23</td>\n",
       "      <td>94</td>\n",
       "      <td>28.1</td>\n",
       "      <td>0.167</td>\n",
       "      <td>21</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>0</td>\n",
       "      <td>137</td>\n",
       "      <td>40</td>\n",
       "      <td>35</td>\n",
       "      <td>168</td>\n",
       "      <td>43.1</td>\n",
       "      <td>2.288</td>\n",
       "      <td>33</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>763</th>\n",
       "      <td>10</td>\n",
       "      <td>101</td>\n",
       "      <td>76</td>\n",
       "      <td>48</td>\n",
       "      <td>180</td>\n",
       "      <td>32.9</td>\n",
       "      <td>0.171</td>\n",
       "      <td>63</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>764</th>\n",
       "      <td>2</td>\n",
       "      <td>122</td>\n",
       "      <td>70</td>\n",
       "      <td>27</td>\n",
       "      <td>0</td>\n",
       "      <td>36.8</td>\n",
       "      <td>0.340</td>\n",
       "      <td>27</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>765</th>\n",
       "      <td>5</td>\n",
       "      <td>121</td>\n",
       "      <td>72</td>\n",
       "      <td>23</td>\n",
       "      <td>112</td>\n",
       "      <td>26.2</td>\n",
       "      <td>0.245</td>\n",
       "      <td>30</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>766</th>\n",
       "      <td>1</td>\n",
       "      <td>126</td>\n",
       "      <td>60</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>30.1</td>\n",
       "      <td>0.349</td>\n",
       "      <td>47</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>767</th>\n",
       "      <td>1</td>\n",
       "      <td>93</td>\n",
       "      <td>70</td>\n",
       "      <td>31</td>\n",
       "      <td>0</td>\n",
       "      <td>30.4</td>\n",
       "      <td>0.315</td>\n",
       "      <td>23</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>768 rows × 9 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "     Pregnancies  Glucose  BloodPressure  SkinThickness  Insulin   BMI  \\\n",
       "0              6      148             72             35        0  33.6   \n",
       "1              1       85             66             29        0  26.6   \n",
       "2              8      183             64              0        0  23.3   \n",
       "3              1       89             66             23       94  28.1   \n",
       "4              0      137             40             35      168  43.1   \n",
       "..           ...      ...            ...            ...      ...   ...   \n",
       "763           10      101             76             48      180  32.9   \n",
       "764            2      122             70             27        0  36.8   \n",
       "765            5      121             72             23      112  26.2   \n",
       "766            1      126             60              0        0  30.1   \n",
       "767            1       93             70             31        0  30.4   \n",
       "\n",
       "     DiabetesPedigreeFunction  Age  Outcome  \n",
       "0                       0.627   50        1  \n",
       "1                       0.351   31        0  \n",
       "2                       0.672   32        1  \n",
       "3                       0.167   21        0  \n",
       "4                       2.288   33        1  \n",
       "..                        ...  ...      ...  \n",
       "763                     0.171   63        0  \n",
       "764                     0.340   27        0  \n",
       "765                     0.245   30        0  \n",
       "766                     0.349   47        1  \n",
       "767                     0.315   23        0  \n",
       "\n",
       "[768 rows x 9 columns]"
      ]
     },
     "execution_count": 3,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": []
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "3. Проверить сбалансированность классов. Можно воспользоваться методом value_counts() датафрейма. Для каждого класса нужно вывести долю примеров от общего числа"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Class 0: 0.651\n",
      "Class 1: 0.349\n"
     ]
    }
   ],
   "source": []
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "4. Провести процедуру нормирования признаков. Вариант 1: так, чтобы все признаки принимали значения в диапазоне $[0;1]$, используя преобразование $X^{*}=(X-X_{min})/{\\Delta{X}}$ для каждого признака. Вариант 2: так, чтобы все признаки принимали значения в диапазоне $[-3\\sigma;+3\\sigma]$, используя преобразование $X^{*}=(X-\\mu_{X})/{\\sigma_{X}}$ для каждого признака.\n",
    "Процедуру обезразмеривания оформить в виде функции"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [],
   "source": [
    "def normalization(X):\n",
    "    "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [],
   "source": [
    "X_norm = normalization(data.drop('Outcome', axis = 1))\n",
    "Y = data.Outcome"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
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
       "      <th>Pregnancies</th>\n",
       "      <th>Glucose</th>\n",
       "      <th>BloodPressure</th>\n",
       "      <th>SkinThickness</th>\n",
       "      <th>Insulin</th>\n",
       "      <th>BMI</th>\n",
       "      <th>DiabetesPedigreeFunction</th>\n",
       "      <th>Age</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>0.639530</td>\n",
       "      <td>0.847771</td>\n",
       "      <td>0.149543</td>\n",
       "      <td>0.906679</td>\n",
       "      <td>-0.692439</td>\n",
       "      <td>0.203880</td>\n",
       "      <td>0.468187</td>\n",
       "      <td>1.425067</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>-0.844335</td>\n",
       "      <td>-1.122665</td>\n",
       "      <td>-0.160441</td>\n",
       "      <td>0.530556</td>\n",
       "      <td>-0.692439</td>\n",
       "      <td>-0.683976</td>\n",
       "      <td>-0.364823</td>\n",
       "      <td>-0.190548</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>1.233077</td>\n",
       "      <td>1.942458</td>\n",
       "      <td>-0.263769</td>\n",
       "      <td>-1.287373</td>\n",
       "      <td>-0.692439</td>\n",
       "      <td>-1.102537</td>\n",
       "      <td>0.604004</td>\n",
       "      <td>-0.105515</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>-0.844335</td>\n",
       "      <td>-0.997558</td>\n",
       "      <td>-0.160441</td>\n",
       "      <td>0.154433</td>\n",
       "      <td>0.123221</td>\n",
       "      <td>-0.493721</td>\n",
       "      <td>-0.920163</td>\n",
       "      <td>-1.040871</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>-1.141108</td>\n",
       "      <td>0.503727</td>\n",
       "      <td>-1.503707</td>\n",
       "      <td>0.906679</td>\n",
       "      <td>0.765337</td>\n",
       "      <td>1.408828</td>\n",
       "      <td>5.481337</td>\n",
       "      <td>-0.020483</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>763</th>\n",
       "      <td>1.826623</td>\n",
       "      <td>-0.622237</td>\n",
       "      <td>0.356200</td>\n",
       "      <td>1.721613</td>\n",
       "      <td>0.869464</td>\n",
       "      <td>0.115094</td>\n",
       "      <td>-0.908090</td>\n",
       "      <td>2.530487</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>764</th>\n",
       "      <td>-0.547562</td>\n",
       "      <td>0.034575</td>\n",
       "      <td>0.046215</td>\n",
       "      <td>0.405181</td>\n",
       "      <td>-0.692439</td>\n",
       "      <td>0.609757</td>\n",
       "      <td>-0.398023</td>\n",
       "      <td>-0.530677</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>765</th>\n",
       "      <td>0.342757</td>\n",
       "      <td>0.003299</td>\n",
       "      <td>0.149543</td>\n",
       "      <td>0.154433</td>\n",
       "      <td>0.279412</td>\n",
       "      <td>-0.734711</td>\n",
       "      <td>-0.684747</td>\n",
       "      <td>-0.275580</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>766</th>\n",
       "      <td>-0.844335</td>\n",
       "      <td>0.159683</td>\n",
       "      <td>-0.470426</td>\n",
       "      <td>-1.287373</td>\n",
       "      <td>-0.692439</td>\n",
       "      <td>-0.240048</td>\n",
       "      <td>-0.370859</td>\n",
       "      <td>1.169970</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>767</th>\n",
       "      <td>-0.844335</td>\n",
       "      <td>-0.872451</td>\n",
       "      <td>0.046215</td>\n",
       "      <td>0.655930</td>\n",
       "      <td>-0.692439</td>\n",
       "      <td>-0.201997</td>\n",
       "      <td>-0.473476</td>\n",
       "      <td>-0.870806</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>768 rows × 8 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "     Pregnancies   Glucose  BloodPressure  SkinThickness   Insulin       BMI  \\\n",
       "0       0.639530  0.847771       0.149543       0.906679 -0.692439  0.203880   \n",
       "1      -0.844335 -1.122665      -0.160441       0.530556 -0.692439 -0.683976   \n",
       "2       1.233077  1.942458      -0.263769      -1.287373 -0.692439 -1.102537   \n",
       "3      -0.844335 -0.997558      -0.160441       0.154433  0.123221 -0.493721   \n",
       "4      -1.141108  0.503727      -1.503707       0.906679  0.765337  1.408828   \n",
       "..           ...       ...            ...            ...       ...       ...   \n",
       "763     1.826623 -0.622237       0.356200       1.721613  0.869464  0.115094   \n",
       "764    -0.547562  0.034575       0.046215       0.405181 -0.692439  0.609757   \n",
       "765     0.342757  0.003299       0.149543       0.154433  0.279412 -0.734711   \n",
       "766    -0.844335  0.159683      -0.470426      -1.287373 -0.692439 -0.240048   \n",
       "767    -0.844335 -0.872451       0.046215       0.655930 -0.692439 -0.201997   \n",
       "\n",
       "     DiabetesPedigreeFunction       Age  \n",
       "0                    0.468187  1.425067  \n",
       "1                   -0.364823 -0.190548  \n",
       "2                    0.604004 -0.105515  \n",
       "3                   -0.920163 -1.040871  \n",
       "4                    5.481337 -0.020483  \n",
       "..                        ...       ...  \n",
       "763                 -0.908090  2.530487  \n",
       "764                 -0.398023 -0.530677  \n",
       "765                 -0.684747 -0.275580  \n",
       "766                 -0.370859  1.169970  \n",
       "767                 -0.473476 -0.870806  \n",
       "\n",
       "[768 rows x 8 columns]"
      ]
     },
     "execution_count": 7,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "X_norm"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "5. Разделить выборку на три части: обучающую, валидационную и тестовую в соотношении 0.6:0.2:0.2. Выбор из исходной выборки должен осуществляться случайным образом. Можно воспользоваться функцией choice() библиотеки numpy.random. Можно также воспользоваться специальной функцией train_test_split() библиотеки sklearn.model_selection. Проверить сбалансированность классов в полученных выборках."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Class 0: 0.643\n",
      "Class 1: 0.357\n",
      "\n",
      "Class 0: 0.675\n",
      "Class 1: 0.325\n",
      "\n",
      "Class 0: 0.649\n",
      "Class 1: 0.351\n",
      "\n"
     ]
    }
   ],
   "source": [
    "from sklearn.model_selection import train_test_split\n",
    "\n",
    "    "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "6. Написать функцию knn_classification(X,y,X_new,k), реализующую метод k-ближайших соседей для задачи классификации. Параметрами функции являются: X,y - обучающая выборка, X_new - массив (или датафрейм) признаков для новых объектов, k - гиперпараметр метода (кол-во ближайших соседей, в функции предусмотреть возможность задания k по умолчанию). Функция должна возвращать прогноз для новых объектов в виде массива классов (y_pred)."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [],
   "source": [
    "def knn(X, Y, X_new, k):\n",
    "\n",
    "    return Y_pred\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "7. Написать функцию для расчета метрик качества модели классификации. Функция должна возвращать значения оценок Precision, Recall, F1-score."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [],
   "source": [
    "def fscore(y, y_pred):\n",
    "\n",
    "    return prec, rec, f1_score, accuracy"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "8. Выполнить предсказания классов для валидационной выборки для произвольного гиперпараметра k, рассчитать и вывести метрики Precision, Recall, F1-score и Accuracy."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "prec: 0.792, recall: 0.380, f1_score: 0.514, accuracy: 0.766\n"
     ]
    }
   ],
   "source": [
    "Y_val_pred = knn(X_train, Y_train, X_val, k=2)\n",
    "prec, recall, f1_score, accuracy = \n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "9. Задать массив или список значений гиперпараметра k (например, от 2 до 10 с шагом 2). Провести процедуру валидации, т.е. подбора оптимального значения k по критерию максимума точности. Для оценки точности использовать F1-score метрику. Выводить для каждого значения k полученное значение f1 score. В конце вывести лучший результат."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "k: 2, f1 score: 0.5135135135135135\n",
      "k: 4, f1 score: 0.5066666666666666\n",
      "k: 6, f1 score: 0.5609756097560976\n",
      "k: 8, f1 score: 0.5121951219512195\n",
      "\n",
      "Best k: 6, best val f1 score: 0.5609756097560976\n"
     ]
    }
   ],
   "source": [
    "K = [i for i in range(2,10,2)]\n",
    "\n",
    "for k in K:\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "10. Рассчитать оценки точности (accuracy, precision, recal, F1-score) классификатора для тестовой выборки с оптимальным значением гиперпараметра k. Сделать вывод о качестве классификатора."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "prec: 0.683, recall: 0.519, f1_score: 0.589, accuracy: 0.747\n"
     ]
    }
   ],
   "source": []
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "11. Сравнить результаты с функцией KNeighborsClassifier из библиотеки sklearn. Если получился другой результат подумать в чем дело."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "prec: 0.683, recall: 0.519, f1_score: 0.589, accuracy: 0.747\n"
     ]
    }
   ],
   "source": [
    "from sklearn.neighbors import KNeighborsClassifier\n",
    "\n",
    "clf = KNeighborsClassifier(n_neighbors=best_k, metric='euclidean')\n",
    "clf.fit(X_train, Y_train)\n",
    "Y_test_pred = clf.predict(X_test)\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
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
   "version": "3.8.8"
  },
  "varInspector": {
   "cols": {
    "lenName": 16,
    "lenType": 16,
    "lenVar": 40
   },
   "kernels_config": {
    "python": {
     "delete_cmd_postfix": "",
     "delete_cmd_prefix": "del ",
     "library": "var_list.py",
     "varRefreshCmd": "print(var_dic_list())"
    },
    "r": {
     "delete_cmd_postfix": ") ",
     "delete_cmd_prefix": "rm(",
     "library": "var_list.r",
     "varRefreshCmd": "cat(var_dic_list()) "
    }
   },
   "types_to_exclude": [
    "module",
    "function",
    "builtin_function_or_method",
    "instance",
    "_Feature"
   ],
   "window_display": false
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
