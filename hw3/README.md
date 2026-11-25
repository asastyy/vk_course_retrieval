# ДЗ 3. Обучение ранжированию

- Kaggle — `asanastasiia`
- Публичный nDCG@10 — **0,62409**
- Бейзлайн соревнования — 0,61621
- nDCG@10 на 1 400 отложенных запросах — **0,622724**

## Параметры модели

LightGBM LambdaRank с процентильными рангами и z-нормировкой признаков внутри запроса. Валидация по `QueryId` с долей 20%.

| Параметр | Значение |
|---|---:|
| Итерации | 183 |
| `num_leaves` | 31 |
| `min_data_in_leaf` | 60 |
| `learning_rate` | 0.035 |
| `lambda_l2` | 5 |
| `feature_fraction` | 0.85 |
| `bagging_fraction` | 0.85 |
| `bagging_freq` | 1 |
| `lambdarank_truncation_level` | 13 |
| `seed` | 42 |
