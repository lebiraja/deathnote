# Model card (v2.0.0)

## Data
NHANES 2007-08 to 2017-18 with the NCHS 2019 public-use linked mortality files. US adults aged 20-79 who were examined, mortality-eligible and had complete answers: **30,423 people, 2,042 deaths**, followed to 31 December 2019. Public mortality files perturb some follow-up values.

## Method
1. Weighted Gompertz proportional-hazards model, age as time scale with delayed entry, separate baseline per sex, small ridge penalty.
2. At serving time only the relative effects are kept. Baseline death rates come from India's [SRS Abridged Life Tables 2020-24](https://censusindia.gov.in/nada/index.php/catalog/47148/download/51393/SRS-Abridged_Life_Tables_2020-2024.pdf), continued past 85 with a Gompertz slope.
3. A person's risk is divided by the average Indian adult's, using tobacco prevalence (GATS 2016-17) and hypertension and diabetes prevalence (NFHS-5, from a secondary analysis, PMC12364263).
4. Headline: median age at death. Range: 25th to 75th percentile. This is spread between similar people, not model uncertainty.
5. Factor breakdown: the gap to a reference person, split in proportion to each factor's log risk.

## Accuracy (US data only)
| C-index | |
|---|---|
| 5-fold cross-validation | 0.828 |
| Held-out 2017-18 cycle (70 deaths, short follow-up) | 0.822 |
| Age and sex only | 0.791 |
| Without the coefficient limits | 0.829 |

| Risk fifth | 5-year predicted / observed | 10-year (2007-10) predicted / observed |
|---|---|---|
| 1 | 0.2% / 0.3% | 0.5% / 0.6% |
| 2 | 0.6% / 0.7% | 1.6% / 1.4% |
| 3 | 1.6% / 1.7% | 3.9% / 3.7% |
| 4 | 4.1% / 3.4% | 9.7% / 10.3% |
| 5 | 14.0% / 12.1% | 30.3% / 31.9% |

These figures describe the Gompertz model on US participants. **The India-adjusted lifespans have not been validated**: no Indian individual-level follow-up data was used.

## Coefficient limits
A worse answer may never add years. Underweight is capped at a hazard ratio of 1.5 per 3 BMI units below 18.5 (Global BMI Mortality Collaboration, Lancet 2016); the free fit was several times higher. Sleep is unrestricted (U-shaped).

Where the data pointed the other way, the effect is now zero: BMI 23-30, total cholesterol, light and moderate drinking, and some adjacent diet and exercise levels. Zero means "not shown by this model", not "harmless".

## Other limits
- Factor effects are from US adults and are associations, often distorted by illness changing habits.
- "Former drinker" carries high risk partly because people stop when ill, so what-if does not offer it.
- Chewing tobacco: hazard ratio 1.35 (men), 1.37 (women) versus never users, Mumbai Cohort Study as tabulated in [Shaikh et al. 2025](https://pmc.ncbi.nlm.nih.gov/articles/PMC12641177/) (ages 35+, education-adjusted). Applied to non-smokers only.
- Sleep was asked differently before and after 2015 (mean 6.86 h against 7.61 h).
- Bedtime clock time has no effect.
- Total risk is clipped to the 0.5-99.5 percentile range seen in training.
- The life table includes the COVID-19 years. National rates hide large differences between states, urban and rural areas and income groups.
- Relative effects are held constant at all ages, which likely overstates them in old age.
- Prevalences are treated as independent, and factors without an Indian figure are assumed at the reference level, so the "average Indian" scaling is approximate.
