export const factorLabels: Record<string, string> = {
  sleep: 'Sleep', smoking: 'Smoking', chewing_tobacco: 'Chewing tobacco', alcohol: 'Alcohol',
  activity: 'Physical activity', diet: 'Diet', weight: 'Weight', diabetes: 'Diabetes',
  hypertension: 'High blood pressure (diagnosed)', heart_disease: 'Heart disease', asthma: 'Asthma',
  blood_pressure: 'Blood pressure reading', cholesterol: 'Cholesterol',
}

type Option = { value: string; label: string; hint?: string }

export const smokingOptions: Option[] = [
  { value: 'never', label: 'Never', hint: 'Under 100 in your life' },
  { value: 'former', label: 'Quit' },
  { value: 'current', label: 'I smoke', hint: 'Cigarettes or bidis' },
]
export const alcoholOptions: Option[] = [
  { value: 'never', label: 'Never' },
  { value: 'former', label: 'Not in the past year' },
  { value: 'light', label: 'Up to 7 a week' },
  { value: 'moderate', label: '8 to 14 a week' },
  { value: 'heavy', label: '15 or more a week' },
]
export const dietOptions: Option[] = [
  { value: 'excellent', label: 'Excellent' },
  { value: 'very_good', label: 'Very good' },
  { value: 'good', label: 'Good' },
  { value: 'fair', label: 'Fair' },
  { value: 'poor', label: 'Poor' },
]
