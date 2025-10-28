// ======================
// Life Expectancy Predictor - JavaScript
// ======================

// Note: Dark mode removed; UI remains in light mode only.

// DOM Elements
const predictionForm = document.getElementById('predictionForm');
const resultsContainer = document.getElementById('resultsContainer');
const submitButton = document.querySelector('.submit-button');

// Form input elements
const heightInput = document.getElementById('height');
const heightRange = document.getElementById('heightRange');
const weightInput = document.getElementById('weight');
const weightRange = document.getElementById('weightRange');
const cholesterolInput = document.getElementById('cholesterol');
const cholesterolRange = document.getElementById('cholesterolRange');
const bmiInput = document.getElementById('bmi');

// Result display elements
const predictionValue = document.getElementById('predictionValue');
const profileBmi = document.getElementById('profileBmi');
const profileCholesterol = document.getElementById('profileCholesterol');
const profileActivity = document.getElementById('profileActivity');
const profileSmoking = document.getElementById('profileSmoking');
const insightsContainer = document.getElementById('insightsContainer');
const recommendationsContainer = document.getElementById('recommendationsContainer');

// ======================
// Event Listeners
// ======================

// Sync height inputs
heightInput.addEventListener('input', (e) => {
    heightRange.value = e.target.value;
    calculateBMI();
});

heightRange.addEventListener('input', (e) => {
    heightInput.value = e.target.value;
    calculateBMI();
});

// Sync weight inputs
weightInput.addEventListener('input', (e) => {
    weightRange.value = e.target.value;
    calculateBMI();
});

weightRange.addEventListener('input', (e) => {
    weightInput.value = e.target.value;
    calculateBMI();
});

// Sync cholesterol inputs
cholesterolInput.addEventListener('input', (e) => {
    cholesterolRange.value = e.target.value;
});

cholesterolRange.addEventListener('input', (e) => {
    cholesterolInput.value = e.target.value;
});

// Form submission
predictionForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    await makePrediction();
});

// ======================
// Helper Functions
// ======================

function calculateBMI() {
    const height = parseFloat(heightInput.value);
    const weight = parseFloat(weightInput.value);

    if (height > 0 && weight > 0) {
        const bmi = weight / (height / 100) ** 2;
        bmiInput.value = bmi.toFixed(1);
    }
}

function getFormData() {
    return {
        gender: document.getElementById('gender').value,
        height: parseFloat(heightInput.value),
        weight: parseFloat(weightInput.value),
        bmi: parseFloat(bmiInput.value),
        physical_activity: document.getElementById('physical_activity').value,
        smoking_status: document.getElementById('smoking_status').value,
        alcohol_consumption: document.getElementById('alcohol_consumption').value,
        diet: document.getElementById('diet').value,
        blood_pressure: document.getElementById('blood_pressure').value,
        cholesterol: parseFloat(cholesterolInput.value),
        diabetes: document.getElementById('diabetes').checked ? 1 : 0,
        hypertension: document.getElementById('hypertension').checked ? 1 : 0,
        heart_disease: document.getElementById('heart_disease').checked ? 1 : 0,
        asthma: document.getElementById('asthma').checked ? 1 : 0
    };
}

function validateForm(data) {
    const requiredFields = [
        'gender', 'physical_activity', 'smoking_status', 
        'alcohol_consumption', 'diet', 'blood_pressure'
    ];

    for (let field of requiredFields) {
        if (!data[field] || data[field] === '') {
            showError(`Please fill in the ${field.replace('_', ' ')} field`);
            return false;
        }
    }
    return true;
}

function showError(message) {
    alert(message);
}

function showLoading(loading) {
    if (loading) {
        submitButton.classList.add('loading');
        submitButton.disabled = true;
    } else {
        submitButton.classList.remove('loading');
        submitButton.disabled = false;
    }
}

// ======================
// Prediction Function
// ======================

async function makePrediction() {
    // Get form data
    const formData = getFormData();

    // Validate form
    if (!validateForm(formData)) {
        return;
    }

    // Show loading state
    showLoading(true);

    try {
        // Make API request
        const response = await fetch('/api/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(formData)
        });

        if (!response.ok) {
            throw new Error('Prediction failed');
        }

        const result = await response.json();

        if (result.success) {
            displayResults(result);
        } else {
            showError(result.error || 'An error occurred');
        }
    } catch (error) {
        showError('Error making prediction: ' + error.message);
    } finally {
        showLoading(false);
    }
}

// ======================
// Display Results
// ======================

function displayResults(result) {
    // Update prediction value
    predictionValue.textContent = result.prediction;
    predictionValue.style.animation = 'none';
    setTimeout(() => {
        predictionValue.style.animation = 'fadeUp 0.6s ease-out';
    }, 10);

    // Update profile summary
    profileBmi.textContent = result.profile.bmi;
    profileCholesterol.textContent = result.profile.cholesterol;
    profileActivity.textContent = result.profile.activity;
    profileSmoking.textContent = result.profile.smoking;

    // Display insights
    displayInsights(result.insights);

    // Display recommendations
    displayRecommendations(result.recommendations);

    // Show results container
    resultsContainer.classList.remove('hidden');

    // Scroll to results
    setTimeout(() => {
        resultsContainer.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }, 100);
}

function displayInsights(insights) {
    insightsContainer.innerHTML = '';

    insights.forEach((insight, index) => {
        const item = document.createElement('div');
        item.className = `insight-item ${insight.type}`;
        item.textContent = insight.text;
        item.style.animationDelay = `${index * 0.1}s`;
        insightsContainer.appendChild(item);
    });
}

function displayRecommendations(recommendations) {
    recommendationsContainer.innerHTML = '';

    recommendations.forEach((recommendation, index) => {
        const item = document.createElement('div');
        item.className = 'recommendation-item success';
        item.textContent = recommendation;
        item.style.animationDelay = `${index * 0.1}s`;
        recommendationsContainer.appendChild(item);
    });
}

// ======================
// Download Report
// ======================

async function downloadReport() {
    const prediction = predictionValue.textContent;
    const formData = getFormData();

    // Pull insights and recommendations from DOM
    const insights = Array.from(document.querySelectorAll('.insight-item')).map(i => i.textContent);
    const recommendations = Array.from(document.querySelectorAll('.recommendation-item')).map(i => i.textContent);

    const payload = {
        formData,
        prediction,
        insights,
        recommendations
    };

    try {
        showLoading(true);
        const resp = await fetch('/api/report', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        if (!resp.ok) {
            const err = await resp.json().catch(() => ({}));
            throw new Error(err.error || 'PDF generation failed');
        }

        const blob = await resp.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `life-expectancy-report-${Date.now()}.pdf`;
        document.body.appendChild(a);
        a.click();
        a.remove();
        window.URL.revokeObjectURL(url);
    } catch (err) {
        showError('Error generating PDF: ' + err.message);
    } finally {
        showLoading(false);
    }
}

function getBMIStatus(bmi) {
    if (bmi < 18.5) return 'Underweight';
    if (bmi < 25) return 'Healthy Weight';
    if (bmi < 30) return 'Overweight';
    return 'Obese';
}

// ======================
// Initialize
// ======================

document.addEventListener('DOMContentLoaded', () => {
    // Calculate initial BMI
    calculateBMI();

    // Add smooth scroll behavior
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth'
                });
            }
        });
    });

    console.log('✓ Life Expectancy Predictor initialized');
});
