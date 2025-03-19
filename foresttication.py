import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# Set page configuration
st.set_page_config(
    page_title="Carbon Neutrality through Forestification",
    page_icon="🌳",
    layout="wide"
)

# App title and description
st.title("Carbon Neutrality Through Forestification")
st.markdown("""
This application helps estimate the carbon sequestration potential of forest projects
to offset carbon emissions. Input your project details and emission data to
calculate the forestification requirements for carbon neutrality.
""")

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Carbon Calculator", "ML Prediction", "About"])

# Generate sample data for demonstration
@st.cache_data
def generate_sample_data(n_samples=1000):
    np.random.seed(42)
    
    # Features
    tree_species = np.random.choice(['Pine', 'Oak', 'Maple', 'Eucalyptus', 'Birch'], n_samples)
    tree_age = np.random.randint(1, 100, n_samples)
    tree_density = np.random.randint(100, 2000, n_samples)  # trees per hectare
    annual_rainfall = np.random.randint(500, 2000, n_samples)  # mm
    avg_temperature = np.random.uniform(5, 30, n_samples)  # Celsius
    soil_quality = np.random.uniform(1, 10, n_samples)  # 1-10 scale
    forest_area = np.random.uniform(1, 1000, n_samples)  # hectares
    
    # Target: Carbon sequestration (tonnes CO2 per year)
    # This is a simplified model for demonstration
    base_sequestration = {
        'Pine': 7.5,
        'Oak': 10.0,
        'Maple': 8.0,
        'Eucalyptus': 15.0,
        'Birch': 6.5
    }
    
    carbon_sequestration = np.zeros(n_samples)
    for i in range(n_samples):
        # Calculate sequestration based on features
        age_factor = 1 - np.exp(-0.05 * tree_age[i])  # Trees sequester more as they grow but plateau
        density_factor = np.log(tree_density[i] / 100) / 5  # Diminishing returns on density
        climate_factor = (annual_rainfall[i] / 1000) * (avg_temperature[i] / 20)  # Climate impact
        soil_factor = soil_quality[i] / 5  # Soil quality impact
        
        # Calculate per hectare sequestration
        per_hectare = base_sequestration[tree_species[i]] * age_factor * density_factor * climate_factor * soil_factor
        
        # Total sequestration for the forest area
        carbon_sequestration[i] = per_hectare * forest_area[i]
        
        # Add some noise
        carbon_sequestration[i] *= np.random.normal(1, 0.1)
    
    # Create DataFrame
    df = pd.DataFrame({
        'tree_species': tree_species,
        'tree_age': tree_age,
        'tree_density': tree_density,
        'annual_rainfall': annual_rainfall,
        'avg_temperature': avg_temperature,
        'soil_quality': soil_quality,
        'forest_area': forest_area,
        'carbon_sequestration': carbon_sequestration
    })
    
    return df

# Load data
df = generate_sample_data()

# Create and train the model
@st.cache_resource
def train_model(df):
    # Prepare the data
    X = df.drop(['carbon_sequestration', 'tree_species'], axis=1)
    y = df['carbon_sequestration']
    
    # Get dummy variables for tree species
    species_dummies = pd.get_dummies(df['tree_species'], prefix='species')
    X = pd.concat([X, species_dummies], axis=1)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train the model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Evaluate the model
    y_pred = model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return model, scaler, mse, r2, X.columns

# Train the model
model, scaler, mse, r2, feature_names = train_model(df)

# Home page
if page == "Home":
    st.header("Welcome to the Carbon Neutrality Forestification Project")
    
    st.markdown("""
    ### How this application works:
    
    1. **Carbon Calculator**: Input your carbon emissions and get an estimate of how much forest area you need to offset them.
    
    2. **ML Prediction**: Use our machine learning model to estimate carbon sequestration potential based on various forest parameters.
    
    3. **About**: Learn more about the methodology and assumptions behind our calculations.
    
    ### Why Forestification?
    
    Forests are natural carbon sinks that absorb CO2 from the atmosphere through photosynthesis and store it in biomass and soil. 
    Strategic forestification can help neutralize carbon emissions while providing additional benefits like biodiversity conservation, 
    soil protection, and water cycle regulation.
    """)
    
    st.image("https://api.placeholder.com/800/300", caption="Forest Carbon Sequestration", use_container_width=True)

# Carbon Calculator page
elif page == "Carbon Calculator":
    st.header("Carbon Offset Calculator")
    
    st.markdown("""
    Input your annual carbon emissions and get an estimate of how much forest area you need to offset them.
    """)
    
    # Input for carbon emissions
    col1, col2 = st.columns(2)
    with col1:
        emission_type = st.selectbox("Emission Source", ["Industry", "Transportation", "Buildings", "Agriculture", "Other"])
        emission_amount = st.number_input("Annual Carbon Emissions (tonnes CO2)", min_value=0.0, value=1000.0, step=100.0)
    
    with col2:
        tree_species = st.selectbox("Preferred Tree Species", ["Pine", "Oak", "Maple", "Eucalyptus", "Birch", "Mixed"])
        forest_age = st.slider("Forest Age (years)", min_value=1, max_value=50, value=10)
    
    # Calculate forest area needed
    if st.button("Calculate Forest Area Needed"):
        # Simplified calculation based on average sequestration rates
        sequestration_rates = {
            "Pine": 7.5,
            "Oak": 10.0,
            "Maple": 8.0,
            "Eucalyptus": 15.0,
            "Birch": 6.5,
            "Mixed": 9.0
        }
        
        base_rate = sequestration_rates[tree_species]
        age_factor = 1 - np.exp(-0.05 * forest_age)
        annual_sequestration_per_hectare = base_rate * age_factor
        
        hectares_needed = emission_amount / annual_sequestration_per_hectare
        
        st.success(f"You need approximately {hectares_needed:.2f} hectares of {tree_species} forest to offset {emission_amount:.2f} tonnes of CO2 annually.")
        
        # Display comparisons
        st.subheader("Comparison")
        st.markdown(f"- {hectares_needed:.2f} hectares is equivalent to approximately {hectares_needed * 2.47:.2f} acres")
        st.markdown(f"- This is roughly the size of {hectares_needed / 100:.2f} soccer fields")
        
        # Show chart comparing different tree species
        st.subheader("Carbon Sequestration by Tree Species")
        species_list = list(sequestration_rates.keys())
        sequestration_list = [sequestration_rates[sp] * age_factor for sp in species_list]
        
        chart_data = pd.DataFrame({
            'Species': species_list,
            'Tonnes CO2 per hectare per year': sequestration_list
        })
        
        st.bar_chart(chart_data.set_index('Species'))

# ML Prediction page
elif page == "ML Prediction":
    st.header("Machine Learning Prediction")
    
    st.markdown("""
    Our machine learning model predicts carbon sequestration based on forest parameters.
    Input your forest details below to get an estimate of carbon sequestration potential.
    """)
    
    # Model information
    st.subheader("Model Performance")
    st.markdown(f"- Mean Squared Error (MSE): {mse:.2f}")
    st.markdown(f"- R-squared (R²): {r2:.2f}")
    
    # Input parameters
    st.subheader("Input Forest Parameters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        species = st.selectbox("Tree Species", ['Pine', 'Oak', 'Maple', 'Eucalyptus', 'Birch'])
        age = st.slider("Tree Age (years)", min_value=1, max_value=100, value=20)
        density = st.slider("Tree Density (trees per hectare)", min_value=100, max_value=2000, value=500)
        area = st.number_input("Forest Area (hectares)", min_value=1.0, max_value=1000.0, value=100.0)
    
    with col2:
        rainfall = st.slider("Annual Rainfall (mm)", min_value=500, max_value=2000, value=1000)
        temperature = st.slider("Average Temperature (°C)", min_value=5.0, max_value=30.0, value=15.0)
        soil = st.slider("Soil Quality (1-10)", min_value=1.0, max_value=10.0, value=5.0)
    
    # Predict button
    if st.button("Predict Carbon Sequestration"):
        # Prepare input data
        input_data = pd.DataFrame({
            'tree_age': [age],
            'tree_density': [density],
            'annual_rainfall': [rainfall],
            'avg_temperature': [temperature],
            'soil_quality': [soil],
            'forest_area': [area],
            'species_Birch': [1 if species == 'Birch' else 0],
            'species_Eucalyptus': [1 if species == 'Eucalyptus' else 0],
            'species_Maple': [1 if species == 'Maple' else 0],
            'species_Oak': [1 if species == 'Oak' else 0],
            'species_Pine': [1 if species == 'Pine' else 0]
        })
        
        # Make sure columns match the model's expected features
        input_data = input_data.reindex(columns=feature_names, fill_value=0)
        
        # Scale the input data
        input_scaled = scaler.transform(input_data)
        
        # Make prediction
        prediction = model.predict(input_scaled)[0]
        
        st.success(f"Predicted Carbon Sequestration: {prediction:.2f} tonnes CO2 per year")
        
        # Show feature importance
        st.subheader("Feature Importance")
        importances = model.feature_importances_
        feature_importance = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values(by='Importance', ascending=False)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=feature_importance[:10], ax=ax)
        plt.title('Top 10 Features by Importance')
        st.pyplot(fig)
        
        # Time to offset emissions
        st.subheader("Emission Offset Calculator")
        emissions = st.number_input("Annual Carbon Emissions to Offset (tonnes CO2)", min_value=0.0, value=1000.0, step=100.0)
        
        years_to_offset = emissions / prediction
        st.markdown(f"It would take approximately {years_to_offset:.2f} years for this forest to offset {emissions:.2f} tonnes of CO2.")

# About page
elif page == "About":
    st.header("About This Project")
    
    st.markdown("""
    ### Methodology
    
    This application uses a machine learning model (Random Forest Regressor) to predict carbon sequestration potential
    based on forest parameters. The model is trained on a dataset that includes various factors that affect carbon
    sequestration, such as tree species, age, density, climate conditions, and soil quality.
    
    ### Assumptions
    
    - The carbon sequestration rates are based on scientific literature and may vary in real-world conditions.
    - The model assumes a linear relationship between forest area and carbon sequestration.
    - Climate factors (rainfall, temperature) are simplified and do not account for seasonal variations.
    - The model does not account for carbon release during forest establishment or management.
    
    ### Limitations
    
    - The predictions are estimates and should be used for planning purposes only.
    - Local conditions may significantly affect actual carbon sequestration rates.
    - The model does not account for potential forest disturbances (fire, pests, etc.).
    
    ### References
    
    - IPCC Guidelines for National Greenhouse Gas Inventories
    - FAO Forestry Paper: Managing forests for climate change
    - Various scientific papers on carbon sequestration in different forest types
    
    ### Future Improvements
    
    - Integration with actual field data for model calibration
    - Incorporation of more detailed climate data
    - Addition of economic analysis for forestification projects
    - Support for mixed-species forests and agroforestry systems
    """)

# Add a footer
st.markdown("---")
st.markdown("© 2025 Carbon Neutrality Forestification Project")
