"""
Definitive Sentinel-2 Burn Scar Detection with Triple-Check (dNBR + BAI + dNDVI)
Optimized for maximum accuracy, with user-drawable ROI and standard USGS thresholds.
Methodology: "Clip and Conquer" for robust, research-standard analysis.
"""

import ee
import uvicorn
import csv
import io
from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from datetime import datetime
from typing import Optional, Dict, Any

# --- CONFIGURATION ---
LATEST_VALID_LANDCOVER_YEAR = 2023 

# Initialize Google Earth Enginem
try:
    ee.Initialize(project='ee-gopalt')
    print("✅ Google Earth Engine initialized with project 'ee-gopalt'")
except Exception as e:
    print(f"❌ GEE Error: {e}")

app = FastAPI(title="Triple-Check S2 Fire Detection", description="Burn Scar Mapping with dNBR, BAI, & dNDVI")

class FireDetectionRequest(BaseModel):
    start_date: str
    end_date: str
    state: Optional[str] = None
    district: Optional[str] = None
    roi: Optional[Dict[str, Any]] = None

STATES_DISTRICTS = {
    "Punjab": ["Amritsar", "Bathinda", "Ferozepur", "Gurdaspur", "Hoshiarpur", "Jalandhar", "Ludhiana", "Patiala", "Sangrur"],
    "Haryana": ["Ambala", "Hisar", "Karnal", "Kurukshetra", "Panipat", "Rohtak", "Sirsa", "Sonipat", "Yamunanagar"],
    "Uttar Pradesh": ["Agra", "Aligarh", "Bareilly", "Ghaziabad", "Kanpur", "Lucknow", "Meerut", "Varanasi"],
    "Delhi": ["Central Delhi", "East Delhi", "New Delhi", "North Delhi", "South Delhi", "West Delhi"]
}

current_fire_data = {'hotspots': [], 'state': '', 'district': '', 'date_range': ''}

def get_district_boundary(state: str, district: str):
    """Get district boundary from FAO/GAUL dataset."""
    try:
        print(f"📍 Getting boundary for {district}, {state}")
        gaul = ee.FeatureCollection("FAO/GAUL/2015/level2")
        district_boundary = gaul.filter(ee.Filter.And(
            ee.Filter.eq('ADM1_NAME', state),
            ee.Filter.eq('ADM2_NAME', district)
        ))

        if district_boundary.size().getInfo() == 0:
            return {'status': 'not_found'}

        return {
            'boundary_geojson': district_boundary.first().geometry().getInfo(),
            'geometry': district_boundary.geometry(),
            'status': 'success'
        }
    except Exception as e:
        print(f"❌ Boundary error: {e}")
        return {'status': 'error', 'error': str(e)}

def get_agricultural_mask(year: int):
    """Creates a global mask for agricultural areas for a specific year."""
    try:
        landcover = ee.ImageCollection("MODIS/061/MCD12Q1") \
                     .filter(ee.Filter.calendarRange(year, year, 'year')) \
                     .first()
        if landcover is None:
            return None
        agri_mask = landcover.select('LC_Type1').eq(12).Or(landcover.select('LC_Type1').eq(14))
        return agri_mask.selfMask()
    except Exception as e:
        print(f"⚠ Agricultural mask creation failed for year {year}: {e}")
        return None

def mask_s2_clouds(image):
    """Masks clouds in a Sentinel-2 image."""
    qa = image.select('QA60')
    cloud_bit_mask = 1 << 10
    cirrus_bit_mask = 1 << 11
    mask = qa.bitwiseAnd(cloud_bit_mask).eq(0).And(qa.bitwiseAnd(cirrus_bit_mask).eq(0))
    return image.updateMask(mask).divide(10000).copyProperties(image, ["system:time_start"])

def calculate_bai(image):
    """Calculates the Burned Area Index (BAI)."""
    red = image.select('B4')
    nir = image.select('B8')
    return image.expression('1.0 / ((0.1 - RED)**2 + (0.06 - NIR)**2)', {'RED': red, 'NIR': nir}).rename('BAI')

def get_burn_severity(dnbr_value):
    """Classifies dNBR value into severity level based on USGS standards."""
    if dnbr_value is None:
        return 'Unknown'
    if dnbr_value >= 0.66:
        return 'Very High Severity'
    elif dnbr_value >= 0.44:
        return 'High Severity'
    elif dnbr_value >= 0.27:
        return 'Moderate Severity'
    elif dnbr_value >= 0.10:
        return 'Low Severity'
    else:
        return 'Unburned or Regrowth'


def extract_burn_scars_s2(roi_geometry, start_date_str, end_date_str):
    """
    Extracts burn scars using the definitive "Triple-Check" (dNBR + BAI + dNDVI) methodology.
    """
    try:
        print("🔥 Extracting burn scars with definitive dNBR + BAI + dNDVI 'Triple-Check' method...")

        post_fire_start = ee.Date(start_date_str)
        post_fire_end = ee.Date(end_date_str)
        pre_fire_start = post_fire_start.advance(-60, 'day')
        pre_fire_end = post_fire_start.advance(-15, 'day')

        print(f"🗓 Pre-fire: {pre_fire_start.format('YYYY-MM-dd').getInfo()} to {pre_fire_end.format('YYYY-MM-dd').getInfo()}")
        print(f"🗓 Post-fire: {post_fire_start.format('YYYY-MM-dd').getInfo()} to {post_fire_end.format('YYYY-MM-dd').getInfo()}")

        s2_collection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
            .filterBounds(roi_geometry) \
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 40))

        pre_fire_collection = s2_collection.filterDate(pre_fire_start, pre_fire_end)
        post_fire_collection = s2_collection.filterDate(post_fire_start, post_fire_end)

        pre_count = pre_fire_collection.size().getInfo()
        post_count = post_fire_collection.size().getInfo()
        print(f"🖼 Found {pre_count} pre-fire and {post_count} post-fire images.")

        if pre_count == 0 or post_count == 0:
            print("⚠ Not enough cloud-free images. Cannot perform analysis.")
            return []

        pre_fire_img = pre_fire_collection.map(mask_s2_clouds).median()
        post_fire_img = post_fire_collection.map(mask_s2_clouds).median()

        dnbr = pre_fire_img.normalizedDifference(['B8', 'B12']).subtract(
               post_fire_img.normalizedDifference(['B8', 'B12'])).rename('dNBR')
        dndvi = pre_fire_img.normalizedDifference(['B8', 'B4']).subtract(
                post_fire_img.normalizedDifference(['B8', 'B4'])).rename('dNDVI')
        bai = calculate_bai(post_fire_img)

        dnbr_threshold = 0.10
        bai_threshold = 89
        dndvi_threshold = 0.2
        print(f"⚙️ Applying Triple-Check thresholds: dNBR>{dnbr_threshold}, BAI>{bai_threshold}, dNDVI>{dndvi_threshold}")

        burn_mask = dnbr.gt(dnbr_threshold).And(bai.gt(bai_threshold)).And(dndvi.gt(dndvi_threshold))

        analysis_year = post_fire_start.get('year').getInfo()
        agri_mask_year = min(analysis_year, LATEST_VALID_LANDCOVER_YEAR)
        print(f"🌾 Requesting agricultural mask for year {agri_mask_year} (latest available).")
        agri_mask = get_agricultural_mask(agri_mask_year)
        
        if agri_mask is None:
            print("⚠ Could not get agricultural mask. Proceeding without it (results may include non-agricultural burns).")
            final_mask = burn_mask
        else:
            print("🌾 Creating final agricultural burn scar mask...")
            final_mask = burn_mask.And(agri_mask)

        analysis_image = dnbr.addBands(bai).addBands(dndvi).updateMask(final_mask)

        print("📍 Sampling directly from the final analysis image...")
        hotspots_fc = analysis_image.sample(
            region=roi_geometry, scale=20, numPixels=20000, geometries=True, dropNulls=True, tileScale=4
        )

        hotspots_info = hotspots_fc.getInfo()
        print(f"🎯 Detected and sampled {len(hotspots_info['features'])} agricultural burn scar pixels.")

        all_hotspots = []
        for i, feature in enumerate(hotspots_info['features']):
            props = feature['properties']
            dnbr_value = props.get('dNBR', 0)
            severity_level = get_burn_severity(dnbr_value)

            all_hotspots.append({
                'id': i + 1,
                'latitude': feature['geometry']['coordinates'][1],
                'longitude': feature['geometry']['coordinates'][0],
                'dnbr': round(dnbr_value, 3),
                'bai': round(props.get("BAI", 0), 2),
                'dndvi': round(props.get("dNDVI", 0), 3),
                'severity': severity_level,
            })
        return all_hotspots

    except ee.EEException as e:
        print(f"❌ Google Earth Engine error during scar detection: {e}")
        if "Parameter 'input' is required and may not be null" in str(e):
            print("💡 Hint: This error often occurs when an image collection (like land cover) is empty for the selected year.")
        return []
    except Exception as e:
        print(f"❌ An unexpected error occurred during scar detection: {e}")
        return []

def run_fire_detection(request: FireDetectionRequest):
    """Orchestrates fire detection based on user input (ROI or district)."""
    global current_fire_data
    try:
        roi_geometry = None
        boundary_geojson = None
        state_name, district_name = "", ""

        if request.roi:
            print("Processing Custom ROI...")
            roi_geometry = ee.Geometry(request.roi['geometry'])
            boundary_geojson = request.roi['geometry']
            state_name, district_name = "Custom", "ROI"
        elif request.state and request.district:
            print(f"Processing District: {request.district}, {request.state}")
            boundary_info = get_district_boundary(request.state, request.district)
            if boundary_info['status'] != 'success':
                raise HTTPException(status_code=404, detail="District not found")
            roi_geometry = boundary_info['geometry']
            boundary_geojson = boundary_info['boundary_geojson']
            state_name, district_name = request.state, request.district
        else:
            raise HTTPException(status_code=400, detail="Either a state/district or a custom ROI must be provided.")

        hotspots = extract_burn_scars_s2(roi_geometry, request.start_date, request.end_date)
        
        max_dnbr = max([h['dnbr'] for h in hotspots], default=0.0)
        current_fire_data = {'hotspots': hotspots, 'state': state_name, 'district': district_name, 'date_range': f"{request.start_date} to {request.end_date}"}
        
        hotspots_geojson = {'type': 'FeatureCollection', 'features': []}
        for spot in hotspots:
            hotspots_geojson['features'].append({
                'type': 'Feature', 
                'geometry': {'type': 'Point', 'coordinates': [spot['longitude'], spot['latitude']]}, 
                'properties': spot
            })
        
        return {
            'fire_hotspots': len(hotspots),
            'fire_area_hectares': round(len(hotspots) * 0.04, 2),
            'max_dnbr': max_dnbr,
            'hotspots_geojson': hotspots_geojson,
            'boundary_geojson': boundary_geojson,
            'status': 'Success'
        }
    except Exception as e:
        print(f"❌ Top-level detection error: {e}")
        return {'status': f'Error: {str(e)}', 'fire_hotspots': 0, 'hotspots_geojson': {'type': 'FeatureCollection', 'features': []}}

# --- API Endpoints and Frontend ---
@app.get("/", response_class=HTMLResponse)
async def dashboard():
    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Triple-Check S2 Burn Scar Detection</title>
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/leaflet.draw/1.0.4/leaflet.draw.css" />
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif; display: flex; height: 100vh; margin: 0; }
        .sidebar { width: 380px; background: #2c3e50; color: white; padding: 20px; overflow-y: auto; display: flex; flex-direction: column; }
        .main-content { flex: 1; display: flex; flex-direction: column; }
        #map { flex: 1; background: #f0f0f0; }
        .section { margin-bottom: 20px; padding: 15px; background: rgba(255,255,255,0.05); border-radius: 8px; }
        .section h3 { color: #e67e22; margin-bottom: 10px; font-size: 14px; text-transform: uppercase; letter-spacing: 1px; }
        .form-control { width: 100%; padding: 10px; margin: 5px 0; border-radius: 4px; border: 1px solid #4a627a; background: #34495e; color: white; box-sizing: border-box; }
        .btn { width: 100%; padding: 12px; border: none; border-radius: 4px; cursor: pointer; font-weight: bold; text-transform: uppercase; letter-spacing: 0.5px; }
        .btn-success { background: #27ae60; color: white; }
        .btn-warning { background: #f39c12; color: white; }
        .btn:disabled { background: #7f8c8d; color: #bdc3c7; cursor: not-allowed; }
        .result-item { display: flex; justify-content: space-between; padding: 8px 0; font-size: 14px; border-bottom: 1px solid #34495e; }
        .result-item:last-child { border-bottom: none; }
        .result-item span:last-child { font-weight: bold; color: #2ecc71; }
        .hidden { display: none; }
        .status-bar { background: #1d2b3a; color: #bdc3c7; padding: 10px; font-size: 12px; text-align: center; }
        .loader { border: 4px solid #f3f3f3; border-top: 4px solid #3498db; border-radius: 50%; width: 20px; height: 20px; animation: spin 1s linear infinite; margin: 10px auto; }
        .legend { padding: 6px 8px; font-size: 12px; line-height: 18px; background: rgba(255,255,255,0.85); box-shadow: 0 0 15px rgba(0,0,0,0.2); border-radius: 5px; }
        .legend i { width: 18px; height: 18px; float: left; margin-right: 8px; opacity: 0.9; }
        .legend h4 { margin: 0 0 5px; color: #333; }
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
    </style>
</head>
<body>
    <div class="sidebar">
        <h2>🛰️ S2 Burn Scar Detection</h2>
        <p style="font-size: 12px; color: #bdc3c7; margin-bottom: 20px;">USGS-Standard dNBR + BAI + dNDVI Method</p>
        
        <div class="section">
            <h3>📍 Analysis Area</h3>
            <select id="modeSelect" class="form-control">
                <option value="select">Select from List</option>
                <option value="draw">Draw on Map</option>
            </select>
            <div id="locationSelector">
                <select id="stateSelect" class="form-control"><option value="">Select State</option></select>
                <select id="districtSelect" class="form-control" disabled><option value="">Select District</option></select>
            </div>
            <div id="drawInstructions" class="hidden" style="font-size: 12px; color: #bdc3c7; margin-top: 10px; padding: 8px; background: rgba(0,0,0,0.2); border-radius: 4px;">
                Use the tools on the map to draw a rectangle or polygon.
            </div>
        </div>

        <div class="section">
            <h3>📅 Post-Fire Date Range</h3>
            <input type="date" id="startDate" class="form-control">
            <input type="date" id="endDate" class="form-control">
        </div>

        <div class="section">
            <h3>🔍 Run Analysis</h3>
            <button id="analyzeButton" onclick="analyzeFires()" class="btn btn-success" disabled>Detect Burn Scars</button>
            <div id="loadingIndicator" class="hidden"><div class="loader"></div><p style="text-align:center; font-size:12px;">Analyzing...</p></div>
        </div>

        <div class="section">
            <h3>🎯 Results</h3>
            <div id="resultsContainer">
                <div class="result-item"><span>Burned Pixels:</span> <span id="firePixels">0</span></div>
                <div class="result-item"><span>Burned Area (ha):</span> <span id="fireArea">0</span></div>
                <div class="result-item"><span>Max dNBR:</span> <span id="maxDNBR">0</span></div>
            </div>
        </div>

        <div class="section">
            <h3>📁 Export</h3>
            <button id="exportButton" onclick="exportCSV()" class="btn btn-warning" disabled>Export to CSV</button>
        </div>
    </div>

    <div class="main-content">
        <div id="map"></div>
        <div class="status-bar" id="statusBar">Select an analysis area and date range.</div>
    </div>

    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/leaflet.draw/1.0.4/leaflet.draw.js"></script>
    <script>
        const map = L.map('map').setView([28.6139, 77.2090], 6);
        L.tileLayer('https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}{r}.png', {
            attribution: '© <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors © <a href="https://carto.com/attributions">CARTO</a>'
        }).addTo(map);

        let boundaryLayer, hotspotsLayer, legend, drawControl;
        let drawnROI = null;
        
        const modeSelect = document.getElementById('modeSelect');
        const locationSelector = document.getElementById('locationSelector');
        const drawInstructions = document.getElementById('drawInstructions');
        const stateSelect = document.getElementById('stateSelect');
        const districtSelect = document.getElementById('districtSelect');
        const analyzeButton = document.getElementById('analyzeButton');
        const exportButton = document.getElementById('exportButton');
        const loadingIndicator = document.getElementById('loadingIndicator');
        const statusBar = document.getElementById('statusBar');

        function setupDrawControls() {
            const drawnItems = new L.FeatureGroup();
            map.addLayer(drawnItems);
            drawControl = new L.Control.Draw({
                edit: { featureGroup: drawnItems, remove: true },
                draw: {
                    polygon: { shapeOptions: { color: '#e67e22' } },
                    rectangle: { shapeOptions: { color: '#e67e22' } },
                    polyline: false, circle: false, marker: false, circlemarker: false
                }
            });
            
            map.on(L.Draw.Event.CREATED, function (e) {
                if (boundaryLayer) map.removeLayer(boundaryLayer);
                drawnItems.clearLayers();
                const layer = e.layer;
                drawnItems.addLayer(layer);
                drawnROI = layer.toGeoJSON();
                boundaryLayer = L.geoJSON(drawnROI, { style: { color: '#e67e22', weight: 3, opacity: 1, fillOpacity: 0.1 } }).addTo(map);
                analyzeButton.disabled = false;
                statusBar.textContent = 'Custom ROI defined. Ready to analyze.';
            });

            map.on('draw:edited', function (e) {
                const layer = e.layers.getLayers()[0];
                drawnROI = layer.toGeoJSON();
                statusBar.textContent = 'Custom ROI updated. Ready to analyze.';
            });

             map.on('draw:deleted', function () {
                drawnROI = null;
                analyzeButton.disabled = true;
                statusBar.textContent = 'ROI deleted. Please draw a new one.';
            });
        }

        modeSelect.addEventListener('change', (e) => {
            const mode = e.target.value;
            if (boundaryLayer) map.removeLayer(boundaryLayer);
            drawnROI = null;
            analyzeButton.disabled = true;

            if (mode === 'draw') {
                locationSelector.classList.add('hidden');
                drawInstructions.classList.remove('hidden');
                stateSelect.value = '';
                districtSelect.value = '';
                districtSelect.disabled = true;
                if (!drawControl) setupDrawControls();
                map.addControl(drawControl);
                statusBar.textContent = 'Draw a polygon or rectangle on the map.';
            } else { // select mode
                locationSelector.classList.remove('hidden');
                drawInstructions.classList.add('hidden');
                if (drawControl) map.removeControl(drawControl);
                statusBar.textContent = 'Select a state and district.';
            }
        });

        async function loadStates() {
            try {
                const response = await fetch('/api/states');
                if (!response.ok) throw new Error('Failed to load states');
                const data = await response.json();
                data.states.forEach(state => stateSelect.add(new Option(state, state)));
            } catch (error) {
                statusBar.textContent = `Error: ${error.message}`;
            }
        }

        stateSelect.addEventListener('change', async (e) => {
            districtSelect.innerHTML = '<option value="">Select District</option>';
            districtSelect.disabled = true;
            analyzeButton.disabled = true;
            const state = e.target.value;
            if (!state) return;
            try {
                const response = await fetch(`/api/districts/${encodeURIComponent(state)}`);
                if (!response.ok) throw new Error('Failed to load districts');
                const data = await response.json();
                data.districts.forEach(d => districtSelect.add(new Option(d, d)));
                districtSelect.disabled = false;
            } catch (error) {
                statusBar.textContent = `Error: ${error.message}`;
            }
        });

        districtSelect.addEventListener('change', (e) => {
            analyzeButton.disabled = !e.target.value;
        });

        async function analyzeFires() {
            loadingIndicator.classList.remove('hidden');
            analyzeButton.disabled = true;
            exportButton.disabled = true;
            statusBar.textContent = 'Requesting data from Google Earth Engine... This may take a minute.';

            const payload = {
                start_date: document.getElementById('startDate').value,
                end_date: document.getElementById('endDate').value,
                state: stateSelect.value,
                district: districtSelect.value,
                roi: drawnROI
            };

            try {
                const response = await fetch('/api/detect-fires', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });

                if (!response.ok) throw new Error(`Server error: ${response.statusText}`);
                const data = await response.json();
                if (data.status && data.status.includes('Error')) throw new Error(data.status);

                statusBar.textContent = `Analysis complete. Found ${data.fire_hotspots} agricultural burn scar pixels.`;
                updateUI(data);

            } catch (error) {
                console.error('Analysis failed:', error);
                statusBar.textContent = `Error: ${error.message}`;
                alert(`Analysis failed: ${error.message}`);
            } finally {
                loadingIndicator.classList.add('hidden');
                analyzeButton.disabled = false;
            }
        }

        function getSeverityColor(severity) {
            if (severity === 'Very High Severity') return '#d73027';
            else if (severity === 'High Severity') return '#fc8d59';
            else if (severity === 'Moderate Severity') return '#fee08b';
            else if (severity === 'Low Severity') return '#ffffbf';
            return '#bababa';
        }
        
        function updateLegend() {
            if (legend) map.removeControl(legend);
            legend = L.control({position: 'bottomright'});
            legend.onAdd = function (map) {
                var div = L.DomUtil.create('div', 'info legend'),
                    grades = ["Very High Severity", "High Severity", "Moderate Severity", "Low Severity"],
                    labels = [];
                div.innerHTML += '<h4>Burn Severity</h4>'
                for (var i = 0; i < grades.length; i++) {
                    div.innerHTML += '<i style="background:' + getSeverityColor(grades[i]) + '"></i> ' + grades[i] + '<br>';
                }
                return div;
            };
            legend.addTo(map);
        }

        function updateUI(data) {
            document.getElementById('firePixels').textContent = data.fire_hotspots.toLocaleString();
            document.getElementById('fireArea').textContent = data.fire_area_hectares.toLocaleString();
            document.getElementById('maxDNBR').textContent = data.max_dnbr;

            if (boundaryLayer && modeSelect.value === 'select') map.removeLayer(boundaryLayer);
            if (data.boundary_geojson) {
                 if (modeSelect.value === 'select') { 
                    boundaryLayer = L.geoJSON(data.boundary_geojson, { style: { color: '#2ecc71', weight: 2, opacity: 1, fillOpacity: 0.1 } }).addTo(map);
                    map.fitBounds(boundaryLayer.getBounds(), {padding: [20, 20]});
                 }
            }

            if (hotspotsLayer) map.removeLayer(hotspotsLayer);
            if (data.hotspots_geojson && data.hotspots_geojson.features.length > 0) {
                hotspotsLayer = L.geoJSON(data.hotspots_geojson, {
                    pointToLayer: (feature, latlng) => {
                        const color = getSeverityColor(feature.properties.severity);
                        return L.circleMarker(latlng, { radius: 4, fillColor: color, color: '#333', weight: 0.5, opacity: 1, fillOpacity: 0.8 });
                    },
                    onEachFeature: (feature, layer) => {
                        const p = feature.properties;
                        layer.bindPopup(`<b>Burn Scar #${p.id}</b><br>Severity: ${p.severity}<br>dNBR: ${p.dnbr}<br>BAI: ${p.bai}<br>dNDVI: ${p.dndvi}`);
                    }
                }).addTo(map);
                exportButton.disabled = false;
                updateLegend();
            } else {
                exportButton.disabled = true;
                if (legend) map.removeControl(legend);
                alert('No agricultural burn scars were detected for the selected criteria. This could be due to cloud cover or no fire events.');
            }
        }
        
        function setDateDefaults() {
            const today = new Date();
            const todayString = today.toISOString().split('T')[0]; // YYYY-MM-DD format

            const endDateInput = document.getElementById('endDate');
            endDateInput.max = todayString;
            endDateInput.value = todayString;

            const startDateInput = document.getElementById('startDate');
            startDateInput.max = todayString;
            
            const twoMonthsAgo = new Date();
            twoMonthsAgo.setMonth(today.getMonth() - 2);
            // Handle year change for Jan/Feb
            if (today.getMonth() < 2) {
                twoMonthsAgo.setFullYear(today.getFullYear() - 1);
            }
            startDateInput.value = twoMonthsAgo.toISOString().split('T')[0];
        }

        async function exportCSV() {
            try {
                const response = await fetch('/api/export-csv');
                if (!response.ok) throw new Error('Export failed');
                const blob = await response.blob();
                const url = window.URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.style.display = 'none';
                a.href = url;
                a.download = `S2_BurnScars_${current_fire_data.state}_${current_fire_data.district}.csv`;
                document.body.appendChild(a);
                a.click();
                window.URL.revokeObjectURL(url);
                document.body.removeChild(a);
            } catch (error) {
                alert(error.message);
            }
        }
        
        // --- INITIALIZE APP ---
        setDateDefaults();
        loadStates();
        modeSelect.dispatchEvent(new Event('change'));

    </script>
</body>
</html>
    """

@app.get("/api/states")
async def get_states_api(): return {"states": list(STATES_DISTRICTS.keys())}

@app.get("/api/districts/{state}")
async def get_districts_api(state: str):
    if state not in STATES_DISTRICTS: raise HTTPException(status_code=404)
    return {"districts": STATES_DISTRICTS[state]}

@app.post("/api/detect-fires")
async def detect_fire_api(request: FireDetectionRequest):
    return run_fire_detection(request)

@app.get("/api/export-csv", response_class=Response)
async def export_csv_api():
    if not current_fire_data['hotspots']: raise HTTPException(status_code=404, detail="No data available to export")
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(['ID', 'Latitude', 'Longitude', 'Severity', 'dNBR', 'BAI', 'dNDVI'])
    for spot in current_fire_data['hotspots']:
        writer.writerow([spot.get(k) for k in ['id', 'latitude', 'longitude', 'severity', 'dnbr', 'bai', 'dndvi']])
    csv_content = output.getvalue()
    output.close()
    filename = f"S2_BurnScars_{current_fire_data['state']}_{current_fire_data['district']}_{current_fire_data['date_range']}.csv"
    return Response(content=csv_content, media_type="text/csv", headers={"Content-Disposition": f"attachment; filename={filename}"})

if __name__ == "__main__":
    print("=" * 60)
    print("TRIPLE-CHECK SENTINEL-2 BURN SCAR DETECTION (dNBR+BAI+dNDVI)")
    print("=" * 60)
    uvicorn.run(app, host="localhost", port=8080)