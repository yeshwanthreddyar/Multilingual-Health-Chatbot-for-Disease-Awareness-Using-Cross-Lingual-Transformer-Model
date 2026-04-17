# Location-Based Hospital Finder Feature

## Overview

The HealthBot now supports finding nearby hospitals and healthcare facilities based on user location or location queries.

## How It Works

### 1. Location Detection

The system detects location queries when users ask:
- "find nearby hospital"
- "where is the nearest hospital"
- "hospitals in [city name]"
- "I need to go to the doctor, can you give me location of nearby hospital"

### 2. Location Sources

The system accepts location in multiple formats:
- **City/Area name**: "Bangalore", "Delhi", "Mumbai"
- **Coordinates**: "12.9716,77.5946" (latitude, longitude)
- **Extracted from message**: Automatically extracts city names from queries

### 3. Hospital Search

Uses **OpenStreetMap Nominatim API** (free, no API key required) to:
- Geocode location names to coordinates
- Search for hospitals within a radius
- Return hospital names, addresses, and locations

**Optional**: Can use Google Places API if `GOOGLE_MAPS_API_KEY` environment variable is set.

## Usage Examples

### Via API

```python
import requests

# With city name
response = requests.post("http://localhost:8000/api/chat", json={
    "message": "I need to go to the doctor can you give me location of nearby hospital",
    "location": "Bangalore"  # Optional: specify location
})

# With coordinates
response = requests.post("http://localhost:8000/api/chat", json={
    "message": "find hospitals near me",
    "location": "12.9716,77.5946"  # lat,lon format
})

# Auto-detect from message
response = requests.post("http://localhost:8000/api/chat", json={
    "message": "hospitals in Mumbai"
})
```

### Via Web Interface

Simply type:
- "find nearby hospital"
- "hospitals in Bangalore"
- "where can I find a doctor near me"

### Via WhatsApp/SMS

Send messages like:
- "find hospital near me"
- "hospitals in Delhi"
- "I need doctor location"

## Configuration

### Environment Variables

```bash
# Optional: Google Maps API key for enhanced results
GOOGLE_MAPS_API_KEY=your_api_key_here
```

### Default Settings

- **Search radius**: 10 km
- **Max results**: 5 hospitals
- **Default location**: India (if no location specified)

## API Response Format

The response includes:
- Hospital names
- Addresses
- Distance/rating (if available)
- Tips for users

Example response:
```
Here are nearby hospitals in Bangalore:

1. Apollo Hospital
   📍 Bannerghatta Road, Bangalore
   ⭐ Rating: 4.5/5

2. Fortis Hospital
   📍 Cunningham Road, Bangalore
   ⭐ Rating: 4.3/5

💡 Tip: Call ahead to confirm availability and services. 
For emergencies, call your local emergency number immediately.
```

## Privacy & Security

- **No tracking**: Location data is only used for the current query
- **No storage**: User locations are not stored or logged
- **User control**: Users can specify location or let the system extract it from their message

## Limitations

1. **OpenStreetMap coverage**: Results depend on OSM data quality in your area
2. **No real-time availability**: Does not show if hospitals are currently open
3. **No appointment booking**: Only provides location information

## Future Enhancements

- Integration with hospital APIs for real-time availability
- Filter by hospital type (general, specialty, emergency)
- Directions/maps integration
- Multi-language support for location names
- Caching for frequently searched locations

## Troubleshooting

### No hospitals found

- Try specifying a larger city name
- Check if location name is spelled correctly
- Try using coordinates instead: "lat,lon"

### Slow responses

- OpenStreetMap API has rate limits (1 request per second)
- Consider using Google Places API for better performance
- Results are cached per session

## Code Structure

- **Location Service**: `app/integrations/location_service.py`
- **Dialogue Manager**: Updated to handle "location" action
- **API Routes**: Accepts `location` parameter in chat requests
