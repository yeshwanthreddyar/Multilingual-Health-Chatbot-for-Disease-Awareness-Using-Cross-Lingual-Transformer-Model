"""
Location Service - Find nearby hospitals and healthcare facilities.

Design goals:
- Prefer high-quality nearby results for "near me" queries (lat/lon).
- Provide text output only (no embedded maps UI), but include clickable links.
- Return phone/contact when available (Google Place Details).

Data sources:
- Google Places API (optional, requires GOOGLE_MAPS_API_KEY) — best quality + phone numbers.
- OpenStreetMap Overpass API (fallback, no key) — better than Nominatim for radius queries.
"""
from __future__ import annotations

import math
import re
import requests
from typing import List, Dict, Optional, Tuple
import os

# OpenStreetMap Nominatim API (free, no API key required)
NOMINATIM_BASE_URL = "https://nominatim.openstreetmap.org/search"

# Google Maps API (optional, requires API key)
GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY", "")
GOOGLE_PLACES_API_URL = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
GOOGLE_PLACE_DETAILS_API_URL = "https://maps.googleapis.com/maps/api/place/details/json"

# Overpass API (OpenStreetMap) — good for radius queries without a key.
# Public endpoints can rate-limit or time out, so we keep a small fallback list.
OVERPASS_API_URLS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.nchc.org.tw/api/interpreter",
]


def _parse_lat_lon(location: str) -> Optional[Tuple[float, float]]:
    if not location or not location.strip() or "," not in location:
        return None
    parts = [p.strip() for p in location.split(",", 1)]
    if len(parts) != 2:
        return None
    try:
        lat = float(parts[0])
        lon = float(parts[1])
        if not (-90 <= lat <= 90 and -180 <= lon <= 180):
            return None
        return (lat, lon)
    except Exception:
        return None


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    # Great-circle distance for small radius sorting/display
    r = 6371.0
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def geocode_location(location: str) -> Optional[Tuple[float, float]]:
    """
    Geocode a location string to latitude/longitude.
    Returns (lat, lon) or None if not found.
    """
    if not location or not location.strip():
        return None

    # If already coordinates, return them
    coords = _parse_lat_lon(location)
    if coords:
        return coords
    
    try:
        params = {
            "q": location,
            "format": "json",
            "limit": 1,
            "addressdetails": 1
        }
        headers = {
            "User-Agent": "HealthBot/1.0"  # Required by Nominatim
        }
        
        response = requests.get(NOMINATIM_BASE_URL, params=params, headers=headers, timeout=5)
        response.raise_for_status()
        data = response.json()
        
        if data and len(data) > 0:
            result = data[0]
            lat = float(result.get("lat", 0))
            lon = float(result.get("lon", 0))
            return (lat, lon)
    except Exception as e:
        print(f"Geocoding error: {e}")
    
    return None


def find_nearby_hospitals(
    location: str,
    radius_km: int = 10,
    limit: int = 5
) -> List[Dict[str, any]]:
    """
    Find nearby hospitals using OpenStreetMap.
    
    Args:
        location: City name, address, or "lat,lon" coordinates
        radius_km: Search radius in kilometers
        limit: Maximum number of results
    
    Returns:
        List of hospital dictionaries with name, address, distance, etc.
    """
    hospitals = []
    
    coords = geocode_location(location)
    if not coords:
        return hospitals
    lat, lon = coords

    # Prefer Google (best results + phone numbers), if configured
    if GOOGLE_MAPS_API_KEY:
        hospitals = _find_hospitals_google(lat, lon, radius_km, limit)
        if hospitals:
            return hospitals
    
    # Fallback: Overpass (OSM) radius query (more correct than Nominatim for nearby search)
    radius_m = int(max(1, radius_km) * 1000)
    # Query hospitals + clinics around coordinates
    query = f"""
[out:json][timeout:10];
(
  node["amenity"="hospital"](around:{radius_m},{lat},{lon});
  way["amenity"="hospital"](around:{radius_m},{lat},{lon});
  relation["amenity"="hospital"](around:{radius_m},{lat},{lon});
  node["amenity"="clinic"](around:{radius_m},{lat},{lon});
  way["amenity"="clinic"](around:{radius_m},{lat},{lon});
  relation["amenity"="clinic"](around:{radius_m},{lat},{lon});
);
out center {max(limit * 3, 20)};
"""
    def _parse_overpass_elements(elements: list) -> List[Dict[str, any]]:
        out: List[Dict[str, any]] = []
        for el in elements or []:
            tags = el.get("tags") or {}
            name = tags.get("name") or tags.get("operator") or "Hospital"
            # center lat/lon for ways/relations
            el_lat = el.get("lat") or (el.get("center") or {}).get("lat")
            el_lon = el.get("lon") or (el.get("center") or {}).get("lon")
            if el_lat is None or el_lon is None:
                continue
            try:
                el_lat_f = float(el_lat)
                el_lon_f = float(el_lon)
            except Exception:
                continue
            phone = tags.get("phone") or tags.get("contact:phone")
            website = tags.get("website") or tags.get("contact:website")
            street = tags.get("addr:street")
            city = tags.get("addr:city")
            postcode = tags.get("addr:postcode")
            parts = [p for p in [street, city, postcode] if p]
            address = ", ".join(parts) if parts else (tags.get("addr:full") or "")
            dist_km = _haversine_km(lat, lon, el_lat_f, el_lon_f)
            osm_url = None
            el_type = el.get("type")
            el_id = el.get("id")
            if el_type and el_id:
                osm_url = f"https://www.openstreetmap.org/{el_type}/{el_id}"
            out.append(
                {
                    "name": name,
                    "address": address,
                    "latitude": el_lat_f,
                    "longitude": el_lon_f,
                    "type": tags.get("amenity") or "hospital",
                    "phone": phone,
                    "website": website,
                    "distance_km": dist_km,
                    "osm_url": osm_url,
                }
            )
        out.sort(key=lambda h: h.get("distance_km", 999999))
        return out

    last_err: Optional[Exception] = None
    for overpass_url in OVERPASS_API_URLS:
        try:
            resp = requests.post(
                overpass_url,
                data=query.encode("utf-8"),
                headers={"User-Agent": "HealthBot/1.0"},
                timeout=20,
            )
            resp.raise_for_status()
            data = resp.json()
            parsed = _parse_overpass_elements((data or {}).get("elements", []) or [])
            if parsed:
                hospitals = parsed[:limit]
                break
        except Exception as e:
            last_err = e
            continue
    if not hospitals and last_err:
        print(f"Overpass error: {last_err}")
    
    return hospitals


def _find_hospitals_google(
    lat: float,
    lon: float,
    radius_km: int,
    limit: int
) -> List[Dict[str, any]]:
    """Find hospitals using Google Places API (requires API key)."""
    hospitals = []
    
    if not GOOGLE_MAPS_API_KEY:
        return hospitals
    
    try:
        params = {
            "location": f"{lat},{lon}",
            "radius": radius_km * 1000,
            "type": "hospital",
            "key": GOOGLE_MAPS_API_KEY
        }
        
        response = requests.get(GOOGLE_PLACES_API_URL, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if data.get("status") == "OK":
            for place in data.get("results", [])[: max(limit * 2, limit)]:
                hospital = {
                    "name": place.get("name", "Hospital"),
                    "address": place.get("vicinity", "") or place.get("formatted_address", ""),
                    "latitude": place["geometry"]["location"]["lat"],
                    "longitude": place["geometry"]["location"]["lng"],
                    "rating": place.get("rating"),
                    "type": "hospital",
                    "place_id": place.get("place_id"),
                }
                # Add a stable Google Maps link without embedding a map UI
                if hospital.get("place_id"):
                    hospital["maps_url"] = f"https://www.google.com/maps/place/?q=place_id:{hospital['place_id']}"
                else:
                    q = requests.utils.quote(hospital["name"])
                    hospital["maps_url"] = f"https://www.google.com/maps/search/?api=1&query={q}&query_place_id="

                hospitals.append(hospital)

            # Fetch phone/contact details for top results
            enriched: List[Dict[str, any]] = []
            for h in hospitals:
                pid = h.get("place_id")
                if not pid:
                    enriched.append(h)
                    continue
                details = _google_place_details(pid)
                if details:
                    h["phone"] = details.get("formatted_phone_number") or details.get("international_phone_number")
                    h["website"] = details.get("website")
                    # Google may return a canonical maps URL too
                    h["maps_url"] = details.get("url") or h.get("maps_url")
                    h["address"] = details.get("formatted_address") or h.get("address")
                enriched.append(h)

            # Sort by distance from user
            for h in enriched:
                try:
                    h["distance_km"] = _haversine_km(lat, lon, float(h["latitude"]), float(h["longitude"]))
                except Exception:
                    pass
            enriched.sort(key=lambda x: x.get("distance_km", 999999))
            hospitals = enriched[:limit]
    except Exception as e:
        print(f"Google Places API error: {e}")
    
    return hospitals


def _google_place_details(place_id: str) -> Optional[Dict[str, any]]:
    if not GOOGLE_MAPS_API_KEY or not place_id:
        return None
    try:
        params = {
            "place_id": place_id,
            "fields": "name,formatted_address,formatted_phone_number,international_phone_number,website,url",
            "key": GOOGLE_MAPS_API_KEY,
        }
        r = requests.get(GOOGLE_PLACE_DETAILS_API_URL, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
        if data.get("status") != "OK":
            return None
        return data.get("result") or None
    except Exception as e:
        print(f"Google Place Details error: {e}")
        return None


def format_hospital_response(hospitals: List[Dict[str, any]], location: str) -> str:
    """Format hospital list into a user-friendly response."""
    if not hospitals:
        return (
            f"I couldn't find hospitals near '{location}'. "
            "Please try again, or type your city/area. You can also open Google Maps search for hospitals near you: "
            "https://www.google.com/maps/search/hospital+near+me "
        )
    
    response_parts = [f"Here are nearby hospitals in {location}:\n\n"]
    
    for i, hospital in enumerate(hospitals, 1):
        name = hospital.get("name", "Hospital")
        address = hospital.get("address", "")
        rating = hospital.get("rating")
        phone = hospital.get("phone")
        maps_url = hospital.get("maps_url")
        if not maps_url:
            # Always provide a link (even for OSM fallback) without embedding a map
            lat = hospital.get("latitude")
            lon = hospital.get("longitude")
            if lat is not None and lon is not None:
                maps_url = f"https://www.google.com/maps/search/?api=1&query={lat},{lon}"
        distance_km = hospital.get("distance_km")
        
        response_parts.append(f"{i}. {name}")
        if address:
            # Truncate long addresses
            short_address = address.split(",")[0] if len(address) > 60 else address
            response_parts.append(f"   Address: {short_address}")
        if isinstance(distance_km, (int, float)):
            response_parts.append(f"   Distance: {distance_km:.1f} km")
        if rating:
            response_parts.append(f"   ⭐ Rating: {rating}/5")
        if phone:
            response_parts.append(f"   Phone: {phone}")
        if maps_url:
            response_parts.append(f"   Link: {maps_url}")
        response_parts.append("")
    
    response_parts.append(
        "Tip: Call ahead to confirm availability and services. "
        "For emergencies, call your local emergency number immediately."
    )
    
    return "\n".join(response_parts)


def extract_location_from_message(message: str) -> Optional[str]:
    """
    Extract location from user message.
    Looks for patterns like "near me", "in [city]", "near [location]"
    """
    message_lower = message.lower()

    # Coordinates like "12.9716,77.5946"
    m = re.search(r"(-?\d{1,2}\.\d+)\s*,\s*(-?\d{1,3}\.\d+)", message_lower)
    if m:
        return f"{m.group(1)},{m.group(2)}"
    
    # Common location patterns
    location_keywords = [
        "near me", "nearby", "close to me", "around me",
        "in ", "at ", "near ", "around ", "close to "
    ]
    
    # Check for "near me" or "nearby"
    if "near me" in message_lower or "nearby" in message_lower:
        return "current_location"  # Signal to use provided coordinates
    
    # Try to extract location after keywords
    for keyword in ["in ", "near ", "at ", "around ", "close to "]:
        if keyword in message_lower:
            # Extract text after keyword
            parts = message_lower.split(keyword, 1)
            if len(parts) > 1:
                location = parts[1].split()[0:3]  # Take first 1-3 words
                return " ".join(location).strip(".,!?")
    
    return None
