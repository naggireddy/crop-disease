"""
Geo-Fenced Agronomic Treatment Engine
Queries structured treatment rules by crop, disease, growth stage, and geographic coordinates/region.
Guarantees that only verified agronomic prescriptions are returned.
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Union

class TreatmentEngine:
    """
    Search and rule-evaluation engine for verified agronomic treatment protocols.
    Supports growth stage filtering and geographic bounding-box validation.
    """
    def __init__(self, rules_file: Optional[Path] = None):
        base_dir = Path(__file__).resolve().parent
        self.rules_file = rules_file or (base_dir / "treatment_rules.json")
        self.rules: List[Dict[str, Any]] = self._load_rules()

    def _load_rules(self) -> List[Dict[str, Any]]:
        """Loads and indexes treatment rules."""
        if not self.rules_file.exists():
            print(f"[!] Treatment rules database not found at: {self.rules_file}")
            return []
        try:
            with open(self.rules_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data.get("rules", [])
        except Exception as e:
            print(f"[!] Error loading treatment rules: {e}")
            return []

    def is_geo_applicable(self, rule: Dict[str, Any], latitude: Optional[float], longitude: Optional[float], region: Optional[str]) -> bool:
        """Determines if a rule applies to the specified coordinates or region name."""
        rule_region = rule.get("region", "Global")
        if rule_region.lower() == "global":
            return True

        # Check region string match
        if region and (region.lower() in rule_region.lower() or rule_region.lower() in region.lower()):
            return True

        # Check GPS coordinate bounding box
        lat_range = rule.get("lat_range")
        lon_range = rule.get("lon_range")
        if latitude is not None and longitude is not None and lat_range and lon_range:
            min_lat, max_lat = lat_range
            min_lon, max_lon = lon_range
            if min_lat <= latitude <= max_lat and min_lon <= longitude <= max_lon:
                return True

        return False

    def is_stage_applicable(self, rule_stage: str, query_stage: Optional[str]) -> bool:
        """Determines if growth stage matches."""
        if not query_stage or query_stage.lower() in ("unknown", "any", "general"):
            return True
        r_stage = rule_stage.lower()
        q_stage = query_stage.lower()
        return r_stage in ("any", "general") or r_stage == q_stage

    def lookup(
        self,
        crop: str,
        disease: str,
        growth_stage: Optional[str] = "vegetative",
        latitude: Optional[float] = None,
        longitude: Optional[float] = None,
        region: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Looks up verified treatment rule for given crop, disease, growth stage, and location.
        Returns verified rule or conservative fallback notice.
        """
        crop_clean = crop.lower().strip()
        disease_clean = disease.lower().strip()

        # Handle 'healthy' prediction
        if disease_clean == "healthy":
            return {
                "matched": True,
                "crop": crop_clean,
                "disease": "healthy",
                "recommended_action": "Crop foliage exhibits healthy physiological characteristics. Maintain standard irrigation and nutrient management.",
                "dosage": "N/A - No chemical or curative intervention required.",
                "organic_alternative": "Routine preventative foliar biostimulants (e.g. seaweed extract) or compost tea.",
                "safety_notes": "Continue regular pest scouting every 7-10 days.",
                "approval_status": "verified",
                "verified_by": "Automated Agronomic Advisory Standard",
                "is_fallback": False
            }

        # Search through rules database
        candidates = []
        for r in self.rules:
            # Check crop and disease
            if r.get("crop", "").lower() != crop_clean or r.get("disease", "").lower() != disease_clean:
                continue

            # Check verification status - strictly require verified
            if r.get("approval_status") != "verified":
                continue

            # Check growth stage
            if not self.is_stage_applicable(r.get("growth_stage", "any"), growth_stage):
                continue

            # Check geo-fencing
            if not self.is_geo_applicable(r, latitude, longitude, region):
                continue

            candidates.append(r)

        if candidates:
            # Prefer specific stage over 'any', and local region over 'Global'
            candidates.sort(key=lambda x: (
                0 if x.get("growth_stage", "").lower() == (growth_stage or "").lower() else 1,
                0 if x.get("region", "").lower() != "global" else 1
            ))
            selected = candidates[0]
            return {
                "matched": True,
                "rule_id": selected.get("rule_id"),
                "crop": selected.get("crop"),
                "disease": selected.get("disease"),
                "growth_stage": selected.get("growth_stage"),
                "region": selected.get("region"),
                "country": selected.get("country"),
                "recommended_action": selected.get("recommended_action"),
                "dosage": selected.get("dosage"),
                "active_ingredient": selected.get("active_ingredient"),
                "organic_alternative": selected.get("organic_alternative"),
                "safety_notes": selected.get("safety_notes"),
                "verified_by": selected.get("verified_by"),
                "verification_date": selected.get("verification_date"),
                "approval_status": selected.get("approval_status"),
                "is_fallback": False
            }

        # Conservative fallback when no verified rule matches
        return {
            "matched": False,
            "crop": crop_clean,
            "disease": disease_clean,
            "growth_stage": growth_stage,
            "message": "No verified treatment rule available for this crop, disease, growth stage, and region.",
            "recommended_action": "Do not apply unverified chemical compounds. Isolate affected leaves and submit image and field history to local agricultural extension agronomist for verified diagnosis.",
            "dosage": "None - Pending local agronomist review.",
            "approval_status": "unverified",
            "is_fallback": True
        }

if __name__ == "__main__":
    engine = TreatmentEngine()
    print("Testing TreatmentEngine for tomato early blight in South Asia...")
    result = engine.lookup("tomato", "early_blight", growth_stage="vegetative", latitude=17.385, longitude=78.486)
    print("Result:", json.dumps(result, indent=2))

