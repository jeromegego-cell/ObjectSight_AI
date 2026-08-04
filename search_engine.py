"""
ObjectSight AI - Internet Search & Knowledge Engine
Fetches rich details, Wikipedia summaries, specifications, images, and facts for detected objects.
"""

import requests
import re
import urllib.parse

try:
    import wikipedia
except ImportError:
    wikipedia = None


def search_object_details(query_term):
    """
    Given an object name/label, query Wikipedia and DuckDuckGo for comprehensive details.
    """
    query_clean = query_term.strip()
    if not query_clean:
        return {"error": "Empty search query"}

    result = {
        "query": query_clean,
        "title": query_clean.title(),
        "summary": "",
        "category": "General Object",
        "key_facts": [],
        "specifications": {},
        "uses": [],
        "image_url": "",
        "source_url": f"https://en.wikipedia.org/wiki/{urllib.parse.quote(query_clean)}",
        "related_topics": [],
        "duckduckgo_abstract": ""
    }

    # 1. Wikipedia Fetching
    try:
        if wikipedia:
            wikipedia.set_lang("en")
            # Search for pages matching query
            search_results = wikipedia.search(query_clean)
            page_title = search_results[0] if search_results else query_clean
            
            try:
                page = wikipedia.page(page_title, auto_suggest=True)
                result["title"] = page.title
                result["summary"] = page.summary[:800] + ("..." if len(page.summary) > 800 else "")
                result["source_url"] = page.url
                
                # Fetch page main image if available
                if page.images:
                    valid_imgs = [img for img in page.images if not img.endswith('.svg') and not img.endswith('.gif')]
                    if valid_imgs:
                        result["image_url"] = valid_imgs[0]
                        
                # Extract key categories
                if hasattr(page, 'categories') and page.categories:
                    cat_clean = [c.replace("Category:", "").strip() for c in page.categories[:5] if not c.startswith("Category:Articles")]
                    if cat_clean:
                        result["category"] = cat_clean[0]
            except wikipedia.exceptions.DisambiguationError as e:
                # Pick first option
                if e.options:
                    try:
                        page = wikipedia.page(e.options[0])
                        result["title"] = page.title
                        result["summary"] = page.summary[:800] + ("..." if len(page.summary) > 800 else "")
                        result["source_url"] = page.url
                    except Exception:
                        pass
            except Exception as wiki_err:
                print(f"Wikipedia page fetch error: {wiki_err}")
    except Exception as e:
        print(f"Wikipedia search error: {e}")

    # Fallback / Direct API if wikipedia module summary missing
    if not result["summary"]:
        try:
            wiki_api_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{urllib.parse.quote(query_clean)}"
            headers = {'User-Agent': 'ObjectSightAI/1.0 (Robotics Project)'}
            resp = requests.get(wiki_api_url, headers=headers, timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                if 'extract' in data:
                    result["summary"] = data['extract']
                if 'title' in data:
                    result["title"] = data['title']
                if 'thumbnail' in data and 'source' in data['thumbnail']:
                    result["image_url"] = data['thumbnail']['source']
                if 'content_urls' in data and 'desktop' in data['content_urls']:
                    result["source_url"] = data['content_urls']['desktop']['page']
        except Exception as api_err:
            print(f"Wikipedia REST API error: {api_err}")

    # 2. DuckDuckGo Instant Answer API
    try:
        ddg_url = f"https://api.duckduckgo.com/?q={urllib.parse.quote(query_clean)}&format=json&pretty=1"
        resp = requests.get(ddg_url, timeout=5)
        if resp.status_code == 200:
            ddg_data = resp.json()
            if ddg_data.get("AbstractText"):
                result["duckduckgo_abstract"] = ddg_data["AbstractText"]
                if not result["summary"]:
                    result["summary"] = ddg_data["AbstractText"]
            
            if ddg_data.get("Image"):
                img = ddg_data["Image"]
                if img.startswith("/"):
                    img = "https://duckduckgo.com" + img
                if not result["image_url"]:
                    result["image_url"] = img
                    
            if ddg_data.get("RelatedTopics"):
                for topic in ddg_data["RelatedTopics"][:6]:
                    if isinstance(topic, dict) and "Text" in topic:
                        result["related_topics"].append(topic["Text"])
    except Exception as ddg_err:
        print(f"DuckDuckGo search error: {ddg_err}")

    # Default fallback summary if both APIs fail
    if not result["summary"]:
        result["summary"] = f"'{result['title']}' is a recognized object. Details are synthesized from optical object classification and standard taxonomy records."

    # 3. Generate Structured Specifications & Facts
    result["key_facts"] = generate_key_facts(query_clean, result["summary"])
    result["specifications"] = generate_specifications(query_clean, result["summary"])
    result["uses"] = generate_uses(query_clean, result["summary"])

    return result


def generate_key_facts(object_name, summary):
    facts = []
    # Extract sentences from summary
    sentences = re.split(r'\. |\n', summary)
    for s in sentences:
        s_clean = s.strip()
        if len(s_clean) > 20 and len(s_clean) < 180:
            facts.append(s_clean if s_clean.endswith('.') else s_clean + '.')
        if len(facts) >= 4:
            break

    if not facts:
        facts = [
            f"{object_name.title()} is a physical item commonly detected in indoor and outdoor visual environments.",
            f"Machine vision models classify this item based on distinct surface features, geometric contours, and visual keypoints.",
            f"Frequently interacted with by humans and mobile robotics units for manipulation, tracking, or spatial navigation."
        ]
    return facts


def generate_specifications(object_name, summary):
    name_lower = object_name.lower()
    specs = {
        "Object Class": object_name.title(),
        "Domain": "Physical World Artifact",
        "Detection Model": "COCO Neural Network / MobileNet-SSD",
        "Visual Traits": "Distinct geometric edges, color boundaries, surface textures"
    }

    # Custom enriched domain knowledge for common robotics / household objects
    if any(k in name_lower for k in ["phone", "cell phone", "mobile"]):
        specs.update({
            "Primary Type": "Handheld Mobile Telecommunications Device",
            "Components": "Display, SoC Processor, Battery, Camera Sensors, Antennas",
            "Connectivity": "Cellular 5G/LTE, Wi-Fi, Bluetooth, NFC",
            "Power Source": "Lithium-Ion / Li-Polymer Rechargeable Battery"
        })
    elif any(k in name_lower for k in ["laptop", "computer"]):
        specs.update({
            "Primary Type": "Portable Personal Computer",
            "Components": "CPU, GPU, RAM, Solid-State Drive, Display Panel, Keyboard",
            "Operating Systems": "Linux, macOS, Windows",
            "Human Interface": "Keyboard, Trackpad, USB/HDMI Peripherals"
        })
    elif any(k in name_lower for k in ["cup", "mug", "bottle", "glass"]):
        specs.update({
            "Primary Type": "Liquid Storage Container / Beverage Ware",
            "Materials": "Ceramic, Glass, Stainless Steel, Polymer",
            "Thermal Properties": "Insulated / Heat Resistant",
            "Robotics Grip": "Cylindrical / Handle Pinch Grip"
        })
    elif any(k in name_lower for k in ["person", "human"]):
        specs.update({
            "Primary Type": "Human Being (Biological Agent)",
            "Kinematics": "Bipedal Locomotion",
            "Interaction Mode": "Voice Commands, Gesture Recognition, Vision Tracking",
            "Safety Protocol": "Level 1 Collaborative Robot Safety Stop Required"
        })
    elif any(k in name_lower for k in ["cat", "dog", "pet", "animal"]):
        specs.update({
            "Primary Type": "Biological Organism / Domestic Quadruped",
            "Robotic Note": "Dynamic unpredictable trajectory; autonomous collision avoidance active",
            "Thermal Signature": "Warm Body (~38°C)"
        })
    elif any(k in name_lower for k in ["car", "vehicle", "bus", "truck"]):
        specs.update({
            "Primary Type": "Motorized Automotive Transport",
            "Propulsion": "Internal Combustion Engine / Electric Drive",
            "Safety Distance": "Maintain minimum 2-meter buffer zone",
            "Autonomy Level": "SAE Level 0-5 Systems"
        })
    elif any(k in name_lower for k in ["chair", "couch", "sofa", "bench"]):
        specs.update({
            "Primary Type": "Furniture / Seating Apparatus",
            "Load Capacity": "80 kg - 250 kg Typical",
            "Robotic Navigation": "Static Spatial Obstacle"
        })
    elif any(k in name_lower for k in ["keyboard", "mouse"]):
        specs.update({
            "Primary Type": "Human Interface Device (HID)",
            "Protocol": "USB Human Interface / Bluetooth HID",
            "Robotic Interaction": "Haptic Tapping / Precision Kinematic End-Effector"
        })

    return specs


def generate_uses(object_name, summary):
    name_lower = object_name.lower()
    if any(k in name_lower for k in ["phone", "cell phone"]):
        return ["Personal Communication & Messaging", "Mobile Computing & Internet Navigation", "Digital Photography & Sensor Data Collection"]
    elif any(k in name_lower for k in ["laptop", "computer"]):
        return ["Software Engineering & Robotics Control", "Data Processing & AI Model Training", "Multimedia & Digital Operations"]
    elif any(k in name_lower for k in ["cup", "bottle"]):
        return ["Hydration & Beverage Consumption", "Liquid Storage & Dispensing", "Robotic Manipulation & Pick-and-Place Demos"]
    elif any(k in name_lower for k in ["book"]):
        return ["Knowledge Documentation & Reading", "Text Recognition & OCR Testing", "Educational Study"]
    else:
        return [
            f"Daily human activity and environmental utility.",
            f"Robotic scene segmentation, object pick-and-place manipulation.",
            f"Spatial mapping and object classification in automated environments."
        ]


if __name__ == "__main__":
    import json
    res = search_object_details("laptop")
    print(json.dumps(res, indent=2))
